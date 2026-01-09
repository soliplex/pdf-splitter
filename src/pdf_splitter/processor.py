"""
Processor Module (The Orchestrator)

Manages parallel workers and enforces memory cleanup through strict
process isolation using maxtasksperchild=1.

CRITICAL: DocumentConverter MUST be instantiated inside the worker function,
not passed from the main thread. This ensures the C++ backend is initialized
in the correct process space.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _process_chunk(chunk_path: str, verbose: bool = False) -> dict[str, Any]:
    """
    Worker function to process a single PDF chunk.

    IMPORTANT: This function instantiates DocumentConverter inside the worker
    to ensure proper C++ backend initialization and memory isolation.

    Args:
        chunk_path: Path to the chunk PDF file
        verbose: If True, enable INFO logging; if False, WARNING only

    Returns:
        Dict containing:
        - 'success': bool indicating if processing succeeded
        - 'chunk_path': original chunk path
        - 'document': serialized DoclingDocument (if success)
        - 'error': error message (if failed)
    """
    # Disable automatic garbage collection to prevent GC thrashing.
    # Docling creates millions of objects; Python's periodic GC triggers
    # mid-conversion and gets stuck traversing the massive object graph.
    # With maxtasksperchild=1, the process dies after one chunk anyway,
    # so letting the OS reclaim memory is faster than Python GC.
    start = time.time()

    import gc

    gc.disable()

    # Configure logging in worker process based on verbose flag
    import logging

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, force=True)
    for logger_name in ["docling", "docling_core", "docling_parse", "src"]:
        logging.getLogger(logger_name).setLevel(level)

    logger.info(f"start processing chunk {chunk_path} pid={os.getpid()}")
    # Import inside worker to ensure proper process isolation
    from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    try:
        # Create converter with image extraction enabled
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = True
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE  # type: ignore[attr-defined]
        pipeline_opts.generate_page_images = True
        pipeline_opts.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_opts, backend=DoclingParseV2DocumentBackend
                )
            }
        )

        # Process the chunk
        result = converter.convert(chunk_path)
        doc = result.document

        # Serialize document for cross-process transfer
        doc_json = doc.export_to_dict()
        proc_time = time.time() - start
        logger.info(
            f"end processing chunk {chunk_path} in {proc_time:.2f} seconds pid={os.getpid()}"
        )
        return {
            "success": True,
            "chunk_path": chunk_path,
            "document_dict": doc_json,
            "error": None,
            "proc_time": proc_time,
        }

    except Exception as e:
        proc_time = time.time() - start
        return {
            "success": False,
            "chunk_path": chunk_path,
            "document_dict": None,
            "error": str(e),
            "proc_time": proc_time,
        }


class BatchProcessor:
    """
    Orchestrates parallel PDF chunk processing with strict memory isolation.

    Uses ProcessPoolExecutor with maxtasksperchild=1 to force a hard reset
    of memory space after every chunk, preventing DoclingParseV2 memory leak.
    """

    def __init__(
        self, max_workers: int | None = None, maxtasksperchild: int = 1, verbose: bool = False
    ):
        """
        Initialize the batch processor.

        Args:
            max_workers: Maximum parallel workers (defaults to CPU count)
            maxtasksperchild: Tasks per worker before restart (default 1 for memory isolation)
            verbose: If True, enable INFO logging in workers; if False, WARNING only
        """
        # Default to 80% of CPUs to leave headroom for other processes
        self.max_workers = max_workers or max(1, int((os.cpu_count() or 4) * 0.8))
        self.maxtasksperchild = maxtasksperchild
        self.verbose = verbose

    def execute_parallel(
        self, chunk_paths: list[Path], ordered: bool = True
    ) -> list[dict[str, Any]]:
        """
        Process multiple PDF chunks in parallel.

        Args:
            chunk_paths: List of paths to PDF chunk files
            ordered: If True, results maintain input order

        Returns:
            List of result dicts from _process_chunk, in order if ordered=True
        """
        if not chunk_paths:
            return []

        chunk_str_paths = [str(p) for p in chunk_paths]
        results: list[dict[str, Any] | None] = [None] * len(chunk_paths)
        path_to_idx = {p: i for i, p in enumerate(chunk_str_paths)}

        logger.debug(
            f"Parallel processing: max_workers={self.max_workers}, "
            f"maxtasksperchild={self.maxtasksperchild}, chunks={len(chunk_paths)}"
        )

        completed_count = 0
        with ProcessPoolExecutor(
            max_workers=self.max_workers, max_tasks_per_child=self.maxtasksperchild
        ) as executor:
            # Submit all chunks
            future_to_path = {}
            for path in chunk_str_paths:
                idx = path_to_idx[path]
                logger.debug(
                    f"[BEGIN] Submitting chunk {idx + 1}/{len(chunk_paths)}: {Path(path).name}"
                )
                future = executor.submit(_process_chunk, path, self.verbose)
                future_to_path[future] = path

            logger.debug(f"All {len(chunk_paths)} chunks submitted to worker pool")

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                idx = path_to_idx[path]
                chunk_name = Path(path).name

                try:
                    result = future.result()
                    results[idx] = result
                    completed_count += 1

                    if result["success"]:
                        logger.debug(
                            f"Processed {completed_count}/{len(chunk_paths)}: {chunk_name}"
                        )
                    else:
                        logger.error(
                            f"[FAILED] Chunk {idx + 1}/{len(chunk_paths)}: {chunk_name} - {result['error']}"
                        )

                except Exception as e:
                    logger.error(
                        f"[EXCEPTION] Chunk {idx + 1}/{len(chunk_paths)}: {chunk_name} - {e}"
                    )
                    results[idx] = {
                        "success": False,
                        "chunk_path": path,
                        "document_dict": None,
                        "error": str(e),
                    }

        # Summary
        success_count = sum(1 for r in results if r and r.get("success"))
        fail_count = len(chunk_paths) - success_count
        logger.debug(f"Processing complete: {success_count} succeeded, {fail_count} failed")

        # Filter out None values for return
        valid_results: list[dict[str, Any]] = [r for r in results if r is not None]
        return valid_results

    def execute_sequential(self, chunk_paths: list[Path]) -> list[dict[str, Any]]:
        """
        Process chunks sequentially (useful for debugging or memory testing).

        Args:
            chunk_paths: List of paths to PDF chunk files

        Returns:
            List of result dicts from _process_chunk
        """
        results = []
        for i, chunk_path in enumerate(chunk_paths):
            logger.debug(f"Processing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
            result = _process_chunk(str(chunk_path), self.verbose)
            results.append(result)

            if not result["success"]:
                logger.error(f"Failed: {result['error']}")

        return results
