"""
Enhanced Segmentation Module

Provides smarter PDF splitting with:
1. Deep bookmark traversal to find actual chapter/section boundaries
2. Validation for balanced chunk distribution
3. Fallback to fixed splitting when bookmarks create uneven chunks
4. Optional Docling-based TOC extraction for complex documents
5. Unified smart_split() that auto-selects the best strategy
"""

import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CHUNK_SIZE = 50
DEFAULT_OVERLAP = 5
MAX_CHUNK_RATIO = 0.4  # Max fraction of doc a single chunk can be before rebalancing
MIN_CHUNKS_FOR_LARGE_DOC = 5  # Minimum chunks for docs > 200 pages


@dataclass
class SplitResult:
    """Result of smart_split() operation."""

    boundaries: list[tuple[int, int]]
    strategy: str
    total_pages: int
    num_chunks: int
    min_chunk_size: int
    max_chunk_size: int
    avg_chunk_size: float
    has_overlap: bool

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Strategy: {self.strategy}\n"
            f"Total pages: {self.total_pages}\n"
            f"Chunks: {self.num_chunks}\n"
            f"Chunk sizes: {self.min_chunk_size}-{self.max_chunk_size} pages "
            f"(avg {self.avg_chunk_size:.1f})"
        )


def smart_split(
    pdf_path: Path,
    max_chunk_pages: int = 100,
    min_chunk_pages: int = 15,
    overlap: int = DEFAULT_OVERLAP,
    force_strategy: str | None = None,
) -> SplitResult:
    """
    Unified smart splitting that auto-selects the best strategy.

    Strategy selection logic:
    1. For small documents (< max_chunk_pages): single chunk or fixed split
    2. For documents with good bookmark structure: hybrid (chapter + sections)
    3. For documents with poor/no bookmarks: fixed splitting with overlap

    Args:
        pdf_path: Path to the PDF file
        max_chunk_pages: Maximum pages per chunk (default 100)
        min_chunk_pages: Minimum pages per chunk (default 15)
        overlap: Overlap pages between chunks for fixed splitting (default 5)
        force_strategy: Force a specific strategy ('fixed', 'hybrid', 'enhanced')

    Returns:
        SplitResult with boundaries and metadata

    Raises:
        ValueError: If max_chunk_pages < 1 or overlap < 0
    """
    # Validate inputs
    if max_chunk_pages < 1:
        raise ValueError(f"max_chunk_pages must be >= 1, got {max_chunk_pages}")
    if min_chunk_pages < 1:
        raise ValueError(f"min_chunk_pages must be >= 1, got {min_chunk_pages}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    # Handle edge cases
    if total_pages == 0:
        logger.debug(f"Empty document: {pdf_path.name}")
        return SplitResult(
            boundaries=[],
            strategy="empty",
            total_pages=0,
            num_chunks=0,
            min_chunk_size=0,
            max_chunk_size=0,
            avg_chunk_size=0,
            has_overlap=False,
        )

    logger.debug(
        f"Selecting strategy for {total_pages} pages (max={max_chunk_pages}, min={min_chunk_pages})"
    )

    # Select strategy
    if force_strategy:
        boundaries, strategy = _apply_forced_strategy(
            pdf_path, total_pages, force_strategy, max_chunk_pages, min_chunk_pages, overlap
        )
    else:
        boundaries, strategy = _auto_select_strategy(
            pdf_path, reader, total_pages, max_chunk_pages, min_chunk_pages, overlap
        )

    # Calculate statistics
    sizes = [end - start for start, end in boundaries]
    has_overlap = _check_overlap(boundaries)

    return SplitResult(
        boundaries=boundaries,
        strategy=strategy,
        total_pages=total_pages,
        num_chunks=len(boundaries),
        min_chunk_size=min(sizes) if sizes else 0,
        max_chunk_size=max(sizes) if sizes else 0,
        avg_chunk_size=sum(sizes) / len(sizes) if sizes else 0,
        has_overlap=has_overlap,
    )


def _apply_forced_strategy(
    pdf_path: Path,
    total_pages: int,
    strategy: str,
    max_chunk_pages: int,
    min_chunk_pages: int,
    overlap: int,
) -> tuple[list[tuple[int, int]], str]:
    """Apply a specific forced strategy."""
    logger.info(f"Forcing strategy: {strategy}")
    if strategy == "fixed":
        return _get_fixed_boundaries(total_pages, max_chunk_pages, overlap), "fixed"
    elif strategy == "hybrid":
        return get_split_boundaries_hybrid(pdf_path, max_chunk_pages, min_chunk_pages, overlap)
    elif strategy == "enhanced":
        return get_split_boundaries_enhanced(pdf_path, max_chunk_pages, overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _auto_select_strategy(
    pdf_path: Path,
    reader: PdfReader,
    total_pages: int,
    max_chunk_pages: int,
    min_chunk_pages: int,
    overlap: int,
) -> tuple[list[tuple[int, int]], str]:
    """Auto-select the best strategy based on document characteristics."""

    # Small documents: use fixed splitting
    if total_pages <= max_chunk_pages:
        logger.debug(
            f"Small document ({total_pages} pages <= {max_chunk_pages}), using single chunk"
        )
        return [(0, total_pages)], "single_chunk"

    if total_pages <= max_chunk_pages * 2:
        logger.debug(f"Medium document ({total_pages} pages), using fixed_small strategy")
        return _get_fixed_boundaries(total_pages, max_chunk_pages, overlap), "fixed_small"

    # Check for bookmarks
    if not reader.outline:
        logger.debug("No bookmarks found, using fixed splitting")
        return _get_fixed_boundaries(total_pages, max_chunk_pages, overlap), "fixed_no_bookmarks"

    # Analyze bookmark structure
    all_bookmarks: list[tuple[int, int, str]] = []
    _collect_bookmarks_recursive(reader, reader.outline, all_bookmarks, level=0)
    logger.debug(f"Found {len(all_bookmarks)} bookmarks")

    if len(all_bookmarks) < 3:
        logger.debug(f"Too few bookmarks ({len(all_bookmarks)}), using fixed splitting")
        return _get_fixed_boundaries(total_pages, max_chunk_pages, overlap), "fixed_few_bookmarks"

    # Check for chapter-level structure
    has_chapters = any(
        "CHAPTER" in title.upper() or "PART" in title.upper() for _, _, title in all_bookmarks
    )
    logger.debug(f"Chapter-level structure detected: {has_chapters}")

    if has_chapters:
        # Use hybrid strategy for documents with chapter structure
        logger.debug("Trying hybrid strategy for chapter-based document")
        boundaries, strategy = get_split_boundaries_hybrid(
            pdf_path, max_chunk_pages, min_chunk_pages, overlap
        )
        if _is_balanced(boundaries, total_pages, MAX_CHUNK_RATIO):
            logger.info(f"Selected hybrid strategy: {strategy} ({len(boundaries)} chunks)")
            return boundaries, f"auto_{strategy}"
        logger.debug("Hybrid strategy unbalanced, trying enhanced")

    # Try enhanced strategy
    logger.debug("Trying enhanced bookmark strategy")
    boundaries, strategy = get_split_boundaries_enhanced(
        pdf_path, max_chunk_pages, overlap, max_chunk_ratio=MAX_CHUNK_RATIO
    )

    if _is_balanced(boundaries, total_pages, MAX_CHUNK_RATIO):
        logger.info(f"Selected enhanced strategy: {strategy} ({len(boundaries)} chunks)")
        return boundaries, f"auto_{strategy}"

    # Final fallback: fixed splitting
    logger.info(
        f"Falling back to fixed splitting ({total_pages} pages, chunk_size={max_chunk_pages})"
    )
    return _get_fixed_boundaries(total_pages, max_chunk_pages, overlap), "fixed_fallback"


def _check_overlap(boundaries: list[tuple[int, int]]) -> bool:
    """Check if boundaries have overlapping regions."""
    return any(boundaries[i][1] > boundaries[i + 1][0] for i in range(len(boundaries) - 1))


def _write_single_chunk(
    reader_path: str, start: int, end: int, idx: int, total_chunks: int, output_dir_str: str
) -> tuple[int, str | None, str | None]:
    """
    Write a single chunk to disk. Designed for parallel execution.

    Args:
        reader_path: Path to the source PDF (string for pickling)
        start: Start page (0-indexed)
        end: End page (exclusive)
        idx: Chunk index
        total_chunks: Total number of chunks
        output_dir_str: Output directory as string (for cross-process pickling)

    Returns:
        Tuple of (idx, chunk_path_str or None, error_message or None)
    """
    # Disable GC in worker to prevent GC thrashing on large PDFs.
    # The process exits after this function returns anyway.
    import gc

    gc.disable()

    try:
        reader = PdfReader(reader_path)
        writer = PdfWriter()

        for page_num in range(start, end):
            page = reader.pages[page_num]
            # Remove annotations (links, bookmarks) to avoid slow _resolve_links
            # during write. Docling doesn't need internal PDF links.
            if "/Annots" in page:
                del page["/Annots"]
            writer.add_page(page)

        chunk_filename = f"chunk_{idx:04d}_pages_{start + 1:04d}_{end:04d}.pdf"
        chunk_path = Path(output_dir_str) / chunk_filename

        with open(chunk_path, "wb") as f:
            writer.write(f)

        # Return string path for clean cross-process serialization
        return (idx, str(chunk_path), None)

    except Exception as e:
        return (idx, None, str(e))


def smart_split_to_files(
    pdf_path: Path,
    output_dir: Path | None = None,
    max_chunk_pages: int = 100,
    min_chunk_pages: int = 15,
    overlap: int = DEFAULT_OVERLAP,
    force_strategy: str | None = None,
    max_workers: int | None = None,
    parallel: bool = True,
) -> tuple[list[Path], SplitResult]:
    """
    Split a PDF into chunk files using smart strategy selection.

    Args:
        pdf_path: Path to the source PDF
        output_dir: Directory to write chunks (uses temp dir if None)
        max_chunk_pages: Maximum pages per chunk
        min_chunk_pages: Minimum pages per chunk
        overlap: Overlap pages between chunks
        force_strategy: Force a specific strategy
        max_workers: Maximum parallel workers for writing (defaults to CPU count)
        parallel: If True, write chunks in parallel; if False, write sequentially

    Returns:
        Tuple of (list of chunk file paths, SplitResult metadata)
    """
    pdf_path = Path(pdf_path)
    logger.debug(f"Starting smart split: {pdf_path.name}")

    # Get split boundaries
    result = smart_split(pdf_path, max_chunk_pages, min_chunk_pages, overlap, force_strategy)

    if not result.boundaries:
        logger.info("No boundaries generated (empty document)")
        return [], result

    # Create output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="pdf_chunks_"))
        logger.debug(f"Created temp output directory: {output_dir}")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using output directory: {output_dir}")

    total_chunks = len(result.boundaries)

    # Determine worker count (default 80% of CPUs to leave headroom)
    if max_workers is None:
        default_workers = max(1, int((os.cpu_count() or 4) * 0.8))
        max_workers = min(default_workers, total_chunks)

    if parallel and total_chunks > 1:
        chunk_paths = _write_chunks_parallel(
            pdf_path, result.boundaries, output_dir, max_workers, total_chunks
        )
    else:
        chunk_paths = _write_chunks_sequential(
            pdf_path, result.boundaries, output_dir, total_chunks
        )

    logger.debug(f"Split complete: {total_chunks} chunks written to {output_dir}")
    return chunk_paths, result


def _write_chunks_parallel(
    pdf_path: Path,
    boundaries: list[tuple[int, int]],
    output_dir: Path,
    max_workers: int,
    total_chunks: int,
) -> list[Path]:
    """Write chunks in parallel using ProcessPoolExecutor.

    Uses processes instead of threads because pypdf is pure Python and
    CPU-bound. The GIL would serialize thread execution, making ThreadPoolExecutor
    ineffective. ProcessPoolExecutor bypasses the GIL for true parallelism.
    """
    logger.debug(f"Parallel chunk writing: {max_workers} processes, {total_chunks} chunks")

    chunk_paths: list[Path | None] = [None] * total_chunks
    completed_count = 0

    # Convert to strings for cross-process pickling
    pdf_path_str = str(pdf_path)
    output_dir_str = str(output_dir)

    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        # Submit all chunk writes
        futures = {}
        for idx, (start, end) in enumerate(boundaries):
            logger.debug(f"[BEGIN] Submitting chunk {idx + 1}/{total_chunks} to write pool")
            future = executor.submit(
                _write_single_chunk, pdf_path_str, start, end, idx, total_chunks, output_dir_str
            )
            futures[future] = idx

        logger.debug(f"All {total_chunks} chunks submitted to write pool")

        # Collect results as they complete
        for future in as_completed(futures):
            idx, chunk_path_str, error = future.result()
            completed_count += 1

            if error:
                logger.error(f"Failed to write chunk {idx + 1}: {error}")
            elif chunk_path_str is not None:
                chunk_path = Path(chunk_path_str)
                chunk_paths[idx] = chunk_path
                logger.debug(f"Wrote {completed_count}/{total_chunks}: {chunk_path.name}")

        # Clear futures to release references before shutdown
        futures.clear()
    finally:
        # Shutdown without waiting - workers completed, let OS clean up
        executor.shutdown(wait=False, cancel_futures=True)

    # Filter out any failed chunks
    valid_paths = [p for p in chunk_paths if p is not None]

    success_count = len(valid_paths)
    fail_count = total_chunks - success_count
    logger.debug(f"Parallel write complete: {success_count} succeeded, {fail_count} failed")

    return valid_paths


def _write_chunks_sequential(
    pdf_path: Path, boundaries: list[tuple[int, int]], output_dir: Path, total_chunks: int
) -> list[Path]:
    """Write chunks sequentially (original behavior)."""
    logger.debug(f"Sequential chunk writing: {total_chunks} chunks")

    reader = PdfReader(str(pdf_path))
    chunk_paths = []

    logger.debug(f"Writing {total_chunks} chunks sequentially")

    for idx, (start, end) in enumerate(boundaries):
        writer = PdfWriter()
        num_pages = end - start

        for page_num in range(start, end):
            page = reader.pages[page_num]
            # Remove annotations to avoid slow _resolve_links during write
            if "/Annots" in page:
                del page["/Annots"]
            writer.add_page(page)

        chunk_filename = f"chunk_{idx:04d}_pages_{start + 1:04d}_{end:04d}.pdf"
        chunk_path = output_dir / chunk_filename

        logger.debug(
            f"[BEGIN] Writing chunk {idx + 1}/{total_chunks}: pages {start + 1}-{end} ({num_pages} pages)"
        )
        with open(chunk_path, "wb") as f:
            writer.write(f)

        chunk_paths.append(chunk_path)
        logger.debug(f"[COMPLETE] Chunk {idx + 1}/{total_chunks}: {chunk_filename}")
        logger.debug(f"Wrote chunk {idx + 1}/{total_chunks}: {chunk_filename}")

    return chunk_paths


def get_split_boundaries_enhanced(
    pdf_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    target_level: int | None = None,
    max_chunk_ratio: float = MAX_CHUNK_RATIO,
) -> tuple[list[tuple[int, int]], str]:
    """
    Enhanced split boundary detection with multiple strategies.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Pages per chunk for fixed splitting
        overlap: Overlap pages between chunks
        target_level: Specific bookmark depth to use (None = auto-detect)
        max_chunk_ratio: Maximum fraction of document for any single chunk

    Returns:
        Tuple of (boundaries, strategy_used)
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    logger.debug(f"Enhanced strategy: analyzing {total_pages} pages")

    if total_pages == 0:
        return [], "empty"

    if total_pages == 1:
        logger.debug("Single page document")
        return [(0, 1)], "single_page"

    # Try deep bookmark extraction
    if reader.outline:
        logger.debug("Document has outline, attempting deep bookmark extraction")
        boundaries, level_used = _get_deep_bookmark_boundaries(reader, total_pages, target_level)

        if boundaries and _is_balanced(boundaries, total_pages, max_chunk_ratio):
            logger.info(f"Using bookmark level {level_used}: {len(boundaries)} balanced chunks")
            return boundaries, f"bookmark_level_{level_used}"

        if boundaries:
            logger.debug(
                f"Bookmark boundaries unbalanced ({len(boundaries)} chunks), trying rebalancing"
            )
            # Try rebalancing large chunks
            rebalanced = _rebalance_chunks(boundaries, total_pages, chunk_size, overlap)
            if rebalanced and _is_balanced(rebalanced, total_pages, max_chunk_ratio):
                logger.info(f"Rebalanced to {len(rebalanced)} chunks")
                return rebalanced, "bookmark_rebalanced"
            logger.debug("Rebalancing did not produce balanced chunks")
    else:
        logger.debug("No outline found in document")

    # Fallback to fixed splitting
    logger.debug(f"Using fixed splitting (chunk_size={chunk_size}, overlap={overlap})")
    boundaries = _get_fixed_boundaries(total_pages, chunk_size, overlap)
    return boundaries, "fixed"


def _get_deep_bookmark_boundaries(
    reader: PdfReader, total_pages: int, target_level: int | None = None
) -> tuple[list[tuple[int, int]], int]:
    """
    Extract boundaries by traversing bookmark tree deeply.

    Finds the optimal level that provides meaningful chapter/section splits.
    """
    # Collect all bookmarks with their levels
    all_bookmarks: list[tuple[int, int, str]] = []
    _collect_bookmarks_recursive(reader, reader.outline, all_bookmarks, level=0)
    logger.debug(f"Deep bookmark scan: found {len(all_bookmarks)} total bookmarks")

    if not all_bookmarks:
        logger.debug("No bookmarks collected")
        return [], -1

    # Group by level
    by_level: dict[int, list[int]] = {}
    for page, level, _title in all_bookmarks:
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(page)

    logger.debug(
        f"Bookmark levels found: {sorted(by_level.keys())} with counts {[len(by_level[k]) for k in sorted(by_level.keys())]}"
    )

    # Find the best level
    if target_level is not None and target_level in by_level:
        best_level = target_level
        logger.debug(f"Using specified target level: {target_level}")
    else:
        best_level = _find_optimal_level(by_level, total_pages)
        logger.debug(f"Auto-selected optimal level: {best_level}")

    if best_level < 0:
        logger.debug("No suitable bookmark level found")
        return [], -1

    # Build boundaries from the chosen level
    page_numbers = sorted(set(by_level[best_level]))
    logger.debug(f"Level {best_level} has {len(page_numbers)} unique page boundaries")

    # Ensure we start from page 0
    if page_numbers[0] != 0:
        page_numbers.insert(0, 0)

    boundaries = []
    for i in range(len(page_numbers)):
        start = page_numbers[i]
        end = page_numbers[i + 1] if i + 1 < len(page_numbers) else total_pages
        if start < end:
            boundaries.append((start, end))

    logger.debug(f"Generated {len(boundaries)} boundaries from level {best_level}")
    return boundaries, best_level


def _collect_bookmarks_recursive(
    reader: PdfReader, outline: list, results: list[tuple[int, int, str]], level: int
):
    """Recursively collect all bookmarks with their depth level."""
    for item in outline:
        if isinstance(item, list):
            # Nested bookmarks
            _collect_bookmarks_recursive(reader, item, results, level + 1)
        else:
            try:
                page_num = reader.get_destination_page_number(item)
                title = item.title if hasattr(item, "title") else str(item)

                # Skip file-like titles (common in scanned/compiled PDFs)
                if title.endswith(".pdf") or "Binder" in title:
                    continue

                if page_num is not None:
                    results.append((page_num, level, title))
            except (KeyError, AttributeError, TypeError):
                continue


def _find_optimal_level(by_level: dict[int, list[int]], total_pages: int) -> int:
    """
    Find the bookmark level that provides the best split points.

    Criteria:
    - Enough points to create meaningful chunks (3-20 for most docs)
    - Points are reasonably distributed
    - Prefers chapter-level splits over section-level
    """
    best_level = -1
    best_score: float = -1.0

    for level, pages in by_level.items():
        unique_pages = sorted(set(pages))
        num_points = len(unique_pages)

        # Skip levels with too few or too many points
        if num_points < 3:
            continue
        if num_points > 50:  # Too granular
            continue

        # Calculate distribution score
        if num_points > 1:
            gaps = [unique_pages[i + 1] - unique_pages[i] for i in range(len(unique_pages) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            max_gap = max(gaps)

            # Penalize very uneven distributions
            evenness = avg_gap / max_gap if max_gap > 0 else 0

            # Score: balance between number of chunks and evenness
            target_chunks = min(20, max(5, total_pages // 50))
            chunk_score = 1 - abs(num_points - target_chunks) / target_chunks
            score = (evenness * 0.6) + (chunk_score * 0.4)

            if score > best_score:
                best_score = score
                best_level = level

    return best_level


def _is_balanced(boundaries: list[tuple[int, int]], total_pages: int, max_ratio: float) -> bool:
    """Check if chunk distribution is reasonably balanced."""
    if not boundaries:
        return False

    for start, end in boundaries:
        chunk_size = end - start
        ratio = chunk_size / total_pages
        if ratio > max_ratio:
            logger.debug(
                f"Chunk pages {start}-{end} ({chunk_size} pages) exceeds max ratio {max_ratio:.0%}"
            )
            return False

    # For large documents, ensure minimum number of chunks
    return not (total_pages > 200 and len(boundaries) < MIN_CHUNKS_FOR_LARGE_DOC)


def _rebalance_chunks(
    boundaries: list[tuple[int, int]], total_pages: int, chunk_size: int, overlap: int
) -> list[tuple[int, int]]:
    """
    Rebalance chunks by splitting large ones with fixed-size strategy.
    """
    if not boundaries:
        return boundaries

    result = []
    for start, end in boundaries:
        chunk_pages = end - start

        # If chunk is too large, split it further
        if chunk_pages > chunk_size * 2:
            sub_boundaries = _get_fixed_boundaries(chunk_pages, chunk_size, overlap)
            # Offset to actual page positions
            for sub_start, sub_end in sub_boundaries:
                result.append((start + sub_start, start + sub_end))
        else:
            result.append((start, end))

    return result


def _get_fixed_boundaries(total_pages: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Generate fixed-size chunk boundaries with overlap."""
    # Validate inputs
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    boundaries = []
    start = 0

    # Cap overlap to at most 50% of chunk_size to ensure reasonable progress
    # (each chunk should advance by at least half the chunk size)
    max_overlap = chunk_size // 2
    effective_overlap = min(overlap, max_overlap)

    while start < total_pages:
        end = min(start + chunk_size, total_pages)
        boundaries.append((start, end))
        if end >= total_pages:
            break
        start = end - effective_overlap

    return boundaries


def analyze_document_structure(pdf_path: Path) -> dict[str, Any]:
    """
    Analyze a PDF's structure for splitting decisions.

    Returns detailed information about bookmarks, potential split points,
    and recommended strategy.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    analysis: dict[str, Any] = {
        "filename": pdf_path.name,
        "total_pages": total_pages,
        "has_outline": bool(reader.outline),
        "bookmark_levels": {},
        "recommended_strategy": None,
        "recommended_level": None,
    }

    if not reader.outline:
        analysis["recommended_strategy"] = "fixed"
        return analysis

    # Collect all bookmarks
    all_bookmarks: list[tuple[int, int, str]] = []
    _collect_bookmarks_recursive(reader, reader.outline, all_bookmarks, level=0)

    # Analyze by level
    by_level: dict[int, list[tuple[int, str]]] = {}
    for page, level, title in all_bookmarks:
        if level not in by_level:
            by_level[level] = []
        by_level[level].append((page, title))

    for level, items in sorted(by_level.items()):
        pages = [p for p, t in items]
        titles = [t for p, t in items][:5]  # Sample titles

        analysis["bookmark_levels"][level] = {
            "count": len(items),
            "unique_pages": len(set(pages)),
            "sample_titles": titles,
        }

    # Determine best level and strategy
    best_level = _find_optimal_level(
        {k: [p for p, t in v] for k, v in by_level.items()}, total_pages
    )

    if best_level >= 0:
        analysis["recommended_level"] = best_level
        analysis["recommended_strategy"] = f"bookmark_level_{best_level}"
    else:
        analysis["recommended_strategy"] = "fixed"

    return analysis


def get_split_boundaries_hybrid(
    pdf_path: Path,
    max_chunk_pages: int = 100,
    min_chunk_pages: int = 10,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[list[tuple[int, int]], str]:
    """
    Hybrid approach: Use chapter boundaries and subdivide large chapters.

    Strategy:
    1. Find chapter-level boundaries (level 4 typically)
    2. For chapters exceeding max_chunk_pages, subdivide using:
       a. Section boundaries within that chapter (level 5)
       b. Fixed splitting if no good section boundaries
    3. Merge tiny chapters with neighbors

    Args:
        pdf_path: Path to the PDF
        max_chunk_pages: Maximum pages per chunk before subdividing
        min_chunk_pages: Minimum pages (smaller chunks get merged)
        overlap: Overlap between chunks

    Returns:
        Tuple of (boundaries, strategy_description)
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    logger.debug(f"Hybrid strategy: analyzing {total_pages} pages")

    if total_pages == 0:
        return [], "empty"

    if total_pages <= max_chunk_pages:
        logger.debug(f"Document fits in single chunk ({total_pages} <= {max_chunk_pages})")
        return [(0, total_pages)], "single_chunk"

    # Collect all bookmarks
    all_bookmarks: list[tuple[int, int, str]] = []
    if reader.outline:
        _collect_bookmarks_recursive(reader, reader.outline, all_bookmarks, level=0)
    logger.debug(f"Collected {len(all_bookmarks)} bookmarks for hybrid analysis")

    # Group by level
    by_level: dict[int, list[tuple[int, str]]] = {}
    for page, level, title in all_bookmarks:
        if level not in by_level:
            by_level[level] = []
        by_level[level].append((page, title))

    # Find chapter level (usually level 4, look for "CHAPTER" keyword)
    chapter_level = None
    for level, items in sorted(by_level.items()):
        titles = [t for p, t in items]
        if any("CHAPTER" in t.upper() or "PART" in t.upper() for t in titles):
            chapter_level = level
            logger.debug(f"Found chapter-level bookmarks at level {level} ({len(items)} items)")
            break

    if chapter_level is None:
        # Fall back to fixed splitting
        logger.debug("No chapter-level bookmarks found, falling back to fixed")
        return _get_fixed_boundaries(total_pages, max_chunk_pages, overlap), "fixed"

    # Get chapter boundaries
    chapter_pages = sorted({p for p, _t in by_level[chapter_level]})
    if chapter_pages[0] != 0:
        chapter_pages.insert(0, 0)
    logger.debug(f"Chapter boundaries at pages: {chapter_pages}")

    # Get section boundaries (one level deeper)
    section_level = chapter_level + 1
    section_pages: set[int] = set()
    if section_level in by_level:
        section_pages = {p for p, _t in by_level[section_level]}
        logger.debug(f"Found {len(section_pages)} section boundaries at level {section_level}")

    # Build final boundaries
    final_boundaries = []
    strategy_parts = []

    for i in range(len(chapter_pages)):
        chap_start = chapter_pages[i]
        chap_end = chapter_pages[i + 1] if i + 1 < len(chapter_pages) else total_pages
        chap_size = chap_end - chap_start

        if chap_size <= max_chunk_pages:
            # Chapter fits in one chunk
            logger.debug(
                f"Chapter {i + 1}: pages {chap_start + 1}-{chap_end} ({chap_size} pages) - single chunk"
            )
            final_boundaries.append((chap_start, chap_end))
        else:
            # Subdivide large chapter
            logger.debug(
                f"Chapter {i + 1}: pages {chap_start + 1}-{chap_end} ({chap_size} pages) - needs subdivision"
            )
            # First try using section boundaries
            relevant_sections = sorted([p for p in section_pages if chap_start < p < chap_end])

            if relevant_sections:
                # Use section boundaries
                logger.debug(f"  Using {len(relevant_sections)} section boundaries")
                section_points = [chap_start, *relevant_sections, chap_end]
                sub_boundaries = []

                for j in range(len(section_points) - 1):
                    s_start = section_points[j]
                    s_end = section_points[j + 1]
                    sub_boundaries.append((s_start, s_end))

                # Merge small sections, split large ones
                merged = _merge_and_split_boundaries(
                    sub_boundaries, max_chunk_pages, min_chunk_pages, overlap
                )
                final_boundaries.extend(merged)
                strategy_parts.append(f"ch{i + 1}:sections")
            else:
                # Fixed split for this chapter
                logger.debug("  No sections found, using fixed split")
                sub = _get_fixed_boundaries(chap_size, max_chunk_pages, overlap)
                for s, e in sub:
                    final_boundaries.append((chap_start + s, chap_start + e))
                strategy_parts.append(f"ch{i + 1}:fixed")

    # Final merge pass for tiny chunks
    pre_merge_count = len(final_boundaries)
    final_boundaries = _merge_tiny_chunks(final_boundaries, min_chunk_pages)
    if len(final_boundaries) != pre_merge_count:
        logger.debug(f"Merged tiny chunks: {pre_merge_count} -> {len(final_boundaries)}")

    strategy = f"hybrid_chapter_l{chapter_level}"
    if strategy_parts:
        strategy += f"({','.join(strategy_parts[:3])}{'...' if len(strategy_parts) > 3 else ''})"

    logger.info(f"Hybrid strategy complete: {len(final_boundaries)} chunks")
    return final_boundaries, strategy


def _merge_and_split_boundaries(
    boundaries: list[tuple[int, int]], max_pages: int, min_pages: int, overlap: int
) -> list[tuple[int, int]]:
    """Merge small boundaries and split large ones."""
    result: list[tuple[int, int]] = []
    accumulated_start: int | None = None
    accumulated_end: int | None = None

    for start, end in boundaries:
        size = end - start

        if size > max_pages:
            # Flush accumulated
            if accumulated_start is not None and accumulated_end is not None:
                result.append((accumulated_start, accumulated_end))
                accumulated_start = None
                accumulated_end = None

            # Split large chunk
            sub = _get_fixed_boundaries(size, max_pages, overlap)
            for s, e in sub:
                result.append((start + s, start + e))

        elif size < min_pages:
            # Accumulate small chunks
            if accumulated_start is None:
                accumulated_start = start
            accumulated_end = end

            # Flush if accumulated enough
            if (
                accumulated_end is not None
                and accumulated_start is not None
                and accumulated_end - accumulated_start >= min_pages
            ):
                result.append((accumulated_start, accumulated_end))
                accumulated_start = None
                accumulated_end = None
        else:
            # Flush accumulated
            if accumulated_start is not None and accumulated_end is not None:
                result.append((accumulated_start, accumulated_end))
                accumulated_start = None
                accumulated_end = None

            result.append((start, end))

    # Final flush
    if accumulated_start is not None and accumulated_end is not None:
        result.append((accumulated_start, accumulated_end))

    return result


def _merge_tiny_chunks(boundaries: list[tuple[int, int]], min_pages: int) -> list[tuple[int, int]]:
    """Merge chunks smaller than min_pages with neighbors."""
    if len(boundaries) <= 1:
        return boundaries

    result = []
    i = 0

    while i < len(boundaries):
        start, end = boundaries[i]
        size = end - start

        # Try to merge with next if too small
        while size < min_pages and i + 1 < len(boundaries):
            i += 1
            _, next_end = boundaries[i]
            end = next_end
            size = end - start

        result.append((start, end))
        i += 1

    return result


def get_split_boundaries_with_docling_toc(
    pdf_path: Path,
    toc_page_range: tuple[int, int] = (0, 50),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[list[tuple[int, int]], str]:
    """
    Use Docling to extract TOC from first pages for split boundaries.

    This is useful when PDF bookmarks are missing or unreliable.

    Args:
        pdf_path: Path to the PDF
        toc_page_range: Page range likely containing TOC (0-indexed)
        chunk_size: Fallback chunk size
        overlap: Fallback overlap

    Returns:
        Tuple of (boundaries, strategy_used)
    """
    import re
    import tempfile

    from pypdf import PdfReader, PdfWriter

    from pdf_splitter.config_factory import create_converter

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    # Extract TOC pages to temporary file
    toc_start, toc_end = toc_page_range
    toc_end = min(toc_end, total_pages)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        writer = PdfWriter()
        for i in range(toc_start, toc_end):
            writer.add_page(reader.pages[i])
        writer.write(tmp)
        toc_pdf_path = Path(tmp.name)

    try:
        # Process TOC pages with Docling
        converter = create_converter()
        result = converter.convert(str(toc_pdf_path))
        doc = result.document

        # Extract text and look for page references
        toc_entries = []
        text_content = doc.export_to_markdown()

        # Pattern: "Chapter/Section Name ... page_number" or similar
        patterns = [
            r"(?:CHAPTER|Chapter|PART|Part|SECTION|Section)\s+(\d+)[^\d]*?(\d{1,4})\s*$",
            r"^([A-Z][A-Z\s]+)\s+\.+\s*(\d{1,4})\s*$",  # "TITLE .... 123"
            r"^(\d+(?:\.\d+)*)\s+([^\.]+)\s+\.+\s*(\d{1,4})\s*$",  # "1.2 Title ... 45"
        ]

        for line in text_content.split("\n"):
            line = line.strip()
            for pattern in patterns:
                match = re.search(pattern, line, re.MULTILINE)
                if match:
                    groups = match.groups()
                    page_num = int(groups[-1])
                    if 0 < page_num <= total_pages:
                        toc_entries.append(page_num - 1)  # Convert to 0-indexed
                    break

        if toc_entries:
            # Deduplicate and sort
            page_numbers = sorted(set(toc_entries))

            if page_numbers[0] != 0:
                page_numbers.insert(0, 0)

            boundaries = []
            for i in range(len(page_numbers)):
                start = page_numbers[i]
                end = page_numbers[i + 1] if i + 1 < len(page_numbers) else total_pages
                if start < end:
                    boundaries.append((start, end))

            if _is_balanced(boundaries, total_pages, MAX_CHUNK_RATIO):
                return boundaries, "docling_toc"

    except Exception as e:
        logger.warning(f"Docling TOC extraction failed: {e}")

    finally:
        toc_pdf_path.unlink(missing_ok=True)

    # Fallback to fixed
    return _get_fixed_boundaries(total_pages, chunk_size, overlap), "fixed"
