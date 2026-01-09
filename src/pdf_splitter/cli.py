#!/usr/bin/env python3
"""
PDF Splitter CLI

Command-line interface for smart PDF splitting.

Usage:
    python -m pdf_splitter.cli analyze <pdf_path>
    python -m pdf_splitter.cli split <pdf_path> [--output <dir>] [--max-pages <n>]
    python -m pdf_splitter.cli compare <pdf_path>
"""

import argparse
import os
import sys
from pathlib import Path

from pdf_splitter.logging_config import setup_logging


def cmd_analyze(args):
    """Analyze a PDF and show splitting recommendations."""
    from pdf_splitter.segmentation_enhanced import analyze_document_structure, smart_split

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        return 1

    error = _validate_options(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"{'=' * 70}")
    print(f"ANALYZING: {pdf_path.name}")
    print(f"{'=' * 70}")

    # Get file info
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    # Analyze structure
    analysis = analyze_document_structure(pdf_path)
    print(f"Total pages: {analysis['total_pages']}")
    print(f"Has bookmarks: {analysis['has_outline']}")

    if analysis["bookmark_levels"]:
        print("\nBookmark Structure:")
        for level, info in sorted(analysis["bookmark_levels"].items()):
            print(f"  Level {level}: {info['count']} items ({info['unique_pages']} unique pages)")
            if args.verbose and info["sample_titles"]:
                for title in info["sample_titles"][:3]:
                    print(f"    - {title[:55]}...")

    # Get smart split result
    result = smart_split(
        pdf_path,
        max_chunk_pages=args.max_pages,
        min_chunk_pages=args.min_pages,
        overlap=args.overlap,
    )

    print(f"\n{'=' * 70}")
    print("SMART SPLIT RESULT")
    print(f"{'=' * 70}")
    print(result.summary())

    if args.verbose:
        print("\nChunk Details:")
        for i, (start, end) in enumerate(result.boundaries):
            pages = end - start
            print(f"  {i + 1:3d}: pages {start + 1:5d} - {end:5d} ({pages:4d} pages)")

    return 0


def cmd_chunk(args):
    """Split a PDF into PDF chunk files."""
    from pdf_splitter.segmentation_enhanced import smart_split_to_files

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        return 1

    error = _validate_options(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else None

    print(f"Splitting: {pdf_path.name}")
    print(f"Max chunk size: {args.max_pages} pages")
    if output_dir:
        print(f"Output directory: {output_dir}")

    # Determine parallelism
    parallel = not args.sequential
    # Default to 80% of CPUs to leave headroom for other processes
    max_workers = args.workers or max(1, int((os.cpu_count() or 4) * 0.8))

    if parallel:
        print(f"Parallel writing: enabled ({max_workers} processes)")
    else:
        print("Parallel writing: disabled (sequential mode)")

    # Perform split
    chunk_paths, result = smart_split_to_files(
        pdf_path,
        output_dir=output_dir,
        max_chunk_pages=args.max_pages,
        min_chunk_pages=args.min_pages,
        overlap=args.overlap,
        force_strategy=args.strategy,
        max_workers=max_workers,
        parallel=parallel,
    )

    print(f"\n{result.summary()}")
    print(f"\nCreated {len(chunk_paths)} chunk files:")

    if output_dir is None and chunk_paths:
        output_dir = chunk_paths[0].parent

    total_size = 0
    for path in chunk_paths:
        size_kb = path.stat().st_size / 1024
        total_size += size_kb
        if args.verbose:
            print(f"  {path.name} ({size_kb:.1f} KB)")

    print(f"\nOutput directory: {output_dir}")
    print(f"Total size: {total_size / 1024:.2f} MB")

    return 0


def cmd_convert(args):
    """Convert PDF chunks to structured documents using Docling."""
    from pdf_splitter.processor import BatchProcessor
    from pdf_splitter.reassembly import merge_from_results

    # Gather chunk files
    input_path = Path(args.input)
    if input_path.is_file():
        chunk_paths = [input_path]
    elif input_path.is_dir():
        chunk_paths = sorted(input_path.glob("*.pdf"))
        if not chunk_paths:
            print(f"Error: No PDF files found in {input_path}", file=sys.stderr)
            return 1
    else:
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"{'=' * 70}")
        print("DOCLING CONVERSION")
        print(f"{'=' * 70}")
        print(f"Input: {input_path}")
        print(f"Chunks to process: {len(chunk_paths)}")
        print(f"Workers: {args.workers or 'auto (CPU count)'}")
        print(f"Tasks per worker: {args.maxtasks}")
        print()

    # Create processor
    processor = BatchProcessor(
        max_workers=args.workers, maxtasksperchild=args.maxtasks, verbose=args.verbose
    )

    # Process chunks
    results = processor.execute_parallel(chunk_paths)

    # Summary
    success_count = sum(1 for r in results if r and r.get("success"))
    fail_count = len(results) - success_count

    if args.verbose:
        print(f"\n{'=' * 70}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total chunks: {len(results)}")
        print(f"Succeeded: {success_count}")
        print(f"Failed: {fail_count}")
        print("\nResults:")
        for r in results:
            if r:
                status = "OK" if r["success"] else f"FAIL: {r['error']}"
                print(f"  {Path(r['chunk_path']).name}: {status}")
    else:
        # Minimal output by default
        if fail_count > 0:
            print(f"Converted {success_count}/{len(results)} chunks ({fail_count} failed)")
        else:
            print(f"Converted {len(results)} chunks")

    # Output results if requested
    if args.output:
        import json

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.keep_parts:
            # Keep individual chunk documents (legacy format)
            output_data = []
            for r in results:
                if r and r["success"]:
                    output_data.append(
                        {
                            "chunk_path": r["chunk_path"],
                            "success": True,
                            "document_dict": r["document_dict"],
                        }
                    )
                elif r:
                    output_data.append(
                        {"chunk_path": r["chunk_path"], "success": False, "error": r["error"]}
                    )

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults written to: {output_path} (individual chunks)")
        else:
            # Default: merge into single Docling-compatible document
            try:
                merged_doc = merge_from_results(results)
            except RuntimeError as e:
                if "docling-core bug" in str(e):
                    print(f"\nError: {e}", file=sys.stderr)
                    print(
                        "\nTip: Re-run with --keep-parts to output individual chunks instead.",
                        file=sys.stderr,
                    )
                    return 1
                raise

            if merged_doc is None:
                print("\nError: No valid documents to merge", file=sys.stderr)
                return 1

            # Export as Docling-compatible JSON
            output_data = merged_doc.export_to_dict()
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nMerged document written to: {output_path}")

    return 0 if fail_count == 0 else 1


def cmd_compare(args):
    """Compare all splitting strategies on a PDF."""
    from pypdf import PdfReader

    from pdf_splitter.segmentation import get_split_boundaries
    from pdf_splitter.segmentation_enhanced import (
        get_split_boundaries_enhanced,
        get_split_boundaries_hybrid,
        smart_split,
    )

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        return 1

    error = _validate_options(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    print(f"{'=' * 75}")
    print(f"STRATEGY COMPARISON: {pdf_path.name} ({total} pages)")
    print(f"{'=' * 75}")

    # Original
    b1 = get_split_boundaries(pdf_path, chunk_size=args.max_pages, overlap=args.overlap)
    sizes1 = [e - s for s, e in b1] if b1 else [0]

    # Enhanced
    b2, strat2 = get_split_boundaries_enhanced(pdf_path, args.max_pages, args.overlap)
    sizes2 = [e - s for s, e in b2] if b2 else [0]

    # Hybrid
    b3, strat3 = get_split_boundaries_hybrid(pdf_path, args.max_pages, args.min_pages, args.overlap)
    sizes3 = [e - s for s, e in b3] if b3 else [0]

    # Smart (auto)
    result = smart_split(pdf_path, args.max_pages, args.min_pages, args.overlap)

    print(f"\n{'Strategy':<35} {'Chunks':>7} {'Min':>6} {'Max':>6} {'Avg':>8}")
    print(f"{'-' * 35} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 8}")

    print(
        f"{'Original (basic)':<35} {len(b1):>7} {min(sizes1):>6} {max(sizes1):>6} {sum(sizes1) / len(sizes1):>8.1f}"
    )
    print(
        f"{'Enhanced (' + strat2[:20] + ')':<35} {len(b2):>7} {min(sizes2):>6} {max(sizes2):>6} {sum(sizes2) / len(sizes2):>8.1f}"
    )
    print(
        f"{'Hybrid (' + strat3[:23] + ')':<35} {len(b3):>7} {min(sizes3):>6} {max(sizes3):>6} {sum(sizes3) / len(sizes3):>8.1f}"
    )
    print(f"{'-' * 35} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 8}")
    print(
        f"{'>>> Smart Auto (' + result.strategy[:15] + ')':<35} {result.num_chunks:>7} {result.min_chunk_size:>6} {result.max_chunk_size:>6} {result.avg_chunk_size:>8.1f}"
    )

    # Recommendation
    print(f"\nRecommendation: smart_split() selected '{result.strategy}'")

    return 0


def cmd_batch(args):
    """Process all PDFs in a directory."""
    from pdf_splitter.segmentation_enhanced import smart_split

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}", file=sys.stderr)
        return 1

    error = _validate_options(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    pdfs = list(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return 0

    print(f"{'=' * 75}")
    print(f"BATCH ANALYSIS: {len(pdfs)} PDFs")
    print(f"{'=' * 75}")

    print(f"\n{'PDF':<30} {'Pages':>7} {'Chunks':>7} {'Max':>6} {'Strategy':<20}")
    print(f"{'-' * 30} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 20}")

    for pdf_path in sorted(pdfs):
        try:
            result = smart_split(pdf_path, args.max_pages, args.min_pages, args.overlap)
            print(
                f"{pdf_path.name[:29]:<30} {result.total_pages:>7} {result.num_chunks:>7} {result.max_chunk_size:>6} {result.strategy[:19]:<20}"
            )
        except Exception as e:
            print(f"{pdf_path.name[:29]:<30} ERROR: {str(e)[:40]}")

    return 0


def cmd_validate(args):
    """Validate Docling output against source chunks."""
    from pdf_splitter.validation import run_validation

    json_path = Path(args.json)
    chunks_dir = Path(args.chunks)

    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 1

    if not chunks_dir.is_dir():
        print(f"Error: Chunks directory not found: {chunks_dir}", file=sys.stderr)
        return 1

    success, stats = run_validation(json_path, chunks_dir, verbose=args.verbose)

    if args.verbose:
        print(f"{'=' * 70}")
        print("VALIDATION RESULTS")
        print(f"{'=' * 70}")
        print(f"JSON: {json_path}")
        print(f"Chunks: {chunks_dir}")
        print(f"{'-' * 70}")

        for v in stats["validations"]:
            status = "OK" if v["valid"] else "FAIL"
            pages_str = ""
            if "original_pages" in v and v["original_pages"][0] is not None:
                pages_str = f" (p{v['original_pages'][0]}-{v['original_pages'][1]})"
            coverage_str = f" {v['coverage_pct']:.0f}%" if "coverage_pct" in v else ""
            content_str = (
                f" [t:{v['num_texts']}, tbl:{v['num_tables']}, pic:{v['num_pictures']}]"
                if "num_texts" in v
                else ""
            )

            print(f"[{status:4}] {v['chunk']}{pages_str}{coverage_str}{content_str}")
            for issue in v.get("issues", []):
                print(f"       - {issue}")

        if stats["global_issues"]:
            print(f"{'-' * 70}")
            for issue in stats["global_issues"]:
                print(f"- {issue}")

        print(f"{'-' * 70}")

    # Summary
    print(
        f"Chunks: {stats['valid_chunks']}/{stats['total_chunks']} valid | "
        f"Elements: {stats['total_texts']} texts, {stats['total_tables']} tables, {stats['total_pictures']} pictures"
    )

    if success:
        print("PASSED")
        return 0
    else:
        print("FAILED")
        return 1


def _add_common_options(parser):
    """Add common splitting options to a parser."""
    parser.add_argument(
        "--max-pages", type=int, default=100, help="Maximum pages per chunk (default: 100)"
    )
    parser.add_argument(
        "--min-pages", type=int, default=15, help="Minimum pages per chunk (default: 15)"
    )
    parser.add_argument(
        "--overlap", type=int, default=0, help="Overlap pages between chunks (default: 0)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO level logging (default: WARNING)"
    )


def _validate_options(args):
    """Validate common options, returns error message or None."""
    if args.max_pages < 1:
        return f"--max-pages must be >= 1, got {args.max_pages}"
    if args.min_pages < 1:
        return f"--min-pages must be >= 1, got {args.min_pages}"
    if args.overlap < 0:
        return f"--overlap must be >= 0, got {args.overlap}"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="PDF Splitter - Smart PDF chunking for parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze a PDF:
    pdf-splitter analyze document.pdf
    pdf-splitter analyze document.pdf --verbose

  Create PDF chunks (parallel writing by default):
    pdf-splitter chunk document.pdf --output ./chunks
    pdf-splitter chunk document.pdf --max-pages 50 --workers 8
    pdf-splitter chunk document.pdf --sequential  # disable parallel

  Convert chunks to structured documents (parallel Docling):
    pdf-splitter convert ./chunks/ -o docling.json          # merged single document (default)
    pdf-splitter convert ./chunks/ -o parts.json --keep-parts  # individual chunks
    pdf-splitter convert ./chunks/ --workers 4 --output results.json

  Compare strategies:
    pdf-splitter compare document.pdf

  Batch analyze:
    pdf-splitter batch ./documents/

  Validate output:
    pdf-splitter validate results.json ./chunks/
""",
    )

    from . import __version__

    parser.add_argument("-V", "--version", action="version", version=f"pdf-splitter {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # analyze command
    p_analyze = subparsers.add_parser("analyze", help="Analyze PDF structure")
    p_analyze.add_argument("pdf", help="Path to PDF file")
    _add_common_options(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze)

    # chunk command
    p_chunk = subparsers.add_parser("chunk", help="Create PDF chunk files from a large PDF")
    p_chunk.add_argument("pdf", help="Path to PDF file")
    p_chunk.add_argument("-o", "--output", help="Output directory")
    p_chunk.add_argument(
        "-s", "--strategy", choices=["fixed", "hybrid", "enhanced"], help="Force specific strategy"
    )
    p_chunk.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for writing (default: CPU count)",
    )
    p_chunk.add_argument(
        "--sequential", action="store_true", help="Disable parallel writing (use sequential mode)"
    )
    _add_common_options(p_chunk)
    p_chunk.set_defaults(func=cmd_chunk)

    # convert command
    p_convert = subparsers.add_parser(
        "convert", help="Convert PDF chunks to single merged Docling document"
    )
    p_convert.add_argument("input", help="Path to chunk PDF or directory of chunks")
    p_convert.add_argument(
        "-o", "--output", help="Output JSON file (merged Docling document by default)"
    )
    p_convert.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    p_convert.add_argument(
        "--maxtasks",
        type=int,
        default=1,
        help="Tasks per worker before restart (default: 1 for memory isolation)",
    )
    p_convert.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO level logging (default: WARNING)"
    )
    p_convert.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep individual chunk documents instead of merging into single document",
    )
    p_convert.set_defaults(func=cmd_convert)

    # compare command
    p_compare = subparsers.add_parser("compare", help="Compare splitting strategies")
    p_compare.add_argument("pdf", help="Path to PDF file")
    _add_common_options(p_compare)
    p_compare.set_defaults(func=cmd_compare)

    # batch command
    p_batch = subparsers.add_parser("batch", help="Analyze all PDFs in directory")
    p_batch.add_argument("input_dir", help="Directory containing PDFs")
    _add_common_options(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    # validate command
    p_validate = subparsers.add_parser("validate", help="Validate Docling output against chunks")
    p_validate.add_argument("json", help="Path to Docling output JSON file")
    p_validate.add_argument("chunks", help="Path to chunks directory")
    p_validate.add_argument("-v", "--verbose", action="store_true", help="Show per-chunk details")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Configure logging based on verbosity
    setup_logging(verbose=args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
