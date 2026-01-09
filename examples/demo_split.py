#!/usr/bin/env python3
"""
Demo script to analyze and split PDFs from the assets folder.
All processing is done locally - no data leaves this machine.
"""

import tempfile
from pathlib import Path
from typing import cast

from pypdf import PdfReader

from pdf_splitter.segmentation import get_page_coverage, get_split_boundaries, split_pdf


def analyze_pdf(pdf_path: Path) -> dict:
    """Analyze a PDF and return metadata."""
    reader = PdfReader(str(pdf_path))

    info = {
        "filename": pdf_path.name,
        "size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "total_pages": len(reader.pages),
        "has_outline": bool(reader.outline),
        "outline_items": 0,
    }

    # Count top-level outline items
    if reader.outline:
        for item in reader.outline:
            if not isinstance(item, list):
                info["outline_items"] = cast(int, info["outline_items"]) + 1

    return info


def demo_split(pdf_path: Path, chunk_size: int = 50, overlap: int = 5):
    """Demonstrate splitting a PDF and show results."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {pdf_path.name}")
    print("=" * 60)

    # Get PDF info
    info = analyze_pdf(pdf_path)
    print(f"  Size: {info['size_mb']:.2f} MB")
    print(f"  Pages: {info['total_pages']}")
    print(f"  Has bookmarks: {info['has_outline']}")
    if info["has_outline"]:
        print(f"  Top-level bookmarks: {info['outline_items']}")

    # Get split boundaries
    boundaries = get_split_boundaries(pdf_path, chunk_size=chunk_size, overlap=overlap)

    print("\nSplit Strategy:")
    if info["has_outline"] and info["outline_items"] > 1:
        print("  Using: Smart splitting (bookmark-based)")
    else:
        print(f"  Using: Fixed splitting (chunk_size={chunk_size}, overlap={overlap})")

    print(f"  Chunks: {len(boundaries)}")

    # Show boundary details
    print("\nChunk Boundaries:")
    for i, (start, end) in enumerate(boundaries):
        pages = end - start
        print(f"  Chunk {i + 1:3d}: pages {start + 1:4d} - {end:4d} ({pages:3d} pages)")

    # Verify coverage
    coverage_ok = get_page_coverage(boundaries, info["total_pages"])
    print(f"\nPage Coverage: {'PASS' if coverage_ok else 'FAIL'}")

    # Actually split to temp directory
    print("\nSplitting to temporary directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        chunk_paths = split_pdf(pdf_path, output_dir, chunk_size=chunk_size, overlap=overlap)

        print(f"  Created {len(chunk_paths)} chunk files:")
        total_chunk_size = 0
        for chunk_path in chunk_paths:
            size_kb = chunk_path.stat().st_size / 1024
            total_chunk_size += size_kb
            chunk_reader = PdfReader(str(chunk_path))
            print(f"    {chunk_path.name}: {len(chunk_reader.pages)} pages, {size_kb:.1f} KB")

        print(f"\n  Total chunk size: {total_chunk_size / 1024:.2f} MB")
        print(f"  Original size: {info['size_mb']:.2f} MB")

    return info, boundaries


def main():
    assets_dir = Path(__file__).parent / "assets"

    print("PDF Splitter Demo")
    print("All processing is LOCAL - no data transmitted")
    print(f"\nAssets directory: {assets_dir}")

    pdfs = list(assets_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF(s)")

    results = []
    for pdf_path in sorted(pdfs):
        info, boundaries = demo_split(pdf_path)
        results.append((pdf_path.name, info, boundaries))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'PDF':<30} {'Pages':>8} {'Chunks':>8} {'Strategy':<15}")
    print("-" * 60)
    for name, info, boundaries in results:
        strategy = "Bookmark" if info["has_outline"] and info["outline_items"] > 1 else "Fixed"
        print(f"{name:<30} {info['total_pages']:>8} {len(boundaries):>8} {strategy:<15}")


if __name__ == "__main__":
    main()
