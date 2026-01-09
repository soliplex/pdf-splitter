"""
Validation utilities for Docling output against source chunks.
"""

import json
import re
from pathlib import Path
from typing import Any


def parse_chunk_filename(filename: str) -> tuple[int | None, int | None, int | None]:
    """Extract chunk index and page range from filename."""
    match = re.match(r"chunk_(\d+)_pages_(\d+)_(\d+)\.pdf", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def extract_provenance_pages(document_dict: dict) -> list[int]:
    """Extract all page numbers from provenance data."""
    pages = set()
    for key in ("texts", "tables", "pictures"):
        for item in document_dict.get(key, []):
            for prov in item.get("prov", []):
                if "page_no" in prov:
                    pages.add(prov["page_no"])
    return sorted(pages)


def validate_chunk(chunk_result: dict, chunks_dir: Path) -> dict[str, Any]:
    """Validate a single chunk result."""
    issues = []
    chunk_path = chunk_result.get("chunk_path", "")
    chunk_name = Path(chunk_path).name

    chunk_idx, start_page, end_page = parse_chunk_filename(chunk_name)

    if chunk_idx is None or start_page is None or end_page is None:
        issues.append(f"Could not parse chunk filename: {chunk_name}")
        return {"chunk": chunk_name, "valid": False, "issues": issues}

    # Assert for mypy type narrowing (values are guaranteed non-None after above check)
    assert start_page is not None and end_page is not None
    chunk_page_count = end_page - start_page + 1

    chunk_file = chunks_dir / chunk_name
    if not chunk_file.exists():
        issues.append(f"Chunk file not found: {chunk_file}")

    if not chunk_result.get("success"):
        error = chunk_result.get("error", "Unknown error")
        issues.append(f"Processing failed: {error}")
        return {
            "chunk": chunk_name,
            "chunk_idx": chunk_idx,
            "original_pages": (start_page, end_page),
            "chunk_page_count": chunk_page_count,
            "valid": False,
            "issues": issues,
        }

    doc_dict = chunk_result.get("document_dict", {})
    if not doc_dict:
        issues.append("No document_dict in result")
        return {
            "chunk": chunk_name,
            "chunk_idx": chunk_idx,
            "original_pages": (start_page, end_page),
            "chunk_page_count": chunk_page_count,
            "valid": False,
            "issues": issues,
        }

    prov_pages = extract_provenance_pages(doc_dict)
    expected_chunk_pages = set(range(1, chunk_page_count + 1))
    actual_pages = set(prov_pages)

    missing_pages = expected_chunk_pages - actual_pages
    extra_pages = actual_pages - expected_chunk_pages

    if missing_pages and len(missing_pages) > chunk_page_count * 0.1:
        issues.append(f"Missing {len(missing_pages)}/{chunk_page_count} pages in provenance")

    if extra_pages:
        issues.append(f"Pages outside chunk range: {sorted(extra_pages)}")

    coverage = (
        len(actual_pages & expected_chunk_pages) / chunk_page_count * 100
        if chunk_page_count > 0
        else 0
    )

    num_texts = len(doc_dict.get("texts", []))
    num_tables = len(doc_dict.get("tables", []))
    num_pictures = len(doc_dict.get("pictures", []))

    if num_texts == 0 and num_tables == 0:
        issues.append("No text or table content extracted")

    is_valid = not any(
        "outside chunk range" in i or "Processing failed" in i or "No text or table" in i
        for i in issues
    )

    return {
        "chunk": chunk_name,
        "chunk_idx": chunk_idx,
        "original_pages": (start_page, end_page),
        "chunk_page_count": chunk_page_count,
        "provenance_pages": prov_pages,
        "coverage_pct": coverage,
        "num_texts": num_texts,
        "num_tables": num_tables,
        "num_pictures": num_pictures,
        "valid": is_valid,
        "issues": issues,
    }


def validate_global_coverage(results: list[dict], chunk_validations: list[dict]) -> list[str]:
    """Validate coverage across all chunks."""
    issues = []

    total_chunks = len(chunk_validations)
    chunks_with_content = sum(
        1 for v in chunk_validations if v.get("num_texts", 0) > 0 or v.get("num_tables", 0) > 0
    )

    if chunks_with_content < total_chunks:
        issues.append(f"{total_chunks - chunks_with_content} chunks have no extracted content")

    coverages = [v.get("coverage_pct", 0) for v in chunk_validations if "coverage_pct" in v]
    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        if avg_coverage < 80:
            issues.append(f"Average page coverage is low: {avg_coverage:.1f}%")

    chunk_indices = sorted(
        [v.get("chunk_idx", -1) for v in chunk_validations if v.get("chunk_idx") is not None]
    )
    if chunk_indices:
        expected_indices = list(range(chunk_indices[0], chunk_indices[-1] + 1))
        missing_chunks = set(expected_indices) - set(chunk_indices)
        if missing_chunks:
            issues.append(f"Missing chunk indices: {sorted(missing_chunks)}")

    all_ranges = []
    for v in chunk_validations:
        if "original_pages" in v and v["original_pages"][0] is not None:
            all_ranges.append(v["original_pages"])

    all_ranges.sort(key=lambda x: x[0])

    for i in range(1, len(all_ranges)):
        prev_end = all_ranges[i - 1][1]
        curr_start = all_ranges[i][0]
        if curr_start > prev_end + 1:
            issues.append(f"Gap between chunks: pages {prev_end + 1}-{curr_start - 1} not covered")

    return issues


def run_validation(
    docling_json: Path, chunks_dir: Path, verbose: bool = False
) -> tuple[bool, dict[str, Any]]:
    """
    Run full validation and return results.

    Returns:
        Tuple of (success, stats_dict)
    """
    with open(docling_json) as f:
        results = json.load(f)

    validations = []
    for result in results:
        v = validate_chunk(result, chunks_dir)
        validations.append(v)

    validations.sort(key=lambda x: x.get("chunk_idx", 999))

    global_issues = validate_global_coverage(results, validations)

    valid_count = sum(1 for v in validations if v["valid"])
    total_texts = sum(v.get("num_texts", 0) for v in validations)
    total_tables = sum(v.get("num_tables", 0) for v in validations)
    total_pictures = sum(v.get("num_pictures", 0) for v in validations)

    all_valid = valid_count == len(validations) and len(global_issues) == 0

    stats = {
        "total_chunks": len(validations),
        "valid_chunks": valid_count,
        "total_texts": total_texts,
        "total_tables": total_tables,
        "total_pictures": total_pictures,
        "validations": validations,
        "global_issues": global_issues,
    }

    return all_valid, stats
