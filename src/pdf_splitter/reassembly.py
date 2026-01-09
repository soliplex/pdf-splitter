"""
Reassembly Module

Stitches processed chunks back into a cohesive document using
custom concatenation logic (docling-core's concatenate() is buggy).

This preserves:
- DOM tree structure
- Provenance data
- Page number offsetting
- All reference integrity
"""

import copy
import logging
import re
from typing import Any

from docling_core.types.doc import DoclingDocument

logger = logging.getLogger(__name__)


# Collections that contain items with references
ITEM_COLLECTIONS = ["texts", "tables", "pictures", "groups", "key_value_items", "form_items"]

# Reference fields that may contain $ref pointers
REF_FIELDS = ["children", "captions", "references", "footnotes"]


def _remap_ref(ref: str, offsets: dict[str, int]) -> str:
    """
    Remap a reference string by applying collection offsets.

    Args:
        ref: Reference string like "#/texts/5" or "#/body"
        offsets: Dict mapping collection names to their offsets

    Returns:
        Remapped reference string

    Examples:
        >>> _remap_ref("#/texts/5", {"texts": 100})
        "#/texts/105"
        >>> _remap_ref("#/body", {"texts": 100})
        "#/body"
    """
    # Match pattern like #/texts/123 or #/tables/0
    match = re.match(r"^#/(\w+)/(\d+)$", ref)
    if match:
        collection = match.group(1)
        index = int(match.group(2))
        if collection in offsets:
            return f"#/{collection}/{index + offsets[collection]}"
    # Non-indexed refs like #/body, #/furniture stay unchanged
    return ref


def _remap_ref_dict(ref_dict: dict[str, str], offsets: dict[str, int]) -> dict[str, str]:
    """
    Remap a reference dict like {"$ref": "#/texts/5"}.

    Args:
        ref_dict: Dict with "$ref" key
        offsets: Collection offsets

    Returns:
        New dict with remapped reference
    """
    if "$ref" in ref_dict:
        return {"$ref": _remap_ref(ref_dict["$ref"], offsets)}
    return ref_dict


def _remap_item_refs(item: dict[str, Any], offsets: dict[str, int]) -> dict[str, Any]:
    """
    Remap all references in an item (text, table, picture, group, etc.).

    Updates:
    - self_ref
    - parent.$ref
    - children[*].$ref
    - captions[*].$ref
    - references[*].$ref
    - footnotes[*].$ref
    - data.table_cells[*].ref.$ref (for RichTableCell in tables)

    Args:
        item: Item dict to update (modified in place)
        offsets: Collection offsets

    Returns:
        The modified item
    """
    # Remap self_ref
    if "self_ref" in item:
        item["self_ref"] = _remap_ref(item["self_ref"], offsets)

    # Remap parent
    if "parent" in item and isinstance(item["parent"], dict):
        item["parent"] = _remap_ref_dict(item["parent"], offsets)

    # Remap reference list fields
    for field in REF_FIELDS:
        if field in item and isinstance(item[field], list):
            item[field] = [
                _remap_ref_dict(ref, offsets) if isinstance(ref, dict) else ref
                for ref in item[field]
            ]

    # Remap RichTableCell refs in table data
    if "data" in item and isinstance(item["data"], dict):
        data = item["data"]
        if "table_cells" in data and isinstance(data["table_cells"], list):
            for cell in data["table_cells"]:
                if isinstance(cell, dict) and "ref" in cell and isinstance(cell["ref"], dict):
                    cell["ref"] = _remap_ref_dict(cell["ref"], offsets)

    return item


def _offset_provenance(item: dict[str, Any], page_offset: int) -> dict[str, Any]:
    """
    Offset page numbers in an item's provenance data.

    Args:
        item: Item dict with 'prov' field
        page_offset: Number to add to each page_no

    Returns:
        The modified item
    """
    if "prov" in item and isinstance(item["prov"], list):
        for prov in item["prov"]:
            if isinstance(prov, dict) and "page_no" in prov:
                prov["page_no"] = prov["page_no"] + page_offset
    return item


def concatenate_documents(docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Concatenate multiple DoclingDocument dicts into a single document.

    This is a custom implementation that works around bugs in
    docling-core's DoclingDocument.concatenate() method.

    Args:
        docs: List of DoclingDocument dicts (from export_to_dict())

    Returns:
        Merged document dict, or None if input is empty
    """
    if not docs:
        logger.warning("No documents provided for concatenation")
        return None

    if len(docs) == 1:
        return copy.deepcopy(docs[0])

    # Initialize master from deep copy of first document
    master = copy.deepcopy(docs[0])
    master["name"] = "merged_document"

    logger.info(f"Starting concatenation with document 1/{len(docs)}")

    for doc_idx, doc in enumerate(docs[1:], start=2):
        logger.debug(f"Concatenating document {doc_idx}/{len(docs)}")

        # Calculate current offsets based on master's collection sizes
        offsets = {
            "texts": len(master.get("texts", [])),
            "tables": len(master.get("tables", [])),
            "pictures": len(master.get("pictures", [])),
            "groups": len(master.get("groups", [])),
            "key_value_items": len(master.get("key_value_items", [])),
            "form_items": len(master.get("form_items", [])),
        }

        # Calculate page offset (max page number in master)
        page_offset = 0
        if master.get("pages"):
            page_offset = max(int(k) for k in master["pages"])

        logger.debug(
            f"  Offsets: page={page_offset}, texts={offsets['texts']}, "
            f"tables={offsets['tables']}, pictures={offsets['pictures']}"
        )

        # Process and append items from each collection
        for collection in ITEM_COLLECTIONS:
            if collection not in doc or not doc[collection]:
                continue

            if collection not in master:
                master[collection] = []

            for item in doc[collection]:
                # Deep copy to avoid modifying original
                new_item = copy.deepcopy(item)

                # Remap all references
                _remap_item_refs(new_item, offsets)

                # Offset provenance page numbers
                _offset_provenance(new_item, page_offset)

                master[collection].append(new_item)

        # Merge body.children
        if "body" in doc and "children" in doc["body"]:
            if "body" not in master:
                master["body"] = {"self_ref": "#/body", "children": []}
            if "children" not in master["body"]:
                master["body"]["children"] = []

            for child_ref in doc["body"]["children"]:
                new_ref = _remap_ref_dict(copy.deepcopy(child_ref), offsets)
                master["body"]["children"].append(new_ref)

        # Merge furniture.children
        if "furniture" in doc and "children" in doc["furniture"]:
            if "furniture" not in master:
                master["furniture"] = {"self_ref": "#/furniture", "children": []}
            if "children" not in master["furniture"]:
                master["furniture"]["children"] = []

            for child_ref in doc["furniture"]["children"]:
                new_ref = _remap_ref_dict(copy.deepcopy(child_ref), offsets)
                master["furniture"]["children"].append(new_ref)

        # Merge pages with offset keys
        if doc.get("pages"):
            if "pages" not in master:
                master["pages"] = {}

            for page_key, page_data in doc["pages"].items():
                new_key = str(int(page_key) + page_offset)
                new_page = copy.deepcopy(page_data)
                # Update page_no in the page data itself
                if "page_no" in new_page:
                    new_page["page_no"] = int(page_key) + page_offset
                master["pages"][new_key] = new_page

        logger.info(f"Merged document {doc_idx}/{len(docs)}")

    # Log final statistics
    logger.info(
        f"Concatenation complete: {len(master.get('pages', {}))} pages, "
        f"{len(master.get('texts', []))} texts, "
        f"{len(master.get('tables', []))} tables, "
        f"{len(master.get('pictures', []))} pictures"
    )

    return master


def merge_documents(docs: list[DoclingDocument]) -> DoclingDocument | None:
    """
    Merge multiple DoclingDocuments into a single cohesive document.

    Uses custom concatenation logic that handles:
    - Page number offsetting
    - Reference remapping
    - DOM tree merging
    - Provenance data preservation

    Args:
        docs: List of DoclingDocument objects to merge

    Returns:
        Merged DoclingDocument, or None if input is empty
    """
    if not docs:
        logger.warning("No documents provided for merging")
        return None

    if len(docs) == 1:
        return docs[0]

    # Convert DoclingDocuments to dicts for custom concatenation
    doc_dicts = [doc.export_to_dict() for doc in docs]

    # Use custom concatenation
    merged_dict = concatenate_documents(doc_dicts)

    if merged_dict is None:
        return None

    # Reconstruct DoclingDocument from merged dict
    try:
        return DoclingDocument.model_validate(merged_dict)
    except Exception as e:
        logger.error(f"Failed to validate merged document: {e}")
        raise


def merge_from_results(results: list[dict[str, Any]]) -> DoclingDocument | None:
    """
    Merge processor results directly into a single DoclingDocument.

    Uses custom concatenation on the raw dicts for efficiency
    (avoids double serialization through DoclingDocument objects).

    Args:
        results: List of result dicts from BatchProcessor.execute_parallel()
                 Each dict should have 'document_dict' with serialized document

    Returns:
        Merged DoclingDocument, or None if no valid documents
    """
    doc_dicts = []

    for i, result in enumerate(results):
        if not result.get("success"):
            logger.warning(f"Skipping failed chunk {i}: {result.get('error')}")
            continue

        doc_dict = result.get("document_dict")
        if doc_dict is None:
            logger.warning(f"Chunk {i} has no document data")
            continue

        doc_dicts.append(doc_dict)

    if not doc_dicts:
        logger.error("No valid documents to merge")
        return None

    logger.info(f"Merging {len(doc_dicts)} valid documents from {len(results)} results")

    # Use custom concatenation directly on dicts
    merged_dict = concatenate_documents(doc_dicts)

    if merged_dict is None:
        return None

    # Validate and return as DoclingDocument
    try:
        return DoclingDocument.model_validate(merged_dict)
    except Exception as e:
        logger.error(f"Failed to validate merged document: {e}")
        raise


def validate_provenance_monotonicity(doc: DoclingDocument) -> bool:
    """
    Verify that page numbers in provenance data are monotonically increasing.

    This validates that concatenate() properly handled page number offsetting.

    Args:
        doc: The merged DoclingDocument to validate

    Returns:
        True if page numbers are monotonically increasing (or equal)
    """
    page_numbers = extract_provenance_pages(doc)

    if not page_numbers:
        return True

    for i in range(1, len(page_numbers)):
        if page_numbers[i] < page_numbers[i - 1]:
            logger.error(
                f"Provenance monotonicity violation: page {page_numbers[i]} "
                f"follows page {page_numbers[i - 1]} at index {i}"
            )
            return False

    return True


def extract_provenance_pages(doc: DoclingDocument) -> list[int]:
    """
    Extract all page numbers from document provenance data.

    Args:
        doc: DoclingDocument to inspect

    Returns:
        List of page numbers in document order
    """
    page_numbers = []

    try:
        # Iterate through all content items with provenance
        for item, _level in doc.iterate_items():
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no") and prov.page_no is not None:
                        page_numbers.append(prov.page_no)
    except Exception as e:
        logger.warning(f"Error extracting provenance pages: {e}")

    return page_numbers


def get_merge_statistics(doc: DoclingDocument) -> dict[str, Any]:
    """
    Get statistics about a merged document.

    Args:
        doc: DoclingDocument to analyze

    Returns:
        Dict with statistics about the document structure
    """
    stats: dict[str, Any] = {
        "total_items": 0,
        "tables": 0,
        "text_items": 0,
        "figures": 0,
        "unique_pages": set(),
        "page_range": (None, None),
    }
    unique_pages: set[int] = set()

    try:
        for item, _level in doc.iterate_items():
            stats["total_items"] += 1

            # Count item types
            item_type = type(item).__name__
            if "Table" in item_type:
                stats["tables"] += 1
            elif "Text" in item_type or "Paragraph" in item_type:
                stats["text_items"] += 1
            elif "Figure" in item_type or "Picture" in item_type:
                stats["figures"] += 1

            # Track pages
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no") and prov.page_no is not None:
                        unique_pages.add(prov.page_no)

        # Calculate page range
        if unique_pages:
            stats["page_range"] = (min(unique_pages), max(unique_pages))
            stats["unique_pages"] = len(unique_pages)
        else:
            stats["unique_pages"] = 0

    except Exception as e:
        logger.warning(f"Error calculating statistics: {e}")

    return stats
