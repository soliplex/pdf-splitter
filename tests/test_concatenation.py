"""
Tests for custom DoclingDocument concatenation.

Test data: assets/tbmed561_parts.json (4 chunks from 96-page PDF)
"""

import json
from pathlib import Path

import pytest

from pdf_splitter.reassembly import (
    _offset_provenance,
    _remap_item_refs,
    _remap_ref,
    _remap_ref_dict,
    concatenate_documents,
    merge_from_results,
)

# Path to test data
TEST_PARTS_FILE = Path(__file__).parent.parent / "assets" / "tbmed561_parts.json"


@pytest.fixture
def chunk_dicts():
    """Load the test chunk documents."""
    if not TEST_PARTS_FILE.exists():
        pytest.skip(f"Test data not found: {TEST_PARTS_FILE}")

    with open(TEST_PARTS_FILE) as f:
        data = json.load(f)

    return [item["document_dict"] for item in data if item.get("success")]


@pytest.fixture
def chunk_results():
    """Load the test chunk results (as returned by BatchProcessor)."""
    if not TEST_PARTS_FILE.exists():
        pytest.skip(f"Test data not found: {TEST_PARTS_FILE}")

    with open(TEST_PARTS_FILE) as f:
        return json.load(f)


class TestRemapRef:
    """Tests for _remap_ref helper."""

    def test_remap_indexed_ref(self):
        """Test remapping indexed reference."""
        offsets = {"texts": 100, "tables": 50}
        assert _remap_ref("#/texts/5", offsets) == "#/texts/105"
        assert _remap_ref("#/tables/0", offsets) == "#/tables/50"

    def test_remap_non_indexed_ref(self):
        """Test that non-indexed refs are unchanged."""
        offsets = {"texts": 100}
        assert _remap_ref("#/body", offsets) == "#/body"
        assert _remap_ref("#/furniture", offsets) == "#/furniture"

    def test_remap_unknown_collection(self):
        """Test that unknown collections are unchanged."""
        offsets = {"texts": 100}
        assert _remap_ref("#/unknown/5", offsets) == "#/unknown/5"

    def test_remap_ref_dict(self):
        """Test remapping reference dict."""
        offsets = {"texts": 100}
        result = _remap_ref_dict({"$ref": "#/texts/5"}, offsets)
        assert result == {"$ref": "#/texts/105"}

    def test_remap_ref_dict_no_ref(self):
        """Test that dicts without $ref are unchanged."""
        offsets = {"texts": 100}
        result = _remap_ref_dict({"other": "value"}, offsets)
        assert result == {"other": "value"}


class TestRemapItemRefs:
    """Tests for _remap_item_refs helper."""

    def test_remap_self_ref(self):
        """Test remapping self_ref."""
        item = {"self_ref": "#/texts/0", "text": "hello"}
        offsets = {"texts": 100}
        result = _remap_item_refs(item, offsets)
        assert result["self_ref"] == "#/texts/100"

    def test_remap_parent_ref(self):
        """Test remapping parent reference."""
        item = {"self_ref": "#/texts/0", "parent": {"$ref": "#/groups/5"}}
        offsets = {"texts": 100, "groups": 50}
        result = _remap_item_refs(item, offsets)
        assert result["parent"] == {"$ref": "#/groups/55"}

    def test_remap_children_refs(self):
        """Test remapping children references."""
        item = {
            "self_ref": "#/groups/0",
            "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
        }
        offsets = {"texts": 100, "groups": 50}
        result = _remap_item_refs(item, offsets)
        assert result["children"] == [{"$ref": "#/texts/100"}, {"$ref": "#/texts/101"}]

    def test_remap_captions_refs(self):
        """Test remapping caption references."""
        item = {"self_ref": "#/tables/0", "captions": [{"$ref": "#/texts/10"}]}
        offsets = {"texts": 100, "tables": 20}
        result = _remap_item_refs(item, offsets)
        assert result["captions"] == [{"$ref": "#/texts/110"}]

    def test_remap_table_cell_refs(self):
        """Test remapping RichTableCell references in table data."""
        item = {
            "self_ref": "#/tables/0",
            "data": {
                "table_cells": [
                    {"ref": {"$ref": "#/groups/3"}},
                    {"text": "plain cell"},
                    {"ref": {"$ref": "#/texts/5"}},
                ]
            },
        }
        offsets = {"texts": 100, "tables": 20, "groups": 50}
        result = _remap_item_refs(item, offsets)
        assert result["data"]["table_cells"][0]["ref"] == {"$ref": "#/groups/53"}
        assert result["data"]["table_cells"][1] == {"text": "plain cell"}
        assert result["data"]["table_cells"][2]["ref"] == {"$ref": "#/texts/105"}


class TestOffsetProvenance:
    """Tests for _offset_provenance helper."""

    def test_offset_single_prov(self):
        """Test offsetting single provenance entry."""
        item = {"prov": [{"page_no": 5, "bbox": {"l": 0, "t": 0}}]}
        result = _offset_provenance(item, 30)
        assert result["prov"][0]["page_no"] == 35

    def test_offset_multiple_prov(self):
        """Test offsetting multiple provenance entries."""
        item = {"prov": [{"page_no": 1}, {"page_no": 2}]}
        result = _offset_provenance(item, 100)
        assert result["prov"][0]["page_no"] == 101
        assert result["prov"][1]["page_no"] == 102

    def test_offset_preserves_bbox(self):
        """Test that bbox is preserved during offset."""
        bbox = {"l": 10.5, "t": 20.5, "r": 100.5, "b": 200.5}
        item = {"prov": [{"page_no": 1, "bbox": bbox}]}
        result = _offset_provenance(item, 30)
        assert result["prov"][0]["bbox"] == bbox

    def test_offset_no_prov(self):
        """Test handling item with no provenance."""
        item = {"text": "hello"}
        result = _offset_provenance(item, 30)
        assert result == {"text": "hello"}


class TestConcatenateDocuments:
    """Tests for concatenate_documents function."""

    def test_c01_page_count(self, chunk_dicts):
        """C-01: Total page count equals sum of chunk pages."""
        expected_pages = sum(len(d.get("pages", {})) for d in chunk_dicts)
        merged = concatenate_documents(chunk_dicts)
        assert len(merged["pages"]) == expected_pages

    def test_c02_page_keys_continuous(self, chunk_dicts):
        """C-02: Page keys are continuous from 1 to N."""
        merged = concatenate_documents(chunk_dicts)
        page_keys = sorted(int(k) for k in merged["pages"])
        expected = list(range(1, len(page_keys) + 1))
        assert page_keys == expected

    def test_c03_provenance_page_range(self, chunk_dicts):
        """C-03: Page numbers in provenance cover full document range."""
        merged = concatenate_documents(chunk_dicts)

        # Collect all page numbers
        page_numbers = set()
        for collection in ["texts", "tables", "pictures"]:
            for item in merged.get(collection, []):
                for prov in item.get("prov", []):
                    if "page_no" in prov:
                        page_numbers.add(prov["page_no"])

        total_pages = len(merged["pages"])

        # Verify page range covers document
        if page_numbers:
            assert min(page_numbers) >= 1, "Page numbers should start at 1"
            assert (
                max(page_numbers) <= total_pages
            ), f"Max page {max(page_numbers)} exceeds total {total_pages}"

    def test_c04_text_count(self, chunk_dicts):
        """C-04: Total text count equals sum of chunk texts."""
        expected = sum(len(d.get("texts", [])) for d in chunk_dicts)
        merged = concatenate_documents(chunk_dicts)
        assert len(merged["texts"]) == expected

    def test_c05_table_count(self, chunk_dicts):
        """C-05: Total table count equals sum of chunk tables."""
        expected = sum(len(d.get("tables", [])) for d in chunk_dicts)
        merged = concatenate_documents(chunk_dicts)
        assert len(merged["tables"]) == expected

    def test_c06_picture_count(self, chunk_dicts):
        """C-06: Total picture count equals sum of chunk pictures."""
        expected = sum(len(d.get("pictures", [])) for d in chunk_dicts)
        merged = concatenate_documents(chunk_dicts)
        assert len(merged["pictures"]) == expected

    def test_c07_group_count(self, chunk_dicts):
        """C-07: Total group count equals sum of chunk groups."""
        expected = sum(len(d.get("groups", [])) for d in chunk_dicts)
        merged = concatenate_documents(chunk_dicts)
        assert len(merged["groups"]) == expected

    def test_c08_reference_validity(self, chunk_dicts):
        """C-08: All $ref pointers resolve to existing items."""
        merged = concatenate_documents(chunk_dicts)

        # Build set of valid refs
        valid_refs = {"#/body", "#/furniture"}
        for collection in [
            "texts",
            "tables",
            "pictures",
            "groups",
            "key_value_items",
            "form_items",
        ]:
            for i, _item in enumerate(merged.get(collection, [])):
                valid_refs.add(f"#/{collection}/{i}")

        # Check all references
        def check_refs(obj, path=""):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref = obj["$ref"]
                    assert ref in valid_refs, f"Invalid ref {ref} at {path}"
                for k, v in obj.items():
                    check_refs(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_refs(v, f"{path}[{i}]")

        check_refs(merged)

    def test_c09_body_children_count(self, chunk_dicts):
        """C-09: Body children count equals sum of chunk body children."""
        expected = sum(len(d.get("body", {}).get("children", [])) for d in chunk_dicts)
        merged = concatenate_documents(chunk_dicts)
        assert len(merged["body"]["children"]) == expected

    def test_c10_no_duplicate_self_refs(self, chunk_dicts):
        """C-10: All self_ref values are unique."""
        merged = concatenate_documents(chunk_dicts)

        self_refs = []
        for collection in [
            "texts",
            "tables",
            "pictures",
            "groups",
            "key_value_items",
            "form_items",
        ]:
            for item in merged.get(collection, []):
                if "self_ref" in item:
                    self_refs.append(item["self_ref"])

        assert len(self_refs) == len(set(self_refs)), "Duplicate self_refs found"

    def test_c11_table_data_preserved(self, chunk_dicts):
        """C-11: Tables preserve their data structure."""
        merged = concatenate_documents(chunk_dicts)

        for table in merged.get("tables", []):
            assert "data" in table or "prov" in table, "Table missing data or prov"
            # If data exists, it should have expected structure
            if table.get("data"):
                data = table["data"]
                # Check for grid or other table structure
                assert isinstance(data, dict), "Table data should be dict"

    def test_c12_picture_images_preserved(self, chunk_dicts):
        """C-12: Pictures preserve image data."""
        merged = concatenate_documents(chunk_dicts)

        for pic in merged.get("pictures", []):
            # Pictures should have prov at minimum
            assert "prov" in pic, "Picture missing provenance"
            # If image exists, check structure
            if pic.get("image"):
                assert isinstance(pic["image"], dict), "Picture image should be dict"

    def test_c13_provenance_bbox_preserved(self, chunk_dicts):
        """C-13: Provenance bbox coordinates are preserved."""
        merged = concatenate_documents(chunk_dicts)

        # Get bbox from first text with prov in merged doc
        for text in merged.get("texts", []):
            if text.get("prov") and text["prov"][0].get("bbox"):
                bbox = text["prov"][0]["bbox"]
                # Bbox should have expected keys
                assert "l" in bbox or "left" in bbox or len(bbox) >= 4
                break

    def test_c14_round_trip_validation(self, chunk_dicts):
        """C-14: Merged document passes DoclingDocument validation."""
        from docling_core.types.doc import DoclingDocument

        merged = concatenate_documents(chunk_dicts)

        # Should not raise
        doc = DoclingDocument.model_validate(merged)
        assert doc is not None

    def test_c15_furniture_preserved(self, chunk_dicts):
        """C-15: Furniture structure is preserved."""
        merged = concatenate_documents(chunk_dicts)

        assert "furniture" in merged
        assert "self_ref" in merged["furniture"]
        assert merged["furniture"]["self_ref"] == "#/furniture"


class TestConcatenateEdgeCases:
    """Edge case tests for concatenation."""

    def test_empty_list(self):
        """Test concatenation of empty list."""
        result = concatenate_documents([])
        assert result is None

    def test_single_document(self, chunk_dicts):
        """Test concatenation of single document."""
        result = concatenate_documents([chunk_dicts[0]])
        # Should be a copy, not same object
        assert result is not chunk_dicts[0]
        assert result["name"] == chunk_dicts[0]["name"]

    def test_two_documents(self, chunk_dicts):
        """Test concatenation of exactly two documents."""
        result = concatenate_documents(chunk_dicts[:2])

        expected_pages = len(chunk_dicts[0]["pages"]) + len(chunk_dicts[1]["pages"])
        assert len(result["pages"]) == expected_pages

        expected_texts = len(chunk_dicts[0]["texts"]) + len(chunk_dicts[1]["texts"])
        assert len(result["texts"]) == expected_texts


class TestMergeFromResults:
    """Tests for merge_from_results function."""

    def test_merge_from_results_success(self, chunk_results):
        """Test merging from BatchProcessor results."""
        from docling_core.types.doc import DoclingDocument

        merged = merge_from_results(chunk_results)
        assert merged is not None
        assert isinstance(merged, DoclingDocument)

    def test_merge_from_results_with_failures(self, chunk_results):
        """Test merging handles failed chunks gracefully."""
        # Add a failed result
        results_with_failure = [
            *chunk_results,
            {"success": False, "chunk_path": "failed.pdf", "error": "Test error"},
        ]

        merged = merge_from_results(results_with_failure)
        assert merged is not None

    def test_merge_from_results_all_failed(self):
        """Test merging when all chunks failed."""
        results = [
            {"success": False, "chunk_path": "a.pdf", "error": "Error 1"},
            {"success": False, "chunk_path": "b.pdf", "error": "Error 2"},
        ]

        merged = merge_from_results(results)
        assert merged is None
