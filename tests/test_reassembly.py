"""
Unit tests for reassembly module.

Tests document concatenation, page number monotonicity,
and proper handling of provenance data.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestMergeDocuments:
    """Tests for merge_documents function."""

    def test_merge_empty_list_returns_none(self):
        """Test that empty list returns None."""
        from pdf_splitter.reassembly import merge_documents

        result = merge_documents([])
        assert result is None

    def test_merge_single_document_returns_same(self):
        """Test that single document is returned as-is."""
        from pdf_splitter.reassembly import merge_documents

        mock_doc = MagicMock()
        result = merge_documents([mock_doc])
        assert result is mock_doc

    @patch("pdf_splitter.reassembly.DoclingDocument")
    def test_merge_multiple_documents_uses_custom_concat(self, mock_doc_class):
        """Test that merge uses custom concatenation for multiple documents."""
        from pdf_splitter.reassembly import merge_documents

        # Create minimal valid document dicts
        doc1 = MagicMock()
        doc2 = MagicMock()
        doc1.export_to_dict.return_value = {
            "name": "doc1",
            "pages": {"1": {"page_no": 1}},
            "texts": [],
            "body": {"self_ref": "#/body", "children": []},
            "furniture": {"self_ref": "#/furniture", "children": []},
        }
        doc2.export_to_dict.return_value = {
            "name": "doc2",
            "pages": {"1": {"page_no": 1}},
            "texts": [],
            "body": {"self_ref": "#/body", "children": []},
            "furniture": {"self_ref": "#/furniture", "children": []},
        }

        mock_merged = MagicMock()
        mock_doc_class.model_validate.return_value = mock_merged

        result = merge_documents([doc1, doc2])

        # Should call export_to_dict on each document
        doc1.export_to_dict.assert_called_once()
        doc2.export_to_dict.assert_called_once()
        # Should validate the merged dict
        mock_doc_class.model_validate.assert_called_once()
        assert result is mock_merged

    @patch("pdf_splitter.reassembly.DoclingDocument")
    def test_merge_handles_validation_error(self, mock_doc_class):
        """Test that merge raises on validation failure."""
        from pdf_splitter.reassembly import merge_documents

        doc1 = MagicMock()
        doc2 = MagicMock()
        doc1.export_to_dict.return_value = {
            "name": "doc1",
            "pages": {"1": {"page_no": 1}},
            "texts": [],
            "body": {"self_ref": "#/body", "children": []},
            "furniture": {"self_ref": "#/furniture", "children": []},
        }
        doc2.export_to_dict.return_value = {
            "name": "doc2",
            "pages": {"1": {"page_no": 1}},
            "texts": [],
            "body": {"self_ref": "#/body", "children": []},
            "furniture": {"self_ref": "#/furniture", "children": []},
        }
        mock_doc_class.model_validate.side_effect = Exception("Validation failed")

        with pytest.raises(Exception, match="Validation failed"):
            merge_documents([doc1, doc2])


class TestMergeFromResults:
    """Tests for merge_from_results function."""

    @patch("pdf_splitter.reassembly.DoclingDocument")
    def test_merge_from_results_skips_failed(self, mock_doc_class):
        """Test that failed results are skipped."""
        from pdf_splitter.reassembly import merge_from_results

        results = [
            {"success": False, "error": "Failed", "document_dict": None},
            {"success": True, "document_dict": {"test": "data"}},
        ]

        mock_doc = MagicMock()
        mock_doc_class.model_validate.return_value = mock_doc

        merge_from_results(results)

        # Should only process successful result
        mock_doc_class.model_validate.assert_called_once_with({"test": "data"})

    @patch("pdf_splitter.reassembly.DoclingDocument")
    def test_merge_from_results_all_failed_returns_none(self, mock_doc_class):
        """Test that all-failed results returns None."""
        from pdf_splitter.reassembly import merge_from_results

        results = [
            {"success": False, "error": "Failed 1", "document_dict": None},
            {"success": False, "error": "Failed 2", "document_dict": None},
        ]

        result = merge_from_results(results)
        assert result is None


class TestProvenanceMonotonicity:
    """Tests for provenance validation."""

    def test_monotonicity_empty_pages(self):
        """Test that empty page list is valid."""
        from pdf_splitter.reassembly import validate_provenance_monotonicity

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = []

        assert validate_provenance_monotonicity(mock_doc) is True

    def test_monotonicity_increasing_valid(self):
        """Test that increasing page numbers are valid."""
        from pdf_splitter.reassembly import validate_provenance_monotonicity

        # Create mock items with monotonic page numbers
        mock_items = []
        for page in [1, 1, 2, 2, 3, 3, 4]:
            item = MagicMock()
            prov = MagicMock()
            prov.page_no = page
            item.prov = [prov]
            mock_items.append((item, 0))

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = mock_items

        assert validate_provenance_monotonicity(mock_doc) is True

    def test_monotonicity_reset_invalid(self):
        """Test that page number reset (1,2,1,2) is invalid."""
        from pdf_splitter.reassembly import validate_provenance_monotonicity

        # Create mock items with resetting page numbers
        mock_items = []
        for page in [1, 2, 3, 1, 2]:  # Reset indicates concatenation failure
            item = MagicMock()
            prov = MagicMock()
            prov.page_no = page
            item.prov = [prov]
            mock_items.append((item, 0))

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = mock_items

        assert validate_provenance_monotonicity(mock_doc) is False


class TestExtractProvenancePages:
    """Tests for provenance page extraction."""

    def test_extract_pages_from_items(self):
        """Test extraction of page numbers from document items."""
        from pdf_splitter.reassembly import extract_provenance_pages

        mock_items = []
        for page in [1, 2, 2, 3]:
            item = MagicMock()
            prov = MagicMock()
            prov.page_no = page
            item.prov = [prov]
            mock_items.append((item, 0))

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = mock_items

        pages = extract_provenance_pages(mock_doc)
        assert pages == [1, 2, 2, 3]

    def test_extract_pages_handles_missing_prov(self):
        """Test handling of items without provenance."""
        from pdf_splitter.reassembly import extract_provenance_pages

        item_with_prov = MagicMock()
        prov = MagicMock()
        prov.page_no = 1
        item_with_prov.prov = [prov]

        item_without_prov = MagicMock()
        item_without_prov.prov = None

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [
            (item_with_prov, 0),
            (item_without_prov, 0),
        ]

        pages = extract_provenance_pages(mock_doc)
        assert pages == [1]


class TestGetMergeStatistics:
    """Tests for merge statistics calculation."""

    def test_statistics_empty_document(self):
        """Test statistics for empty document."""
        from pdf_splitter.reassembly import get_merge_statistics

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = []

        stats = get_merge_statistics(mock_doc)

        assert stats["total_items"] == 0
        assert stats["tables"] == 0
        assert stats["text_items"] == 0
        assert stats["unique_pages"] == 0

    def test_statistics_counts_item_types(self):
        """Test that statistics correctly count item types."""
        from pdf_splitter.reassembly import get_merge_statistics

        # Create mock items of different types
        table_item = MagicMock()
        table_item.__class__.__name__ = "TableItem"
        table_item.prov = []

        text_item = MagicMock()
        text_item.__class__.__name__ = "TextItem"
        text_item.prov = []

        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [
            (table_item, 0),
            (text_item, 0),
            (text_item, 0),
        ]

        stats = get_merge_statistics(mock_doc)

        assert stats["total_items"] == 3
        assert stats["tables"] == 1
        assert stats["text_items"] == 2
