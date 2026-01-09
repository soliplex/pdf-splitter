"""
Unit tests for segmentation module.

Tests split logic to verify segmentation does not lose pages
and properly handles bookmarks and fallback splitting.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetSplitBoundaries:
    """Tests for get_split_boundaries function."""

    def test_empty_pdf_returns_empty_list(self):
        """Test that empty PDF returns empty boundary list."""
        from pdf_splitter.segmentation import get_split_boundaries

        with patch("pdf_splitter.segmentation.PdfReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.pages = []
            mock_reader_class.return_value = mock_reader

            boundaries = get_split_boundaries(Path("test.pdf"))
            assert boundaries == []

    def test_single_page_pdf(self):
        """Test that single page PDF returns single boundary."""
        from pdf_splitter.segmentation import get_split_boundaries

        with patch("pdf_splitter.segmentation.PdfReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.pages = [MagicMock()]  # 1 page
            mock_reader.outline = []
            mock_reader_class.return_value = mock_reader

            boundaries = get_split_boundaries(Path("test.pdf"))
            assert boundaries == [(0, 1)]


class TestFixedBoundaries:
    """Tests for fixed-range splitting with overlap."""

    def test_fixed_boundaries_basic(self):
        """Test basic fixed boundary generation."""
        from pdf_splitter.segmentation import _get_fixed_boundaries

        # 100 pages, chunk_size=50, overlap=5
        boundaries = _get_fixed_boundaries(100, 50, 5)

        # Should produce: (0,50), (45,95), (90,100)
        assert len(boundaries) == 3
        assert boundaries[0] == (0, 50)
        assert boundaries[1] == (45, 95)
        assert boundaries[2] == (90, 100)

    def test_fixed_boundaries_no_overlap(self):
        """Test fixed boundaries without overlap."""
        from pdf_splitter.segmentation import _get_fixed_boundaries

        boundaries = _get_fixed_boundaries(100, 50, 0)
        assert boundaries == [(0, 50), (50, 100)]

    def test_fixed_boundaries_small_pdf(self):
        """Test boundaries for PDF smaller than chunk size."""
        from pdf_splitter.segmentation import _get_fixed_boundaries

        boundaries = _get_fixed_boundaries(30, 50, 5)
        assert boundaries == [(0, 30)]

    def test_fixed_boundaries_exact_chunk_size(self):
        """Test when total pages is exact multiple of chunk size."""
        from pdf_splitter.segmentation import _get_fixed_boundaries

        boundaries = _get_fixed_boundaries(100, 25, 5)
        # Should cover: (0,25), (20,45), (40,65), (60,85), (80,100)
        assert len(boundaries) == 5
        assert boundaries[0] == (0, 25)
        assert boundaries[-1][1] == 100


class TestPageCoverage:
    """Tests for page coverage verification."""

    def test_page_coverage_complete(self):
        """Test that page coverage detects complete coverage."""
        from pdf_splitter.segmentation import get_page_coverage

        boundaries = [(0, 50), (45, 100)]
        assert get_page_coverage(boundaries, 100) is True

    def test_page_coverage_with_gap(self):
        """Test that page coverage detects gaps."""
        from pdf_splitter.segmentation import get_page_coverage

        boundaries = [(0, 40), (50, 100)]  # Gap at 40-49
        assert get_page_coverage(boundaries, 100) is False

    def test_page_coverage_missing_end(self):
        """Test that page coverage detects missing end pages."""
        from pdf_splitter.segmentation import get_page_coverage

        boundaries = [(0, 50), (45, 90)]  # Missing 90-99
        assert get_page_coverage(boundaries, 100) is False

    def test_page_coverage_empty(self):
        """Test empty boundaries for empty document."""
        from pdf_splitter.segmentation import get_page_coverage

        assert get_page_coverage([], 0) is True
        assert get_page_coverage([], 10) is False


class TestBookmarkBoundaries:
    """Tests for bookmark-based splitting."""

    def test_bookmark_boundaries_extraction(self):
        """Test extraction of boundaries from bookmarks."""
        from pdf_splitter.segmentation import _get_bookmark_boundaries

        mock_reader = MagicMock()
        mock_reader.outline = [
            MagicMock(),  # Chapter 1 -> page 0
            MagicMock(),  # Chapter 2 -> page 25
            MagicMock(),  # Chapter 3 -> page 60
        ]

        # Mock get_destination_page_number to return sequential pages
        mock_reader.get_destination_page_number = MagicMock(side_effect=[0, 25, 60])

        boundaries = _get_bookmark_boundaries(mock_reader, 100)

        assert (0, 25) in boundaries
        assert (25, 60) in boundaries
        assert (60, 100) in boundaries

    def test_bookmark_boundaries_empty_outline(self):
        """Test fallback when no bookmarks exist."""
        from pdf_splitter.segmentation import _get_bookmark_boundaries

        mock_reader = MagicMock()
        mock_reader.outline = []

        boundaries = _get_bookmark_boundaries(mock_reader, 100)
        assert boundaries == []

    def test_bookmark_boundaries_none_outline(self):
        """Test fallback when outline is None."""
        from pdf_splitter.segmentation import _get_bookmark_boundaries

        mock_reader = MagicMock()
        mock_reader.outline = None

        boundaries = _get_bookmark_boundaries(mock_reader, 100)
        assert boundaries == []


class TestSplitPdf:
    """Tests for actual PDF splitting."""

    @pytest.fixture
    def sample_pdf(self):
        """Create a simple test PDF."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            # Create 10 blank pages
            for _ in range(10):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            pdf_path = Path(f.name)

        yield pdf_path

        # Cleanup
        if pdf_path.exists():
            pdf_path.unlink()

    def test_split_pdf_creates_chunks(self, sample_pdf):
        """Test that split_pdf creates chunk files."""
        from pdf_splitter.segmentation import split_pdf

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            chunks = split_pdf(sample_pdf, output_dir, chunk_size=3, overlap=1)

            assert len(chunks) > 0
            for chunk_path in chunks:
                assert chunk_path.exists()
                assert chunk_path.suffix == ".pdf"

    def test_split_pdf_coverage(self, sample_pdf):
        """Test that split preserves all pages."""
        from pypdf import PdfReader

        from pdf_splitter.segmentation import get_page_coverage, get_split_boundaries

        boundaries = get_split_boundaries(sample_pdf, chunk_size=3, overlap=1)
        total_pages = len(PdfReader(str(sample_pdf)).pages)

        assert get_page_coverage(boundaries, total_pages) is True
