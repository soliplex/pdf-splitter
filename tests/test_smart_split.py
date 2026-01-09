"""
Tests for smart_split() unified function.
"""

import tempfile
from pathlib import Path

import pytest


class TestSplitResult:
    """Tests for SplitResult dataclass."""

    def test_split_result_creation(self):
        """Test SplitResult can be created with all fields."""
        from pdf_splitter.segmentation_enhanced import SplitResult

        result = SplitResult(
            boundaries=[(0, 50), (45, 100)],
            strategy="fixed",
            total_pages=100,
            num_chunks=2,
            min_chunk_size=50,
            max_chunk_size=55,
            avg_chunk_size=52.5,
            has_overlap=True,
        )

        assert result.num_chunks == 2
        assert result.strategy == "fixed"
        assert result.has_overlap is True

    def test_split_result_summary(self):
        """Test SplitResult summary method."""
        from pdf_splitter.segmentation_enhanced import SplitResult

        result = SplitResult(
            boundaries=[(0, 100)],
            strategy="single_chunk",
            total_pages=100,
            num_chunks=1,
            min_chunk_size=100,
            max_chunk_size=100,
            avg_chunk_size=100.0,
            has_overlap=False,
        )

        summary = result.summary()
        assert "single_chunk" in summary
        assert "100" in summary


class TestSmartSplit:
    """Tests for smart_split() function."""

    @pytest.fixture
    def small_pdf(self):
        """Create a small test PDF (50 pages)."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            for _ in range(50):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            pdf_path = Path(f.name)

        yield pdf_path
        pdf_path.unlink(missing_ok=True)

    @pytest.fixture
    def medium_pdf(self):
        """Create a medium test PDF (150 pages)."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            for _ in range(150):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            pdf_path = Path(f.name)

        yield pdf_path
        pdf_path.unlink(missing_ok=True)

    @pytest.fixture
    def large_pdf(self):
        """Create a large test PDF (500 pages)."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            for _ in range(500):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            pdf_path = Path(f.name)

        yield pdf_path
        pdf_path.unlink(missing_ok=True)

    def test_small_pdf_single_chunk(self, small_pdf):
        """Test that small PDFs return single chunk."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(small_pdf, max_chunk_pages=100)

        assert result.num_chunks == 1
        assert result.strategy == "single_chunk"
        assert result.total_pages == 50

    def test_medium_pdf_fixed_small(self, medium_pdf):
        """Test that medium PDFs use fixed_small strategy."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(medium_pdf, max_chunk_pages=100)

        assert result.num_chunks >= 2
        assert "fixed" in result.strategy
        assert result.total_pages == 150

    def test_large_pdf_creates_multiple_chunks(self, large_pdf):
        """Test that large PDFs create multiple balanced chunks."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(large_pdf, max_chunk_pages=100)

        assert result.num_chunks >= 5
        assert result.max_chunk_size <= 105  # Allow small overflow
        assert result.total_pages == 500

    def test_force_strategy_fixed(self, large_pdf):
        """Test forcing fixed strategy."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(large_pdf, max_chunk_pages=100, force_strategy="fixed")

        assert result.strategy == "fixed"
        assert result.num_chunks >= 5

    def test_invalid_force_strategy_raises(self, small_pdf):
        """Test that invalid strategy raises ValueError."""
        from pdf_splitter.segmentation_enhanced import smart_split

        with pytest.raises(ValueError, match="Unknown strategy"):
            smart_split(small_pdf, force_strategy="invalid_strategy")

    def test_invalid_max_chunk_pages_raises(self, small_pdf):
        """Test that max_chunk_pages < 1 raises ValueError."""
        from pdf_splitter.segmentation_enhanced import smart_split

        with pytest.raises(ValueError, match="max_chunk_pages must be >= 1"):
            smart_split(small_pdf, max_chunk_pages=0)

        with pytest.raises(ValueError, match="max_chunk_pages must be >= 1"):
            smart_split(small_pdf, max_chunk_pages=-1)

    def test_invalid_overlap_raises(self, small_pdf):
        """Test that overlap < 0 raises ValueError."""
        from pdf_splitter.segmentation_enhanced import smart_split

        with pytest.raises(ValueError, match="overlap must be >= 0"):
            smart_split(small_pdf, overlap=-1)

    def test_boundaries_cover_all_pages(self, large_pdf):
        """Test that boundaries cover all pages."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(large_pdf, max_chunk_pages=100)

        # Check all pages are covered
        covered = set()
        for start, end in result.boundaries:
            for page in range(start, end):
                covered.add(page)

        expected = set(range(result.total_pages))
        assert covered == expected

    def test_overlap_detection(self, large_pdf):
        """Test overlap detection in results."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(large_pdf, max_chunk_pages=100, overlap=5)

        # With overlap, has_overlap should be True for fixed strategy
        if "fixed" in result.strategy:
            assert result.has_overlap is True

    def test_overlap_greater_than_chunk_size(self, small_pdf):
        """Test that overlap >= chunk_size doesn't break splitting."""
        from pdf_splitter.segmentation_enhanced import smart_split

        # overlap (5) > max_chunk_pages (2) - should still create multiple chunks
        result = smart_split(small_pdf, max_chunk_pages=2, overlap=5)

        # 50-page PDF with 2-page chunks should create many chunks
        assert result.num_chunks > 1
        assert result.max_chunk_size == 2
        # All pages should be covered
        covered = set()
        for start, end in result.boundaries:
            for page in range(start, end):
                covered.add(page)
        assert covered == set(range(result.total_pages))


class TestSmartSplitToFiles:
    """Tests for smart_split_to_files() function."""

    @pytest.fixture
    def test_pdf(self):
        """Create a test PDF."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            for _ in range(100):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            pdf_path = Path(f.name)

        yield pdf_path
        pdf_path.unlink(missing_ok=True)

    def test_creates_chunk_files(self, test_pdf):
        """Test that chunk files are created."""
        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            chunk_paths, _result = smart_split_to_files(test_pdf, output_dir, max_chunk_pages=50)

            assert len(chunk_paths) >= 2
            for path in chunk_paths:
                assert path.exists()
                assert path.suffix == ".pdf"

    def test_chunk_files_have_correct_pages(self, test_pdf):
        """Test that chunk files contain correct number of pages."""
        from pypdf import PdfReader

        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            chunk_paths, result = smart_split_to_files(test_pdf, output_dir, max_chunk_pages=50)

            total_chunk_pages = 0
            for i, path in enumerate(chunk_paths):
                reader = PdfReader(str(path))
                chunk_pages = len(reader.pages)

                # Check matches boundary
                start, end = result.boundaries[i]
                expected_pages = end - start
                assert chunk_pages == expected_pages

                total_chunk_pages += chunk_pages

            # Total should cover all pages (may have overlap)
            assert total_chunk_pages >= result.total_pages

    def test_uses_temp_dir_when_no_output(self, test_pdf):
        """Test that temp directory is used when output_dir is None."""
        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        chunk_paths, _result = smart_split_to_files(test_pdf, output_dir=None, max_chunk_pages=50)

        assert len(chunk_paths) >= 1
        # Should be in a temp directory
        assert "pdf_chunks_" in str(chunk_paths[0].parent)

        # Cleanup
        for path in chunk_paths:
            path.unlink(missing_ok=True)
        chunk_paths[0].parent.rmdir()


class TestParallelChunkWriting:
    """Tests for parallel chunk writing functionality."""

    @pytest.fixture
    def test_pdf(self):
        """Create a test PDF with 100 pages."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            for _ in range(100):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            pdf_path = Path(f.name)

        yield pdf_path
        pdf_path.unlink(missing_ok=True)

    def test_parallel_creates_same_chunks_as_sequential(self, test_pdf):
        """Test that parallel and sequential produce same results."""
        from pypdf import PdfReader

        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            # Parallel
            parallel_paths, _parallel_result = smart_split_to_files(
                test_pdf, Path(tmpdir1), max_chunk_pages=25, parallel=True
            )

            # Sequential
            seq_paths, _seq_result = smart_split_to_files(
                test_pdf, Path(tmpdir2), max_chunk_pages=25, parallel=False
            )

            # Same number of chunks
            assert len(parallel_paths) == len(seq_paths)

            # Same page counts in each chunk
            for p_path, s_path in zip(sorted(parallel_paths), sorted(seq_paths), strict=True):
                p_reader = PdfReader(str(p_path))
                s_reader = PdfReader(str(s_path))
                assert len(p_reader.pages) == len(s_reader.pages)

    def test_parallel_with_custom_workers(self, test_pdf):
        """Test parallel writing with custom worker count."""
        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_paths, _result = smart_split_to_files(
                test_pdf, Path(tmpdir), max_chunk_pages=20, max_workers=2, parallel=True
            )

            assert len(chunk_paths) >= 5
            for path in chunk_paths:
                assert path.exists()

    def test_sequential_mode_explicit(self, test_pdf):
        """Test explicit sequential mode."""
        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_paths, _result = smart_split_to_files(
                test_pdf, Path(tmpdir), max_chunk_pages=25, parallel=False
            )

            assert len(chunk_paths) >= 4
            for path in chunk_paths:
                assert path.exists()

    def test_single_chunk_uses_sequential(self, test_pdf):
        """Test that single chunk doesn't use parallel (no benefit)."""
        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # 100 pages, max 200 = single chunk
            chunk_paths, result = smart_split_to_files(
                test_pdf, Path(tmpdir), max_chunk_pages=200, parallel=True
            )

            assert len(chunk_paths) == 1
            assert result.num_chunks == 1

    def test_parallel_maintains_order(self, test_pdf):
        """Test that parallel writing maintains chunk order."""
        from pypdf import PdfReader

        from pdf_splitter.segmentation_enhanced import smart_split_to_files

        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_paths, result = smart_split_to_files(
                test_pdf, Path(tmpdir), max_chunk_pages=20, parallel=True
            )

            # Verify chunks are named in order and have expected pages
            for i, (path, (start, end)) in enumerate(
                zip(chunk_paths, result.boundaries, strict=True)
            ):
                expected_name = f"chunk_{i:04d}_pages_{start + 1:04d}_{end:04d}.pdf"
                assert path.name == expected_name

                reader = PdfReader(str(path))
                assert len(reader.pages) == end - start


class TestSmartSplitOnRealPDFs:
    """Integration tests using PDFs from assets folder."""

    @pytest.fixture
    def assets_dir(self):
        """Get assets directory."""
        return Path(__file__).parent.parent / "assets"

    @pytest.fixture
    def sample_pdf(self, assets_dir):
        """Get sample.pdf from assets (or first available PDF)."""
        sample = assets_dir / "sample.pdf"
        if sample.exists():
            return sample
        # Fall back to any PDF in assets
        pdfs = list(assets_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
        pytest.skip("No PDFs found in assets folder")

    @pytest.mark.integration
    def test_sample_pdf_splits(self, sample_pdf):
        """Test smart_split on sample PDF."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(sample_pdf, max_chunk_pages=20)

        assert result.total_pages > 0
        assert result.num_chunks >= 1
        assert result.max_chunk_size <= result.total_pages

    @pytest.mark.integration
    def test_small_chunk_size(self, sample_pdf):
        """Test splitting with small chunk size."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(sample_pdf, max_chunk_pages=10)

        if result.total_pages > 10:
            assert result.num_chunks >= 2

    @pytest.mark.integration
    def test_large_chunk_size(self, sample_pdf):
        """Test splitting with large chunk size."""
        from pdf_splitter.segmentation_enhanced import smart_split

        result = smart_split(sample_pdf, max_chunk_pages=1000)

        # Should be single chunk if doc is smaller than max
        if result.total_pages <= 1000:
            assert result.num_chunks == 1

    @pytest.mark.integration
    def test_all_pdfs_balanced(self, assets_dir):
        """Test that all PDFs get reasonably balanced splits."""
        from pdf_splitter.segmentation_enhanced import smart_split

        pdfs = list(assets_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs in assets folder")

        for pdf_path in pdfs:
            result = smart_split(pdf_path, max_chunk_pages=100)

            # No single chunk should exceed 50% of document for large docs
            if result.total_pages > 200:
                max_ratio = result.max_chunk_size / result.total_pages
                assert max_ratio < 0.5, f"max chunk is {max_ratio:.0%} of document"
