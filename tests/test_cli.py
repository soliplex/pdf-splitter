"""
Tests for CLI commands.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCLIChunk:
    """Tests for chunk CLI command."""

    @pytest.fixture
    def test_pdf(self):
        """Create a test PDF."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            writer = PdfWriter()
            for _ in range(50):
                writer.add_blank_page(width=612, height=792)
            writer.write(f)
            pdf_path = Path(f.name)

        yield pdf_path
        pdf_path.unlink(missing_ok=True)

    def test_chunk_command_parallel_default(self, test_pdf):
        """Test chunk command uses parallel by default."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_chunk

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                pdf=str(test_pdf),
                output=tmpdir,
                max_pages=20,
                min_pages=5,
                overlap=0,
                strategy=None,
                workers=None,
                sequential=False,
                verbose=False,
            )

            result = cmd_chunk(args)
            assert result == 0

            # Check chunks were created
            chunks = list(Path(tmpdir).glob("*.pdf"))
            assert len(chunks) >= 2

    def test_chunk_command_sequential_flag(self, test_pdf):
        """Test chunk command with --sequential flag."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_chunk

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                pdf=str(test_pdf),
                output=tmpdir,
                max_pages=20,
                min_pages=5,
                overlap=0,
                strategy=None,
                workers=None,
                sequential=True,
                verbose=False,
            )

            result = cmd_chunk(args)
            assert result == 0

            chunks = list(Path(tmpdir).glob("*.pdf"))
            assert len(chunks) >= 2

    def test_chunk_command_custom_workers(self, test_pdf):
        """Test chunk command with custom workers."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_chunk

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                pdf=str(test_pdf),
                output=tmpdir,
                max_pages=20,
                min_pages=5,
                overlap=0,
                strategy=None,
                workers=2,
                sequential=False,
                verbose=False,
            )

            result = cmd_chunk(args)
            assert result == 0

    def test_chunk_command_file_not_found(self):
        """Test chunk command with non-existent file."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_chunk

        args = Namespace(
            pdf="nonexistent.pdf",
            output="./output",
            max_pages=20,
            min_pages=5,
            overlap=0,
            strategy=None,
            workers=None,
            sequential=False,
            verbose=False,
        )

        result = cmd_chunk(args)
        assert result == 1


class TestCLIConvert:
    """Tests for convert CLI command."""

    @pytest.fixture
    def chunk_dir(self):
        """Create a directory with test chunk PDFs."""
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 small chunk PDFs
            for i in range(3):
                pdf_path = tmpdir / f"chunk_{i:04d}.pdf"
                writer = PdfWriter()
                for _ in range(5):
                    writer.add_blank_page(width=612, height=792)
                with open(pdf_path, "wb") as f:
                    writer.write(f)

            yield tmpdir

    def test_convert_command_basic(self, chunk_dir):
        """Test convert command on directory of chunks."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_convert

        args = Namespace(
            input=str(chunk_dir),
            output=None,
            workers=1,
            maxtasks=1,
            verbose=False,
            keep_parts=False,
        )

        # Mock the entire BatchProcessor to avoid pickling issues
        with patch("pdf_splitter.processor.BatchProcessor") as MockBatchProcessor:
            mock_instance = MagicMock()
            mock_instance.execute_parallel.return_value = [
                {
                    "success": True,
                    "chunk_path": str(chunk_dir / "chunk_0000.pdf"),
                    "document_dict": {},
                    "error": None,
                },
                {
                    "success": True,
                    "chunk_path": str(chunk_dir / "chunk_0001.pdf"),
                    "document_dict": {},
                    "error": None,
                },
                {
                    "success": True,
                    "chunk_path": str(chunk_dir / "chunk_0002.pdf"),
                    "document_dict": {},
                    "error": None,
                },
            ]
            MockBatchProcessor.return_value = mock_instance

            result = cmd_convert(args)
            assert result == 0

    def test_convert_command_not_found(self):
        """Test convert command with non-existent path."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_convert

        args = Namespace(
            input="nonexistent_dir",
            output=None,
            workers=1,
            maxtasks=1,
            verbose=False,
            keep_parts=False,
        )

        result = cmd_convert(args)
        assert result == 1

    def test_convert_command_empty_dir(self):
        """Test convert command with empty directory."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_convert

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                input=tmpdir, output=None, workers=1, maxtasks=1, verbose=False, keep_parts=False
            )

            result = cmd_convert(args)
            assert result == 1

    def test_convert_command_with_output(self, chunk_dir):
        """Test convert command writes output JSON with --keep-parts."""
        import json
        from argparse import Namespace

        from pdf_splitter.cli import cmd_convert

        with tempfile.TemporaryDirectory() as out_dir:
            output_path = Path(out_dir) / "results.json"

            args = Namespace(
                input=str(chunk_dir),
                output=str(output_path),
                workers=1,
                maxtasks=1,
                verbose=False,
                keep_parts=True,  # Skip merge, just output raw results
            )

            # Mock the entire BatchProcessor to avoid pickling issues
            with patch("pdf_splitter.processor.BatchProcessor") as MockBatchProcessor:
                mock_instance = MagicMock()
                mock_instance.execute_parallel.return_value = [
                    {
                        "success": True,
                        "chunk_path": str(chunk_dir / "chunk_0000.pdf"),
                        "document_dict": {"content": "test0"},
                        "error": None,
                    },
                    {
                        "success": True,
                        "chunk_path": str(chunk_dir / "chunk_0001.pdf"),
                        "document_dict": {"content": "test1"},
                        "error": None,
                    },
                    {
                        "success": True,
                        "chunk_path": str(chunk_dir / "chunk_0002.pdf"),
                        "document_dict": {"content": "test2"},
                        "error": None,
                    },
                ]
                MockBatchProcessor.return_value = mock_instance

                result = cmd_convert(args)
                assert result == 0
                assert output_path.exists()

                # Verify JSON structure
                with open(output_path) as f:
                    data = json.load(f)
                    assert isinstance(data, list)
                    assert len(data) == 3


class TestCLIValidate:
    """Tests for validate CLI command."""

    @pytest.fixture
    def valid_docling_output(self):
        """Create a valid Docling output JSON and matching chunks."""
        import json

        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunks_dir = tmpdir / "chunks"
            chunks_dir.mkdir()

            # Create chunk PDFs with proper naming
            for i in range(2):
                start_page = i * 10 + 1
                end_page = (i + 1) * 10
                pdf_path = chunks_dir / f"chunk_{i:04d}_pages_{start_page:04d}_{end_page:04d}.pdf"
                writer = PdfWriter()
                for _ in range(10):
                    writer.add_blank_page(width=612, height=792)
                with open(pdf_path, "wb") as f:
                    writer.write(f)

            # Create matching Docling JSON
            docling_data = []
            for i in range(2):
                start_page = i * 10 + 1
                end_page = (i + 1) * 10
                docling_data.append(
                    {
                        "chunk_path": str(
                            chunks_dir / f"chunk_{i:04d}_pages_{start_page:04d}_{end_page:04d}.pdf"
                        ),
                        "success": True,
                        "document_dict": {
                            "texts": [{"prov": [{"page_no": p}]} for p in range(1, 11)],
                            "tables": [],
                            "pictures": [],
                        },
                    }
                )

            json_path = tmpdir / "docling.json"
            with open(json_path, "w") as f:
                json.dump(docling_data, f)

            yield json_path, chunks_dir

    def test_validate_command_success(self, valid_docling_output):
        """Test validate command with valid output."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_validate

        json_path, chunks_dir = valid_docling_output

        args = Namespace(json=str(json_path), chunks=str(chunks_dir), verbose=False)

        result = cmd_validate(args)
        assert result == 0

    def test_validate_command_verbose(self, valid_docling_output):
        """Test validate command with verbose flag."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_validate

        json_path, chunks_dir = valid_docling_output

        args = Namespace(json=str(json_path), chunks=str(chunks_dir), verbose=True)

        result = cmd_validate(args)
        assert result == 0

    def test_validate_command_json_not_found(self):
        """Test validate command with non-existent JSON."""
        from argparse import Namespace

        from pdf_splitter.cli import cmd_validate

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(json="nonexistent.json", chunks=tmpdir, verbose=False)

            result = cmd_validate(args)
            assert result == 1

    def test_validate_command_chunks_not_found(self):
        """Test validate command with non-existent chunks dir."""
        import json
        from argparse import Namespace

        from pdf_splitter.cli import cmd_validate

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test.json"
            with open(json_path, "w") as f:
                json.dump([], f)

            args = Namespace(json=str(json_path), chunks="nonexistent_dir", verbose=False)

            result = cmd_validate(args)
            assert result == 1

    def test_validate_command_failed_chunks(self):
        """Test validate command with failed chunks in output."""
        import json
        from argparse import Namespace

        from pypdf import PdfWriter

        from pdf_splitter.cli import cmd_validate

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunks_dir = tmpdir / "chunks"
            chunks_dir.mkdir()

            # Create one chunk
            pdf_path = chunks_dir / "chunk_0000_pages_0001_0010.pdf"
            writer = PdfWriter()
            for _ in range(10):
                writer.add_blank_page(width=612, height=792)
            with open(pdf_path, "wb") as f:
                writer.write(f)

            # Create Docling JSON with failure
            docling_data = [{"chunk_path": str(pdf_path), "success": False, "error": "Test error"}]

            json_path = tmpdir / "docling.json"
            with open(json_path, "w") as f:
                json.dump(docling_data, f)

            args = Namespace(json=str(json_path), chunks=str(chunks_dir), verbose=False)

            result = cmd_validate(args)
            assert result == 1


class TestCLIMain:
    """Tests for main CLI entry point."""

    def test_main_no_command(self):
        """Test main with no command shows help."""
        from pdf_splitter.cli import main

        with patch.object(sys, "argv", ["pdf-splitter"]):
            result = main()
            assert result == 1

    def test_main_help(self):
        """Test --help flag."""
        from pdf_splitter.cli import main

        with patch.object(sys, "argv", ["pdf-splitter", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_chunk_help(self):
        """Test chunk --help."""
        from pdf_splitter.cli import main

        with patch.object(sys, "argv", ["pdf-splitter", "chunk", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_convert_help(self):
        """Test convert --help."""
        from pdf_splitter.cli import main

        with patch.object(sys, "argv", ["pdf-splitter", "convert", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_validate_help(self):
        """Test validate --help."""
        from pdf_splitter.cli import main

        with patch.object(sys, "argv", ["pdf-splitter", "validate", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
