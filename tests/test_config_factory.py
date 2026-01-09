"""
Unit tests for config_factory module.

Tests configuration integrity to ensure heavy features (OCR, Images)
are disabled and proper backend is selected.
"""

from unittest.mock import patch

import pytest


class TestConfigFactory:
    """Tests for configuration factory module."""

    def test_pipeline_options_ocr_disabled(self):
        """Test that OCR is disabled in pipeline options."""
        from pdf_splitter.config_factory import get_pipeline_options

        opts = get_pipeline_options()
        assert opts.do_ocr is False, "OCR should be disabled for digital-native PDFs"

    def test_pipeline_options_table_mode_fast(self):
        """Test that table mode is set to FAST."""
        from docling.datamodel.pipeline_options import TableFormerMode

        from pdf_splitter.config_factory import get_pipeline_options

        opts = get_pipeline_options()
        assert (
            opts.table_structure_options.mode == TableFormerMode.FAST
        ), "Table mode should be FAST for 2x speedup"

    def test_pipeline_options_images_disabled(self):
        """Test that image generation is disabled to prevent OOM."""
        from pdf_splitter.config_factory import get_pipeline_options

        opts = get_pipeline_options()
        assert opts.generate_page_images is False, "Page image generation should be disabled"
        assert opts.generate_picture_images is False, "Picture image generation should be disabled"

    @patch("pdf_splitter.config_factory.DocumentConverter")
    def test_create_converter_uses_v2_backend(self, mock_converter_class):
        """Test that create_converter uses DoclingParseV2DocumentBackend."""
        from docling.datamodel.base_models import InputFormat

        from pdf_splitter.config_factory import create_converter

        create_converter()

        # Verify DocumentConverter was called
        mock_converter_class.assert_called_once()

        # Get the format_options passed to constructor
        call_kwargs = mock_converter_class.call_args[1]
        assert "format_options" in call_kwargs

        format_options = call_kwargs["format_options"]
        assert InputFormat.PDF in format_options

        # Check that the backend is DoclingParseV2DocumentBackend
        pdf_option = format_options[InputFormat.PDF]
        from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

        assert (
            pdf_option.backend == DoclingParseV2DocumentBackend
        ), "Should use V2 backend for 10x faster loading"

    @patch("pdf_splitter.config_factory.DocumentConverter")
    def test_create_converter_pipeline_options(self, mock_converter_class):
        """Test that converter is created with correct pipeline options."""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import TableFormerMode

        from pdf_splitter.config_factory import create_converter

        create_converter()

        call_kwargs = mock_converter_class.call_args[1]
        pdf_option = call_kwargs["format_options"][InputFormat.PDF]
        pipeline_opts = pdf_option.pipeline_options

        assert pipeline_opts.do_ocr is False
        assert pipeline_opts.table_structure_options.mode == TableFormerMode.FAST
        assert pipeline_opts.generate_page_images is False
        assert pipeline_opts.generate_picture_images is False


class TestConfigFactoryIntegration:
    """Integration tests that actually create the converter (requires docling)."""

    @pytest.mark.integration
    def test_create_converter_returns_valid_instance(self):
        """Test that create_converter returns a valid DocumentConverter."""
        from docling.document_converter import DocumentConverter

        from pdf_splitter.config_factory import create_converter

        converter = create_converter()
        assert isinstance(converter, DocumentConverter)
