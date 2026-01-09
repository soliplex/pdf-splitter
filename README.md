# PDF Splitter

[![CI](https://github.com/soliplex/pdf-splitter/actions/workflows/soliplex.yaml/badge.svg)](https://github.com/soliplex/pdf-splitter/actions/workflows/soliplex.yaml)
[![codecov](https://codecov.io/gh/soliplex/pdf-splitter/branch/main/graph/badge.svg)](https://codecov.io/gh/soliplex/pdf-splitter)

Split-Process-Merge pipeline for converting large PDFs to Docling documents.

## Installation

**With pip:**
```bash
pip install pdf-splitter            # core only
pip install pdf-splitter[dev]       # with test/lint tools
```

**With uv:**
```bash
uv add pdf-splitter                 # core only
uv add pdf-splitter --group dev     # with test/lint tools
```

**From source:**
```bash
git clone https://github.com/soliplex/pdf-splitter.git
cd pdf-splitter

# pip
pip install -e .                    # core only
pip install -e ".[dev]"             # with dev tools

# uv
uv sync                             # core only
uv sync --group dev                 # with dev tools
```

Requires Python 3.12+.

## Usage

```bash
pdf-splitter analyze doc.pdf              # analyze structure
pdf-splitter chunk doc.pdf -o ./chunks    # split into chunks
pdf-splitter convert ./chunks -o out.json # process & merge
pdf-splitter validate out.json ./chunks   # validate output
```

### Options

| Option | Description |
|--------|-------------|
| `-v` | Verbose logging |
| `-s <strategy>` | Force: `fixed`, `hybrid`, `enhanced` |
| `--max-pages N` | Max pages per chunk (default: 100) |
| `-w N` | Worker processes |
| `--keep-parts` | Output individual chunks |

## Python API

```python
from pdf_splitter.segmentation_enhanced import smart_split_to_files
from pdf_splitter.processor import BatchProcessor
from pdf_splitter.reassembly import merge_from_results

chunks, _ = smart_split_to_files("doc.pdf", output_dir="./chunks")
results = BatchProcessor(max_workers=4).execute_parallel(chunks)
merged = merge_from_results(results)
merged.export_to_json("output.json")
```

## Development

```bash
make help       # show commands
make test       # run tests
make lint       # run ruff
make typecheck  # run mypy
make quality    # all checks
```
