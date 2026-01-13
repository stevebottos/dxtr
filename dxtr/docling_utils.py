"""
Utilities for processing Docling JSON exports

Provides functions to extract lightweight text-only versions from
full Docling JSON exports (which include bboxes and layout info),
and local functions to run Docling conversion and embedding generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import BytesIO

# Lazy imports for heavy libraries
_converter = None
_embed_model = None


def _get_converter():
    """Lazy load Docling converter"""
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat

        # Configuration (optimized for arXiv papers)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # arXiv papers are text-based PDFs
        pipeline_options.do_table_structure = True
        pipeline_options.images_scale = 4.0
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
    return _converter


def _get_embed_device() -> str:
    """Determine device for embedding model based on GPU memory availability."""
    try:
        import torch
        if torch.cuda.is_available():
            # Check free GPU memory (need ~6GB for embedding model)
            free_mem_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            if free_mem_gb >= 6.0:
                return "cuda"
    except Exception:
        pass
    return "cpu"


def get_embed_model():
    """Lazy load embedding model (uses GPU if >=6GB available, else CPU)"""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
        device = _get_embed_device()
        print(f"Loading embedding model: {EMBED_MODEL_NAME} ({device.upper()})")
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
            trust_remote_code=True,
            device=device
        )
    return _embed_model


def convert_pdf_local(pdf_path: Path) -> Dict[str, Any]:
    """
    Convert a PDF file to Markdown format using local Docling instance.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict with 'markdown' and 'docling_json' keys
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    from docling.datamodel.base_models import DocumentStream

    converter = _get_converter()

    # Convert PDF
    # We pass the file path directly to convert() which is supported and efficient
    result = converter.convert(pdf_path)

    # Export to markdown
    markdown_content = result.document.export_to_markdown()

    # Export to Docling's JSON format
    docling_json = result.document.export_to_dict()

    return {
        "markdown": markdown_content,
        "docling_json": docling_json,
    }


def generate_embeddings_local(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using local model.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    model = get_embed_model()
    embeddings = model.get_text_embedding_batch(texts)
    return embeddings


def extract_text_only(docling_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a lightweight text-only version from full Docling JSON.

    This creates a "lossy" version that drops all bbox/provenance data
    but preserves the text content and document structure.

    Args:
        docling_json: Full Docling export from export_to_dict()

    Returns:
        Lightweight dict with hierarchical text content
    """
    texts = docling_json.get('texts', [])

    # Group content by type
    sections = []
    current_section = None

    for text_elem in texts:
        label = text_elem.get('label', 'unknown')
        text = text_elem.get('text', '')
        layer = text_elem.get('content_layer', 'unknown')

        # Skip furniture (headers/footers)
        if layer == 'furniture':
            continue

        if label == 'section_header':
            # Start new section
            if current_section:
                sections.append(current_section)
            current_section = {
                'title': text,
                'content': [],
                'subsections': []
            }
        elif label == 'text':
            # Add to current section or create default section
            if current_section is None:
                current_section = {
                    'title': '(Untitled)',
                    'content': [],
                    'subsections': []
                }
            current_section['content'].append(text)
        elif label == 'caption':
            if current_section:
                current_section['content'].append(f'[Figure/Table: {text}]')
        # Skip footnotes, page numbers, etc.

    # Add last section
    if current_section:
        sections.append(current_section)

    # Extract tables (text only)
    tables = []
    for table in docling_json.get('tables', []):
        # Tables have a grid structure with cells
        data = table.get('data', {})
        grid = data.get('grid', [])

        if grid:
            # Extract text from grid cells
            rows = []
            for row in grid:
                row_texts = [cell.get('text', '') for cell in row]
                rows.append(row_texts)

            # Get caption if available
            caption = ''
            if table.get('captions'):
                # Captions are references like {'$ref': '#/texts/42'}
                # For now, just note that there's a caption
                caption = '(See caption in document)'

            tables.append({
                'caption': caption,
                'num_rows': data.get('num_rows', len(rows)),
                'num_cols': data.get('num_cols', len(rows[0]) if rows else 0),
                'grid': rows
            })

    # Metadata
    metadata = {
        'name': docling_json.get('name', 'unknown'),
        'num_pages': len(docling_json.get('pages', [])),
        'num_pictures': len(docling_json.get('pictures', [])),
        'num_tables': len(docling_json.get('tables', [])),
    }

    return {
        'metadata': metadata,
        'sections': sections,
        'tables': tables,
    }


def save_text_only(docling_json: Dict[str, Any], output_path: Path):
    """
    Extract and save text-only version from full Docling JSON.

    Args:
        docling_json: Full Docling export
        output_path: Where to save the text-only JSON
    """
    text_only = extract_text_only(docling_json)
    output_path.write_text(
        json.dumps(text_only, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )


def load_paper_text(paper_dir: Path) -> Dict[str, Any]:
    """
    Load the text-only version of a paper.

    Args:
        paper_dir: Directory containing paper files

    Returns:
        Text-only JSON dict, or None if not found
    """
    text_json_path = paper_dir / "paper_text.json"
    if not text_json_path.exists():
        return None

    return json.loads(text_json_path.read_text(encoding='utf-8'))


def get_section_by_title(paper_text: Dict[str, Any], title_pattern: str) -> Dict[str, Any]:
    """
    Find a section by title (case-insensitive partial match).

    Args:
        paper_text: Text-only paper JSON
        title_pattern: Pattern to match in section titles

    Returns:
        Section dict or None
    """
    pattern_lower = title_pattern.lower()
    for section in paper_text.get('sections', []):
        if pattern_lower in section['title'].lower():
            return section
    return None


def skip_references_section(paper_text: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove references/bibliography sections from paper text.

    Args:
        paper_text: Text-only paper JSON

    Returns:
        Modified paper_text with references removed
    """
    ref_keywords = ['reference', 'bibliography', 'citations']

    filtered_sections = []
    for section in paper_text.get('sections', []):
        title_lower = section['title'].lower()
        if any(kw in title_lower for kw in ref_keywords):
            continue  # Skip this section
        filtered_sections.append(section)

    paper_text['sections'] = filtered_sections
    return paper_text
