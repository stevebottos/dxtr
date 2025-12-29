"""
Utilities for processing Docling JSON exports

Provides functions to extract lightweight text-only versions from
full Docling JSON exports (which include bboxes and layout info).
"""

import json
from pathlib import Path
from typing import Dict, List, Any


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
