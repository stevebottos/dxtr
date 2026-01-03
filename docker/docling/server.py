"""
Docling PDF to Markdown Conversion Service

A FastAPI-based HTTP service that converts PDF files to Markdown format
and generates embeddings for text chunks.

Endpoints:
----------
POST /convert - Convert a PDF file to markdown
POST /embed - Generate embeddings for text chunks
GET /health - Health check endpoint
GET / - API documentation
"""

import io
import tempfile
from pathlib import Path
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

app = FastAPI(
    title="Docling PDF Converter",
    description="Convert PDF files to Markdown using Docling",
    version="1.0.0"
)

# Configuration (optimized for arXiv papers - quality over speed)
PDF_BACKEND = "pypdfium2"
DO_OCR = False  # arXiv papers are text-based PDFs
DO_TABLE_STRUCTURE = True
IMAGES_SCALE = 4.0
IMAGE_EXPORT_FORMAT = "png"

# Initialize converter at startup
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = DO_OCR
pipeline_options.do_table_structure = DO_TABLE_STRUCTURE
pipeline_options.images_scale = IMAGES_SCALE
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = True

backend_map = {
    "pypdfium2": PyPdfiumDocumentBackend,
    "dlparse_v1": DoclingParseDocumentBackend,
    "dlparse_v2": DoclingParseDocumentBackend
}
selected_backend = backend_map.get(PDF_BACKEND, PyPdfiumDocumentBackend)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=selected_backend
        )
    }
)

# Embedding model configuration
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"  # 8192 token context, 768 dims
embed_model = None  # Lazy load on first request


def get_embed_model():
    """Lazy load embedding model on first use"""
    global embed_model
    if embed_model is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, trust_remote_code=True)
    return embed_model


class EmbedRequest(BaseModel):
    """Request body for embedding generation"""
    texts: list[str]


class EmbedResponse(BaseModel):
    """Response body for embedding generation"""
    embeddings: list[list[float]]
    model: str
    dimension: int


@app.get("/")
async def root():
    """API information and usage instructions"""
    return {
        "service": "Docling PDF to Markdown Converter",
        "version": "1.1.0",
        "endpoints": {
            "POST /convert": "Convert PDF to Markdown",
            "POST /embed": "Generate embeddings for text chunks",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        },
        "configuration": {
            "backend": PDF_BACKEND,
            "ocr_enabled": DO_OCR,
            "table_structure": DO_TABLE_STRUCTURE,
            "image_scale": IMAGES_SCALE,
            "embed_model": EMBED_MODEL_NAME
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "docling-converter"}


@app.post("/convert")
async def convert_pdf(
    file: UploadFile = File(..., description="PDF file to convert")
):
    """
    Convert a PDF file to Markdown format

    Args:
        file: PDF file uploaded as multipart/form-data

    Returns:
        JSON response with markdown content and metadata

    Example:
        curl -X POST "http://localhost:8080/convert" \\
             -F "file=@paper.pdf"
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF (*.pdf)"
        )

    try:
        # Read PDF content
        pdf_content = await file.read()

        # Create DocumentStream from bytes
        buf = BytesIO(pdf_content)
        source = DocumentStream(name=file.filename, stream=buf)

        # Convert PDF
        result = converter.convert(source)

        # Export to markdown (for debugging)
        markdown_content = result.document.export_to_markdown()

        # Export to Docling's JSON format for LlamaIndex
        # This provides lossless representation of document structure
        docling_json = result.document.export_to_dict()

        # Get metadata
        num_pages = len(result.document.pages) if hasattr(result.document, 'pages') else None

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "markdown": markdown_content,
                "docling_json": docling_json,  # Docling's JSON format for LlamaIndex
                "metadata": {
                    "num_pages": num_pages,
                    "size_bytes": len(pdf_content),
                    "backend": PDF_BACKEND
                }
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error converting PDF: {str(e)}"
        )


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """
    Generate embeddings for a list of text chunks

    Args:
        request: JSON body with 'texts' field containing list of strings

    Returns:
        JSON response with embeddings and metadata

    Example:
        curl -X POST "http://localhost:8080/embed" \\
             -H "Content-Type: application/json" \\
             -d '{"texts": ["Hello world", "Another text"]}'
    """
    if not request.texts:
        raise HTTPException(
            status_code=400,
            detail="texts field cannot be empty"
        )

    try:
        model = get_embed_model()
        embeddings = model.get_text_embedding_batch(request.texts)

        return EmbedResponse(
            embeddings=embeddings,
            model=EMBED_MODEL_NAME,
            dimension=len(embeddings[0]) if embeddings else 0
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
