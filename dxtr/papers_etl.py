"""
Papers ETL Pipeline

End-to-end pipeline for retrieving and processing research papers:
1. Start Docling conversion service
2. Download papers from HuggingFace
3. Convert PDFs to markdown
4. Stop Docling service

Usage:
    from dxtr.papers_etl import PapersETL

    etl = PapersETL()
    etl.run()  # Process today's papers
    etl.run(date="2025-12-26")  # Process specific date
"""

import time
import requests
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from io import BytesIO

import docker
from docker.errors import NotFound, APIError

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser

from dxtr.config import config
from dxtr.util import get_daily_papers


class DoclingService:
    """Manages the Docling Docker container lifecycle"""

    def __init__(
        self,
        image_name: str = "docling-converter",
        container_name: str = "docling-converter",
        port: int = 8080,
        compose_file: Path = None,
    ):
        self.image_name = image_name
        self.container_name = container_name
        self.port = port
        self.compose_file = compose_file or Path("docker-compose.yml")
        self.client = docker.from_env()
        self.base_url = f"http://localhost:{port}"

    def start(self, wait_timeout: int = 60):
        """
        Start the Docling service container

        Args:
            wait_timeout: Maximum seconds to wait for service to be ready

        Returns:
            bool: True if started successfully
        """
        print(f"Starting Docling service...")

        try:
            # Check if container already exists
            try:
                container = self.client.containers.get(self.container_name)
                if container.status == "running":
                    print(f"  Service already running")
                    return True
                else:
                    print(f"  Starting existing container...")
                    container.start()
            except NotFound:
                # Check if image exists before building
                if not self._image_exists():
                    print(f"  Building image...")
                    self._build_image()
                else:
                    print(f"  Using existing image...")

                print(f"  Starting container...")
                self._run_container()

            # Wait for service to be ready
            print(f"  Waiting for service to be ready...", end="", flush=True)
            if self._wait_for_health(timeout=wait_timeout):
                print(" ✓")
                return True
            else:
                print(" ✗ Timeout")
                return False

        except APIError as e:
            print(f"  ✗ Docker API error: {e}")
            return False

    def stop(self):
        """Stop the Docling service container"""
        print(f"Stopping Docling service...")
        try:
            container = self.client.containers.get(self.container_name)
            container.stop(timeout=10)
            print(f"  ✓ Stopped")
        except NotFound:
            print(f"  Container not running")
        except APIError as e:
            print(f"  ✗ Error stopping: {e}")

    def _image_exists(self) -> bool:
        """Check if the Docker image exists"""
        try:
            self.client.images.get(self.image_name)
            return True
        except NotFound:
            return False

    def _build_image(self):
        """Build the Docker image from docker-compose context"""
        # Use docker-compose build context
        build_path = Path("docker/docling")
        self.client.images.build(
            path=str(build_path),
            tag=self.image_name,
            rm=True,
        )

    def _run_container(self):
        """Run the container with proper configuration"""
        self.client.containers.run(
            self.image_name,
            name=self.container_name,
            ports={f"{self.port}/tcp": self.port},
            detach=True,
            remove=False,
            environment={"NVIDIA_VISIBLE_DEVICES": "all"},
            device_requests=[
                docker.types.DeviceRequest(
                    count=-1,  # all GPUs
                    capabilities=[["gpu"]],
                )
            ],
        )

    def _wait_for_health(self, timeout: int = 60) -> bool:
        """Wait for service to respond to health check"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False

    def convert_pdf(self, pdf_path: Path, timeout: int = 300) -> dict:
        """
        Convert a PDF to markdown and LlamaIndex format using the service

        Args:
            pdf_path: Path to PDF file
            timeout: Request timeout in seconds

        Returns:
            Dict with 'markdown' and 'llama_index_data' keys
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/convert",
                files={"file": (pdf_path.name, f, "application/pdf")},
                timeout=timeout,
            )

        response.raise_for_status()
        result = response.json()
        return {
            "markdown": result["markdown"],
            "docling_json": result.get("docling_json"),
        }

    def generate_embeddings(self, texts: list[str], timeout: int = 300) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks

        Args:
            texts: List of text strings to embed
            timeout: Request timeout in seconds

        Returns:
            List of embedding vectors
        """
        response = requests.post(
            f"{self.base_url}/embed",
            json={"texts": texts},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]


class PapersETL:
    """End-to-end pipeline for paper retrieval and processing"""

    def __init__(self, output_dir: Path = None):
        """
        Initialize the ETL pipeline

        Args:
            output_dir: Where to save papers (defaults to config.paths.papers_dir)
        """
        self.output_dir = output_dir or config.paths.papers_dir
        self.docling = DoclingService()

    @contextmanager
    def _managed_service(self):
        """Context manager for Docling service lifecycle"""
        try:
            print("=" * 80)
            if not self.docling.start():
                raise RuntimeError("Failed to start Docling service")
            print()
            yield self.docling
        finally:
            print()
            print("=" * 80)
            self.docling.stop()

    def run(self, date: str = None, max_papers: int = None):
        """
        Run the complete ETL pipeline

        Args:
            date: Date string in YYYY-MM-DD format (defaults to today)
            max_papers: Maximum papers to process (None = all)
        """
        # Default to today
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        print(f"Papers ETL Pipeline - {date}")

        # Step 1: Start Docling service
        with self._managed_service() as docling:
            # Step 2: Download papers
            print(f"Downloading papers from HuggingFace...")
            papers = self._download_papers(date, max_papers)

            if not papers:
                print("  No papers found")
                return

            print(f"  ✓ Downloaded {len(papers)} paper(s)\n")

            # Step 3: Convert PDFs to markdown + build indices
            print(f"Processing papers...")
            self._convert_papers(papers, docling)

        # Service automatically stopped via context manager
        print()
        print("Pipeline complete")

    def _download_papers(self, date: str, max_papers: int = None) -> list[Path]:
        """Download papers and return list of PDF paths"""
        papers = get_daily_papers(output_root=self.output_dir, date=date)

        # Collect all paper.pdf files
        date_dir = self.output_dir / date
        if not date_dir.exists():
            return []

        pdf_files = list(date_dir.glob("*/paper.pdf"))

        if max_papers:
            pdf_files = pdf_files[:max_papers]

        return pdf_files

    def _convert_papers(self, pdf_files: list[Path], docling: DoclingService):
        """Convert all PDFs to markdown and LlamaIndex format"""
        successful = 0
        failed = 0
        skipped = 0

        for idx, pdf_path in enumerate(pdf_files, 1):
            paper_id = pdf_path.parent.name
            md_path = pdf_path.parent / "paper.md"
            layout_path = pdf_path.parent / "layout.json"
            index_dir = pdf_path.parent / "paper.index"

            # Skip if already processed
            if md_path.exists() and layout_path.exists() and index_dir.exists():
                print(f"  [{idx}/{len(pdf_files)}] {paper_id}: Already processed")
                skipped += 1
                continue

            print(f"  [{idx}/{len(pdf_files)}] {paper_id}:")

            try:
                # Step 1: Convert PDF to markdown + layout JSON
                print(f"    Converting to markdown...", end=" ", flush=True)
                result = docling.convert_pdf(pdf_path, timeout=600)

                md_path.write_text(result["markdown"], encoding="utf-8")

                import json

                layout_path.write_text(
                    json.dumps(result["docling_json"], indent=2), encoding="utf-8"
                )
                print("✓")

                # Step 2: Build index from markdown (no docling needed on host)
                print(f"    Building index:")
                document = Document(text=result["markdown"])

                print(f"      Parsing nodes...", end=" ", flush=True)
                # Use SentenceSplitter with reasonable chunk size for embeddings
                from llama_index.core.node_parser import SentenceSplitter

                node_parser = SentenceSplitter(
                    chunk_size=1024,  # ~750 tokens, well under 8k limit
                    chunk_overlap=128,
                )
                nodes = node_parser.get_nodes_from_documents([document])
                print(f"{len(nodes)} nodes")

                # Generate embeddings via docker service
                print(f"      Generating embeddings...", end=" ", flush=True)
                texts = [node.get_content() for node in nodes]
                embeddings = docling.generate_embeddings(texts)

                # Assign embeddings to nodes
                for node, embedding in zip(nodes, embeddings):
                    node.embedding = embedding
                print("✓")

                # Build index with pre-computed embeddings (no embed_model needed)
                print(f"      Creating index...", end=" ", flush=True)
                index = VectorStoreIndex(nodes, embed_model=None)
                print("✓")

                print(f"      Persisting...", end=" ", flush=True)
                index.storage_context.persist(persist_dir=str(index_dir))
                print("✓")

                successful += 1

            except requests.exceptions.HTTPError as e:
                try:
                    error_detail = e.response.json().get("detail", str(e))
                    print(f"✗ {error_detail}")
                except:
                    print(f"✗ {e}")
                failed += 1
            except Exception as e:
                print(f"✗ {e}")
                failed += 1

        # Summary
        print()
        print(f"  Results: {successful} successful, {skipped} skipped, {failed} failed")


# Convenience function for CLI
def run_etl(date: str = None, max_papers: int = None):
    """
    Run the papers ETL pipeline

    Args:
        date: Date string in YYYY-MM-DD format (defaults to today)
        max_papers: Maximum papers to process (None = all)
    """
    etl = PapersETL()
    etl.run(date=date, max_papers=max_papers)
