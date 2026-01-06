from datetime import datetime
from pathlib import Path
import requests
import time
import xml.etree.ElementTree as ET
import json

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}"


def extract_pdf(pdf_path, output_path):
    """Extract PDF content to markdown"""
    # TODO: Implement PDF extraction using docling or another high-quality extractor
    pass


def get_daily_papers(output_root, date=None):
    """Fetch daily papers from HF and Arxiv (batched & rate-limited)."""
    date = date or datetime.today().strftime("%Y-%m-%d")
    out_dir = Path(output_root) / str(date)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching papers for {date}...")
    try:
        data = requests.get(f"https://huggingface.co/api/daily_papers?date={date}").json()
    except Exception as e:
        return print(f"HF API Error: {e}")

    if not isinstance(data, list): return print("Invalid HF response")
    
    # Map ID -> Data (handle nested 'paper' key)
    papers = {p.get('paper', {}).get('id', p.get('id')): p for p in data}
    pids = [k for k in papers if k]

    # Process in batches
    for i in range(0, len(pids), 10):
        batch = pids[i:i+10]
        # Query Arxiv Metadata (Batch)
        try:
            requests.get(ARXIV_API_URL, params={"id_list": ",".join(batch), "max_results": len(batch)})
        except Exception as e:
            print(f"Arxiv Batch Error: {e}")

        for pid in batch:
            p_dir = out_dir / pid
            p_dir.mkdir(parents=True, exist_ok=True)
            
            # Download PDF
            if not (p_dir / "paper.pdf").exists():
                try:
                    r = requests.get(ARXIV_PDF_URL.format(id=pid))
                    if r.status_code == 200:
                        (p_dir / "paper.pdf").write_bytes(r.content)
                        print(f"Downloaded: {pid}")
                        extract_pdf(p_dir / "paper.pdf", p_dir / "paper.md")
                        time.sleep(1) # Throttle PDF downloads
                    else:
                        print(f"PDF Failed {pid}: {r.status_code}")
                except Exception as e:
                    print(f"PDF Error {pid}: {e}")

            # Save Metadata
            meta = papers[pid].copy()
            if 'paper' in meta: meta.update(meta.pop('paper'))
            meta['id'] = pid
            (p_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        
        time.sleep(1) # Throttle batches
