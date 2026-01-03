#!/usr/bin/env python3
"""
Simple SGLang server launcher.

Uses Qwen2.5-7B-Instruct with AWQ 4-bit quantization (~4-5GB VRAM).
"""

import subprocess
import sys

# Official Qwen Coder AWQ model (4-bit, ~4-5GB VRAM)
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
# MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

PORT = "30000"

print(f"Starting SGLang with {MODEL} (4-bit AWQ) on port {PORT}...")
print("Press Ctrl+C to stop\n")

cmd = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--model-path",
    MODEL,
    "--port",
    PORT,
    "--quantization",
    "awq_marlin",
    "--tool-call-parser",
    "qwen",  # Enable Qwen tool calling parser
    "--max-total-tokens",
    "32768",  # 32K context window for paper analysis
]

# vram_optimization_args = [
#     "--mem-fraction-static",
#     "0.7",  # Leaves 30% room for memory spikes and OS
#     "--max-running-requests",
#     "2",  # ONLY process 2 files at a time to save VRAM
#     "--chunked-prefill-size",
#     "1024",  # Process long files in small chunks to avoid OOM
#     "--schedule-policy",
#     "lpm",  # Longest Prompt Match: prioritizes files that share prefixes
#     # "--enable-prefix-caching",  # Essential: Keeps your system prompt 'hot' in VRAM
# ]
#
# cmd.extend(vram_optimization_args)
try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\nServer stopped")
except FileNotFoundError:
    print("Error: SGLang not installed. Run: pip install 'sglang[all]'")
