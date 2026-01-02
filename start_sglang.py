#!/usr/bin/env python3
"""
Simple SGLang server launcher.

Uses Mistral-7B-Instruct with AWQ 4-bit quantization (~4-5GB VRAM).
Has proper OpenAI-compatible tool calling.
"""

import subprocess
import sys

# Using AWQ quantized Mistral-7B (4-bit, ~4-5GB VRAM, proper function calling)
MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
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
    "awq",
]

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\nServer stopped")
except FileNotFoundError:
    print("Error: SGLang not installed. Run: pip install 'sglang[all]'")
