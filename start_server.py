#!/usr/bin/env python3
"""
SGLang server launcher for dxtr.

Usage:
    python start_server.py              # Uses default model (qwen-coder-7b)
    python start_server.py qwen-7b      # Use Qwen 7B Instruct
    python start_server.py mistral-nemo # Use Mistral Nemo 12B
    python start_server.py deepseek-r1-8b # Use DeepSeek R1 8B
    python start_server.py --list       # List available models
"""

import argparse
import subprocess
import sys

# Model configurations
# Each model has: path, quantization method, and recommended settings
MODELS = {  # --- NEW: Reasoning & Tool Calling (Jan 2026) ---
    "falcon-h1r-7b": {
        "path": "tiiuae/falcon-h1r-7b-awq",  # TII's Jan 2026 release
        "quantization": "awq_marlin",
        "tool_call_parser": "falcon_agent",  # Optimized for hybrid Mamba-Transformer tool use
        "max_total_tokens": 262144,  # Massive context window support
        "max_running_requests": 12,  # High throughput due to hybrid architecture
        "description": "Falcon H1R 7B (~5GB VRAM) - Jan 2026 SOTA reasoning/agentic hybrid",
    },
    "qwen3-8b": {
        "path": "Qwen/Qwen3-8B-AWQ",
        "quantization": "awq_marlin",
        "tool_call_parser": "qwen",  # Native /think and /no_think switching
        "max_total_tokens": 32768,
        "max_running_requests": 8,
        "description": "Qwen 3 8B Instruct AWQ (~6GB VRAM) - Best tool calling stability",
    },
    "nemotron-nano-8b": {
        "path": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD",
        "trust_remote_code": True,
        "tool_call_parser": "llama3",  # Follows Llama 3.1 tool schema
        "max_total_tokens": 32768,
        "max_running_requests": 8,
        "description": "Nemotron-Nano 8B AWQ (~6GB VRAM) - NVIDIA's reasoning/agent specialist",
    },
    "qwen-coder-7b": {
        "path": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "quantization": "awq_marlin",
        "tool_call_parser": "qwen",
        "max_total_tokens": 32768,
        "max_running_requests": 8,
        "description": "Qwen 2.5 Coder 7B AWQ (~5GB VRAM) - Best for code tasks",
    },
    "qwen-7b": {
        "path": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "quantization": "awq_marlin",
        "tool_call_parser": "qwen",
        "max_total_tokens": 32768,
        "max_running_requests": 8,
        "description": "Qwen 2.5 7B Instruct AWQ (~5GB VRAM) - General purpose",
    },
    "qwen-14b": {
        "path": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "quantization": "awq_marlin",
        "tool_call_parser": "qwen",
        "max_total_tokens": 16384 // 2,
        "max_running_requests": 4,
        "mem_fraction_static": 0.85,
        "kv_cache_dtype": "fp8_e5m2",
        "chunked_prefill_size": 1024,
        "description": "Qwen 2.5 14B Instruct AWQ (~10GB VRAM) - Higher quality",
    },
    "qwen-coder-14b": {
        "path": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        "quantization": "awq_marlin",
        "tool_call_parser": "qwen",
        "max_total_tokens": 16384,
        "max_running_requests": 4,
        "mem_fraction_static": 0.85,
        "kv_cache_dtype": "fp8_e5m2",
        "chunked_prefill_size": 2048,
        "description": "Qwen 2.5 Coder 14B AWQ (~10GB VRAM) - Best code quality",
    },
    "mistral-nemo": {
        "path": "casperhansen/mistral-nemo-instruct-2407-awq",
        "quantization": "awq_marlin",
        "max_total_tokens": 8192,
        "max_running_requests": 4,
        "mem_fraction_static": 0.8,
        "kv_cache_dtype": "fp8_e5m2",
        "chunked_prefill_size": 1024,
        "description": "Mistral Nemo 12B AWQ (~10GB VRAM) - Strong reasoning",
    },
    "deepseek-r1-8b": {
        "path": "casperhansen/DeepSeek-R1-Distill-Llama-8B-AWQ",
        "quantization": "awq_marlin",
        "tool_call_parser": "llama3",
        "max_total_tokens": 32768,
        "max_running_requests": 8,
        "description": "DeepSeek R1 Distill Llama 8B AWQ - SOTA Reasoning",
    },
    "deepseek-r1-14b": {
        "path": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
        "quantization": "awq_marlin",
        "tool_call_parser": "qwen",
        "max_total_tokens": 16384 // 2,
        "max_running_requests": 4,
        "mem_fraction_static": 0.85,
        "kv_cache_dtype": "fp8_e5m2",
        "chunked_prefill_size": 1024,
        "description": "DeepSeek R1 Distill Qwen 14B AWQ (~10GB VRAM) - SOTA Reasoning",
    },
    "llama-8b": {
        "path": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "quantization": "awq_marlin",
        "tool_call_parser": "llama3",
        "max_total_tokens": 32768,
        "max_running_requests": 8,
        "description": "Llama 3.1 8B Instruct AWQ (~6GB VRAM) - Meta's flagship",
    },
}

DEFAULT_MODEL = "deepseek-r1-8b"
DEFAULT_PORT = 30000


def build_command(model_key: str, port: int) -> list[str]:
    """Build the sglang launch command for a given model."""
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    config = MODELS[model_key]
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config["path"],
        "--port",
        str(port),
        "--quantization",
        config["quantization"],
    ]

    # Optional arguments based on config
    if "tool_call_parser" in config:
        cmd.extend(["--tool-call-parser", config["tool_call_parser"]])

    if "chat_template" in config:
        cmd.extend(["--chat-template", config["chat_template"]])

    if "reasoning_parser" in config:
        cmd.extend(["--reasoning-parser", config["reasoning_parser"]])

    if config.get("enable_auto_tool_choice"):
        cmd.append("--enable-auto-tool-choice")

    if "max_total_tokens" in config:
        cmd.extend(["--max-total-tokens", str(config["max_total_tokens"])])

    if "max_running_requests" in config:
        cmd.extend(["--max-running-requests", str(config["max_running_requests"])])

    if "mem_fraction_static" in config:
        cmd.extend(["--mem-fraction-static", str(config["mem_fraction_static"])])

    if "kv_cache_dtype" in config:
        cmd.extend(["--kv-cache-dtype", config["kv_cache_dtype"]])

    if "chunked_prefill_size" in config:
        cmd.extend(["--chunked-prefill-size", str(config["chunked_prefill_size"])])

    if "context_length" in config:
        cmd.extend(["--context-length", str(config["context_length"])])

    return cmd


def list_models():
    """Print available models."""
    print("Available models:\n")
    for key, config in MODELS.items():
        default = " (default)" if key == DEFAULT_MODEL else ""
        print(f"  {key}{default}")
        print(f"    {config['description']}")
        print(f"    Path: {config['path']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Launch SGLang server for dxtr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python start_server.py                  # Default model\n"
        "  python start_server.py qwen-14b         # Use Qwen 14B\n"
        "  python start_server.py --port 8000      # Custom port\n",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run server on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    config = MODELS.get(args.model)
    if not config:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Use --list to see available models")
        sys.exit(1)

    print(f"Starting SGLang server...")
    print(f"  Model: {args.model}")
    print(f"  Path:  {config['path']}")
    print(f"  Port:  {args.port}")
    print(f"\nPress Ctrl+C to stop\n")

    cmd = build_command(args.model, args.port)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except FileNotFoundError:
        print("Error: SGLang not installed. Run: pip install 'sglang[all]'")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Server exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
