#!/usr/bin/env python3
"""
Profile Creation Pipeline Evaluation

Evaluates the quality of: github_summarize -> profile_synthesize

Phases:
1. Generation: Run github_summarize and profile_synthesize agents
2. Verification: Evaluate both github summaries and synthesized profile

Usage: python eval/profile_creation/eval.py
"""

import sys
import shutil
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dxtr.agents import github_summarize, profile_synthesize

# Import eval agent (local to eval folder)
from eval.profile_creation import agent as eval_agent


def run_eval():
    """Run profile creation pipeline evaluation."""

    # Clean and create eval directory
    eval_dir = project_root / ".dxtr_eval"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir()

    # Check for profile.md
    profile_path = project_root / "profile.md"
    if not profile_path.exists():
        print("Error: profile.md not found")
        print("Create profile.md in project root with your info and GitHub URL")
        sys.exit(1)

    original_profile = profile_path.read_text()

    # === PHASE 1: GENERATION ===
    print("=" * 60)
    print("PHASE 1: GENERATION")
    print("=" * 60)

    # Step 1: Summarize GitHub repos
    print("\n[1/2] Summarizing GitHub repos...")
    summarize_agent = github_summarize.Agent()
    github_summary = summarize_agent.run(profile_path, output_dir=eval_dir)

    if not github_summary:
        print("No GitHub repos found or analyzed. Exiting.")
        sys.exit(1)

    print(f"Analyzed {len(github_summary)} files")

    # Step 2: Synthesize profile
    print("\n[2/2] Synthesizing profile...")
    synthesize_agent = profile_synthesize.Agent()
    synthesized_profile = synthesize_agent.run(
        profile_path,
        output_dir=eval_dir,
        additional_context=json.dumps(github_summary, indent=2),
    )

    print(f"\nGeneration complete:")
    print(f"  - GitHub summary: {eval_dir / 'github_summary.json'}")
    print(f"  - Profile: {eval_dir / 'profile.md'}")

    # === PHASE 2: VERIFICATION ===
    print("\n" + "=" * 60)
    print("PHASE 2: VERIFICATION")
    print("=" * 60)

    # Run verification on both outputs
    verifier = eval_agent.Agent()
    verifier.run(
        github_summary=github_summary,
        original_profile=original_profile,
        synthesized_profile=synthesized_profile,
        output_dir=eval_dir,
    )


if __name__ == "__main__":
    run_eval()
