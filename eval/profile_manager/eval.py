#!/usr/bin/env python3
"""
Profile Manager Agent Evaluation

Runs profile_manager agent using profile.md, saves outputs, and verifies quality.

Usage: python eval/profile_manager/eval.py
"""

import sys
import shutil
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dxtr.agents.profile_manager.agent import ProfileManagerAgent


def run_eval():
    """Run profile manager evaluation with verification."""

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

    agent = ProfileManagerAgent()

    # === GENERATION PHASE ===
    print("=" * 60)
    print("PHASE 1: GENERATION")
    print("=" * 60)

    enriched_profile = agent.create_profile(profile_path, output_dir=eval_dir)

    # Save profile output
    output_file = eval_dir / "dxtr_profile.md"
    output_file.write_text(enriched_profile)

    print(f"\nProfile saved: {output_file}")
    print(f"GitHub summary saved: {eval_dir / 'github_summary.json'}")

    # === VERIFICATION PHASE ===
    print("\n" + "=" * 60)
    print("PHASE 2: VERIFICATION")
    print("=" * 60)

    # Load the generated github summary
    summary_file = eval_dir / "github_summary.json"
    if not summary_file.exists():
        print("No github_summary.json found, skipping verification")
        return

    github_summary = json.loads(summary_file.read_text())
    print(f"\nVerifying {len(github_summary)} file analyses...")

    # Load verification prompt from eval directory
    verification_prompt = Path(__file__).parent / "prompt.md"
    verification_results = agent.verify_github_summary(github_summary, verification_prompt)

    # Save verification results
    verification_file = eval_dir / "verification_results.json"
    verification_file.write_text(json.dumps(verification_results, indent=2))

    # === METRICS REPORT ===
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    metrics = verification_results["metrics"]

    print(f"\nFiles analyzed: {metrics['total_files']}")
    print(f"\nKeywords Score: {metrics['avg_keywords_score']:.2f}/5.0")
    print(f"  Distribution: {metrics['keywords_score_distribution']}")
    print(f"\nSummary Score:  {metrics['avg_summary_score']:.2f}/5.0")
    print(f"  Distribution: {metrics['summary_score_distribution']}")

    # Overall quality score
    overall = (metrics["avg_keywords_score"] + metrics["avg_summary_score"]) / 2
    print(f"\nOverall Quality: {overall:.2f}/5.0")

    # Show files with issues (score < 4)
    issues = []
    for file_path, result in verification_results["file_results"].items():
        if isinstance(result, dict) and "error" not in result:
            if result.get("keywords_score", 5) < 4 or result.get("summary_score", 5) < 4:
                issues.append((file_path, result))

    if issues:
        print(f"\n{'=' * 60}")
        print(f"FILES WITH ISSUES ({len(issues)} files)")
        print("=" * 60)
        for file_path, result in issues[:5]:  # Show first 5
            short_path = "/".join(Path(file_path).parts[-3:])
            print(f"\n{short_path}")
            print(f"  Keywords: {result.get('keywords_score', '?')}/5 - {result.get('keywords_issues', [])}")
            print(f"  Summary:  {result.get('summary_score', '?')}/5 - {result.get('summary_issues', [])}")

    print(f"\nFull results saved: {verification_file}")


if __name__ == "__main__":
    run_eval()
