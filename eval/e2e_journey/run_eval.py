#!/usr/bin/env python3
"""
End-to-End User Journey Evaluation

Automates the complete DXTR user journey from a fresh start:
1. Initiate chat (dxtr chat)
2. Profile creation (pass profile.md, GitHub summarization, profile synthesis)
3. Paper ranking (check/download/rank papers)
4. Deep research (ask for best paper recommendation from top-5)

All outputs are captured for LLM-as-a-judge evaluation.

Usage: python eval/e2e_journey/run_eval.py
"""

import sys
import os
import re
import shutil
import time
from pathlib import Path
from datetime import datetime

# Try pexpect, fall back to manual instructions
try:
    import pexpect
    HAS_PEXPECT = True
except ImportError:
    HAS_PEXPECT = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Eval output directory
EVAL_DIR = PROJECT_ROOT / ".dxtr_eval"
DEBUG_DIR = EVAL_DIR / "debug"
TRANSCRIPT_FILE = DEBUG_DIR / "full_transcript.txt"
RANKINGS_FILE = DEBUG_DIR / "rankings.txt"
RECOMMENDATION_FILE = DEBUG_DIR / "recommendation.txt"
PROFILE_ARTIFACTS_DIR = DEBUG_DIR / "profile_artifacts"

# User journey script - what we say at each step
USER_JOURNEY = [
    # Step 1: Initial greeting - DXTR will notice no profile
    # We wait for DXTR's greeting and profile prompt

    # Step 2: Provide profile path
    {"say": "./profile.md", "wait_for": "read", "description": "Provide profile path"},

    # Step 3: Confirm reading profile (Missing in original)
    {"say": "yes", "wait_for": "analyze", "description": "Confirm reading profile"},

    # Step 4: Confirm GitHub summarization
    {"say": "yes", "wait_for": "synthesize", "description": "Confirm GitHub summarization"},

    # Step 5: Confirm profile synthesis
    {"say": "yes", "wait_for": "created|updated|ready|help", "description": "Confirm profile synthesis", "timeout": 180},

    # Step 6: Ask for paper ranking
    {"say": "rank today's papers for me", "wait_for": "download|check", "description": "Request paper ranking"},

    # Step 7: Confirm paper download (if needed)
    {"say": "yes", "wait_for": "rank|proceed|different date", "description": "Confirm paper download"},

    # Step 8: Confirm ranking (if asked)
    {"say": "yes", "wait_for": "Ranked", "description": "Confirm ranking", "capture_as": "rankings", "timeout": 180},

    # Step 9: Ask for best paper recommendation
    {"say": "Based on the top 5 papers, which single paper would be the absolute best for a side project given my profile? Explain why.",
     "wait_for": "Analysis complete", "description": "Request best paper recommendation", "capture_as": "recommendation", "timeout": 120},
]


def clean_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def setup_eval_directory():
    """Create fresh eval directory structure."""
    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)

    DEBUG_DIR.mkdir(parents=True)
    PROFILE_ARTIFACTS_DIR.mkdir(parents=True)

    # Create metadata
    metadata = {
        "started_at": datetime.now().isoformat(),
        "profile_source": str(PROJECT_ROOT / "profile.md"),
    }
    (DEBUG_DIR / "metadata.txt").write_text(
        f"E2E Journey Evaluation\n"
        f"Started: {metadata['started_at']}\n"
        f"Profile: {metadata['profile_source']}\n"
    )

    return metadata


def clear_dxtr_state():
    """Clear .dxtr directory to simulate fresh start."""
    dxtr_dir = PROJECT_ROOT / ".dxtr"
    if dxtr_dir.exists():
        # Backup papers if they exist (expensive to re-download)
        papers_dir = dxtr_dir / "hf_papers"
        papers_backup = None
        if papers_dir.exists():
            papers_backup = PROJECT_ROOT / ".dxtr_papers_backup"
            if papers_backup.exists():
                shutil.rmtree(papers_backup)
            shutil.move(str(papers_dir), str(papers_backup))

        # Clear profile and other state
        for item in dxtr_dir.iterdir():
            if item.name != "hf_papers":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # Restore papers backup
        if papers_backup and papers_backup.exists():
            shutil.move(str(papers_backup), str(papers_dir))

    print("[Setup] Cleared .dxtr state (preserved papers)")


class Tee:
    """Helper to write to stdout and a file simultaneously."""
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

def run_with_pexpect():
    """Run the eval using pexpect for automation."""
    print("\n" + "=" * 60)
    print("E2E JOURNEY EVALUATION (AUTOMATED)")
    print("=" * 60 + "\n")

    setup_eval_directory()
    clear_dxtr_state()

    captures = {}

    # Start dxtr chat
    print("[Starting] dxtr chat...")
    
    # Use Tee to capture everything to full_transcript.txt
    logfile = Tee(TRANSCRIPT_FILE)
    
    child = pexpect.spawn(
        "python", ["-m", "dxtr.cli", "chat"],
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
        timeout=300,
    )
    child.logfile_read = logfile

    try:
        # Wait for initial greeting
        print("\n[Waiting] Initial greeting...")
        child.expect(["profile", "Profile", "DXTR"], timeout=60)

        # Execute user journey
        for i, step in enumerate(USER_JOURNEY):
            print(f"\n[Step {i+1}] {step['description']}")

            # Send user input
            user_input = step["say"]
            # We don't need to print [You] here because child.logfile_read captures the echo
            
            child.sendline(user_input)

            # Wait for expected response
            timeout = step.get("timeout", 120)
            wait_pattern = step["wait_for"]

            try:
                child.expect(wait_pattern, timeout=timeout)
                # Capture specific outputs - child.before contains everything since last expect (including tool output)
                if "capture_as" in step:
                    capture_name = step["capture_as"]
                    response = clean_ansi(child.before) 
                    captures[capture_name] = response
                    print(f"[Captured] {capture_name} ({len(response)} chars)")

            except pexpect.TIMEOUT:
                print(f"[TIMEOUT] Waiting for: {wait_pattern}")
                # Continue anyway

            except pexpect.EOF:
                print("[EOF] Process ended unexpectedly")
                break

        # Give time for final output
        time.sleep(2)

    except Exception as e:
        print(f"\n[ERROR] {e}")

    finally:
        child.close()
        logfile.close()

    # Save specific captures
    save_outputs(captures)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {DEBUG_DIR}")
    print(f"  - full_transcript.txt")
    print(f"  - rankings.txt")
    print(f"  - recommendation.txt")
    print(f"\nNext: Run LLM-as-a-judge evaluation")
    print(f"  See: eval/llm_as_a_judge.md")


def run_manual():
    """Provide manual instructions when pexpect is unavailable."""
    print("\n" + "=" * 60)
    print("E2E JOURNEY EVALUATION (MANUAL)")
    print("=" * 60)
    print("\npexpect not installed. Follow these manual steps:\n")

    setup_eval_directory()
    clear_dxtr_state()

    print("1. Run: python -m dxtr.cli chat")
    print("2. When DXTR asks for profile, say: ./profile.md")
    print("3. Say 'yes' to summarize GitHub")
    print("4. Say 'yes' to synthesize profile")
    print("5. Say: rank today's papers for me")
    print("6. Say 'yes' to download papers (if asked)")
    print("7. Say 'yes' to proceed with ranking")
    print(f"8. Copy ranking output to: {RANKINGS_FILE}")
    print("9. Say: Based on the top 5 papers, which single paper would be")
    print("   the absolute best for a side project given my profile?")
    print(f"10. Copy recommendation to: {RECOMMENDATION_FILE}")
    print("\nThen run LLM-as-a-judge evaluation.")
    print("See: eval/llm_as_a_judge.md")


def save_outputs(captures: dict):
    """Save all captured outputs to files."""
    # Full transcript is already saved by Tee

    # Rankings
    if "rankings" in captures:
        RANKINGS_FILE.write_text(captures["rankings"])
        print(f"[Saved] {RANKINGS_FILE}")
    else:
        RANKINGS_FILE.write_text("[No rankings captured - check transcript]")

    # Recommendation
    if "recommendation" in captures:
        RECOMMENDATION_FILE.write_text(captures["recommendation"])
        print(f"[Saved] {RECOMMENDATION_FILE}")
    else:
        RECOMMENDATION_FILE.write_text("[No recommendation captured - check transcript]")

    # Copy profile artifacts if they were created
    dxtr_dir = PROJECT_ROOT / ".dxtr"
    artifacts = [
        "github_summary.json",
        "dxtr_profile.md",
    ]
    for artifact in artifacts:
        src = dxtr_dir / artifact
        if src.exists():
            dst = PROFILE_ARTIFACTS_DIR / artifact
            shutil.copy(src, dst)
            print(f"[Copied] {artifact}")


def main():
    """Main entry point."""
    if HAS_PEXPECT:
        run_with_pexpect()
    else:
        run_manual()
        print("\nTo enable automation: pip install pexpect")


if __name__ == "__main__":
    main()
