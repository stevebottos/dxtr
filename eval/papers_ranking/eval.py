#!/usr/bin/env python3
"""
Papers Ranking Evaluation

Evaluates the papers_ranking agent's ability to rank papers by relevance.

Usage: python eval/papers_ranking/eval.py
"""

import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dxtr.agents import papers_ranking


def load_user_context(fixtures_dir: Path) -> str:
    """
    Load user context from fixtures (profile + github summary).

    Replicates the logic from cli.py:_load_user_context()
    """
    context_parts = []

    # Load profile
    profile_path = fixtures_dir / "profile.md"
    if profile_path.exists():
        profile_content = profile_path.read_text()
        context_parts.append("# User Profile\n\n" + profile_content)

    # Load and parse GitHub summary
    github_summary_path = fixtures_dir / "github_summary.json"
    if github_summary_path.exists():
        try:
            github_summary = json.loads(github_summary_path.read_text())

            # Extract all keywords and create technology index
            keyword_to_files = defaultdict(list)
            repo_summaries = defaultdict(list)

            for file_path, analysis_json in github_summary.items():
                try:
                    analysis = json.loads(analysis_json)
                    keywords = analysis.get("keywords", [])
                    summary = analysis.get("summary", "")

                    # Extract repo name from path
                    path_parts = Path(file_path).parts
                    if "repos" in path_parts:
                        idx = path_parts.index("repos")
                        if idx + 2 < len(path_parts):
                            repo = f"{path_parts[idx + 1]}/{path_parts[idx + 2]}"
                            rel_file = "/".join(path_parts[idx + 3:])

                            # Build keyword index
                            for keyword in keywords:
                                keyword_to_files[keyword.lower()].append(
                                    f"{repo}/{rel_file}"
                                )

                            # Store file summaries by repo
                            if summary:
                                repo_summaries[repo].append({
                                    "file": rel_file,
                                    "summary": summary,
                                    "keywords": keywords,
                                })

                except json.JSONDecodeError:
                    continue

            # Format GitHub analysis context
            if keyword_to_files or repo_summaries:
                context_parts.append("\n\n---\n\n# GitHub Code Analysis")

                # Technology Index
                context_parts.append("\n## Technologies & Libraries Used")
                context_parts.append("\n(Keyword → Files where it appears)\n")

                # Sort keywords by frequency
                sorted_keywords = sorted(
                    keyword_to_files.items(), key=lambda x: len(x[1]), reverse=True
                )[:50]

                for keyword, files in sorted_keywords:
                    file_count = len(set(files))
                    context_parts.append(f"- **{keyword}**: {file_count} file(s)")

                # Repository Summaries
                context_parts.append("\n\n## Implementation Details by Repository\n")

                for repo, files in sorted(repo_summaries.items()):
                    context_parts.append(f"\n### {repo}\n")
                    for f in files[:5]:  # Limit to 5 files per repo
                        context_parts.append(f"- **{f['file']}**: {f['summary'][:200]}...")

        except json.JSONDecodeError:
            pass

    return "\n".join(context_parts)


def run_eval():
    """Run papers ranking evaluation."""

    # Setup paths
    eval_dir = project_root / ".dxtr_eval"
    fixtures_dir = Path(__file__).parent / "fixtures"
    papers_dir = fixtures_dir / "2026-01-02"

    # Clean and create eval directory
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir()

    # Check fixtures exist
    if not papers_dir.exists():
        print(f"Error: Papers fixtures not found at {papers_dir}")
        sys.exit(1)

    # Load user context
    print("=" * 60)
    print("PAPERS RANKING EVALUATION")
    print("=" * 60)

    print("\n[1/4] Loading user context from fixtures...")
    user_context = load_user_context(fixtures_dir)
    print(f"  Context length: {len(user_context)} chars")

    # List available papers
    print("\n[2/4] Available papers:")
    paper_ids = []
    for paper_folder in sorted(papers_dir.iterdir()):
        if paper_folder.is_dir():
            metadata_file = paper_folder / "metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                paper_ids.append(metadata.get("id"))
                print(f"  - {metadata.get('id')}: {metadata.get('title', 'No title')}")
    print(f"\n  Total: {len(paper_ids)} papers")

    # Test query
    user_query = "Rank all papers by relevance to my interests"

    print("\n[3/4] Running papers_ranking agent...")
    print(f"  Query: \"{user_query}\"")

    agent = papers_ranking.Agent()

    result = agent.run(
        date="2026-01-02",
        user_context=user_context,
        user_query=user_query,
        papers_dir=fixtures_dir,  # Parent of date folder
    )

    # Save results
    print("\n[4/4] Saving results...")

    # Save full result
    output_file = eval_dir / "papers_ranking_result.json"
    output_file.write_text(json.dumps(result, indent=2))
    print(f"  Result: {output_file}")

    # Save the user message that was sent to LLM for debugging
    debug_file = eval_dir / "debug_user_message.txt"
    debug_file.write_text(user_context)
    print(f"  Debug context: {debug_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if "error" in result:
        print(f"\nError: {result['error']}")
    else:
        print(f"\nPapers evaluated: {result.get('paper_count', 'unknown')}")

        # Show individual scores (map phase)
        print("\n" + "-" * 60)
        print("INDIVIDUAL ASSESSMENTS (Map Phase):")
        print("-" * 60)
        for ps in result.get("individual_scores", []):
            print(f"\n### {ps['id']} - {ps['title']}")
            print(ps['score'])
            print()

        # Show final ranking (reduce phase)
        print("\n" + "-" * 60)
        print("FINAL RANKING (Reduce Phase):")
        print("-" * 60)
        print(result.get("final_ranking", "No ranking generated"))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_eval()
