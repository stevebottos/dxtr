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


def load_ground_truth(date: str = "2026-01-01") -> dict:
    """Load ground truth rankings for a specific date."""
    if date == "2026-01-01":
        gt_path = Path(__file__).parent / "ground_truth_jan01.json"
    else:
        gt_path = Path(__file__).parent / "ground_truth.json"
    return json.loads(gt_path.read_text())


def compute_metrics(agent_scores: list[dict], ground_truth: dict) -> dict:
    """Compare agent rankings against ground truth."""
    gt_rankings = {r["id"]: r for r in ground_truth["rankings"]}

    # Extract agent scores (using final_score from agentic flow)
    agent_rankings = {}
    for ps in agent_scores:
        agent_rankings[ps["id"]] = {
            "score": ps.get("final_score"),
            "title": ps["title"],
        }

    metrics = {
        "score_comparison": [],
        "score_diffs": [],
        "rank_order_gt": [],
        "rank_order_agent": [],
        "in_acceptable_range": 0,
    }

    # Compare scores for each paper
    for paper_id, gt in gt_rankings.items():
        agent = agent_rankings.get(paper_id, {})
        gt_score = gt["score"]
        agent_score = agent.get("score")
        acceptable_range = gt.get("acceptable_range", [gt_score, gt_score])

        # Check if agent score is in acceptable range
        in_range = False
        if agent_score:
            in_range = acceptable_range[0] <= agent_score <= acceptable_range[1]
            if in_range:
                metrics["in_acceptable_range"] += 1

        metrics["score_comparison"].append(
            {
                "id": paper_id,
                "title": gt["title"][:50],
                "gt_score": gt_score,
                "agent_score": agent_score,
                "diff": abs(gt_score - agent_score) if agent_score else None,
                "in_range": in_range,
                "key_match": gt.get("key_match", ""),
            }
        )

        if agent_score:
            metrics["score_diffs"].append(abs(gt_score - agent_score))

    # Compute rank orders (sorted by score descending)
    gt_sorted = sorted(gt_rankings.items(), key=lambda x: x[1]["score"], reverse=True)
    agent_sorted = sorted(
        [(k, v) for k, v in agent_rankings.items() if v.get("score")],
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    metrics["rank_order_gt"] = [p[0] for p in gt_sorted]
    metrics["rank_order_agent"] = [p[0] for p in agent_sorted]

    # Summary stats
    if metrics["score_diffs"]:
        metrics["mean_absolute_error"] = sum(metrics["score_diffs"]) / len(
            metrics["score_diffs"]
        )
        metrics["exact_matches"] = sum(1 for d in metrics["score_diffs"] if d == 0)
        metrics["within_one"] = sum(1 for d in metrics["score_diffs"] if d <= 1)

    # Top-5 overlap (for 20 papers)
    gt_top5 = set(metrics["rank_order_gt"][:5])
    agent_top5 = set(metrics["rank_order_agent"][:5])
    metrics["top5_overlap"] = len(gt_top5 & agent_top5)

    return metrics


def load_user_context(fixtures_dir: Path, include_github: bool = True) -> str:
    """
    Load user context from fixtures (profile + optionally github summary).

    Replicates the logic from cli.py:_load_user_context()
    """
    context_parts = []

    # Load profile
    profile_path = fixtures_dir / "profile.md"
    if profile_path.exists():
        profile_content = profile_path.read_text()
        context_parts.append("# User Profile\n\n" + profile_content)

    # Load and parse GitHub summary (optional - adds ~6KB of context)
    if not include_github:
        return "\n".join(context_parts)

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
                            rel_file = "/".join(path_parts[idx + 3 :])

                            # Build keyword index
                            for keyword in keywords:
                                keyword_to_files[keyword.lower()].append(
                                    f"{repo}/{rel_file}"
                                )

                            # Store file summaries by repo
                            if summary:
                                repo_summaries[repo].append(
                                    {
                                        "file": rel_file,
                                        "summary": summary,
                                        "keywords": keywords,
                                    }
                                )

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
                        context_parts.append(
                            f"- **{f['file']}**: {f['summary'][:200]}..."
                        )

        except json.JSONDecodeError:
            pass

    return "\n".join(context_parts)


def run_eval():
    """Run papers ranking evaluation."""

    # Setup paths
    eval_dir = project_root / ".dxtr_eval"
    fixtures_dir = Path(__file__).parent / "fixtures"
    papers_dir = fixtures_dir / "2026-01-01"

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
    print(f'  Query: "{user_query}"')

    agent = papers_ranking.Agent()

    result = agent.run(
        date="2026-01-01",
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
    print("AGENT OUTPUT")
    print("=" * 60)

    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    print(f"\nPapers evaluated: {result.get('paper_count', 'unknown')}")

    # Show individual scores
    print("\n" + "-" * 60)
    print("INDIVIDUAL SCORES:")
    print("-" * 60)
    for ps in result.get("individual_scores", []):
        boosted = " [BOOSTED]" if ps.get("boosted") else ""
        adj = " [ADJ]" if ps.get("critic_adjusted") else ""
        reason = ps.get("reason", "")[:80]
        print(
            f"{ps['id']}: {ps.get('final_score')}/5{boosted}{adj} - {ps['title'][:50]}"
        )
        if reason:
            print(f"  → {reason}")

    # Show final ranking (reduce phase)
    print("\n" + "-" * 60)
    print("FINAL RANKING (Reduce Phase):")
    print("-" * 60)
    print(result.get("final_ranking", "No ranking generated"))

    # === EVALUATION METRICS ===
    gt_path = Path(__file__).parent / "ground_truth_jan01.json"
    if gt_path.exists():
        print("\n" + "=" * 60)
        print("EVALUATION METRICS (vs Ground Truth)")
        print("=" * 60)

        ground_truth = load_ground_truth("2026-01-01")
        metrics = compute_metrics(result.get("individual_scores", []), ground_truth)

        # Score comparison table
        print("\n### Score Comparison:")
        print(
            f"{'Paper ID':<15} {'GT':>3} {'Agent':>6} {'Diff':>5} {'OK':>3}  Key Match"
        )
        print("-" * 70)
        for comp in metrics["score_comparison"]:
            agent_str = str(comp["agent_score"]) if comp["agent_score"] else "N/A"
            diff_str = str(comp["diff"]) if comp["diff"] is not None else "N/A"
            ok_str = "✓" if comp.get("in_range") else "✗"
            print(
                f"{comp['id']:<15} {comp['gt_score']:>3} {agent_str:>6} {diff_str:>5} {ok_str:>3}  {comp['key_match']}"
            )

        # Summary stats
        total = len(metrics["score_comparison"])
        print("\n### Summary:")
        print(
            f"  Mean Absolute Error: {metrics.get('mean_absolute_error', 'N/A'):.2f}"
            if metrics.get("mean_absolute_error")
            else "  Mean Absolute Error: N/A"
        )
        print(f"  In acceptable range: {metrics.get('in_acceptable_range', 0)}/{total}")
        print(f"  Exact score matches: {metrics.get('exact_matches', 0)}/{total}")
        print(f"  Within 1 point: {metrics.get('within_one', 0)}/{total}")
        print(f"  Top-5 overlap: {metrics.get('top5_overlap', 0)}/5")

        # Rank order comparison
        print("\n### Rank Order:")
        print(f"  Ground Truth: {metrics['rank_order_gt']}")
        print(f"  Agent:        {metrics['rank_order_agent']}")

        # Save metrics
        metrics_file = eval_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))
        print(f"\n  Metrics saved to: {metrics_file}")
    else:
        print("\n(No ground truth file - skipping metrics comparison)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_eval()
