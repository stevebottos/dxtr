"""
Evaluation utilities for profile creation pipeline.

Contains schemas and helper functions for verification.
"""

import json
from pathlib import Path

# JSON schema for verification of generated summaries
SUMMARY_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "keywords_issues": {"type": "array", "items": {"type": "string"}},
        "summary_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "summary_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords_score", "summary_score"],
}

# JSON schema for verification of synthesized profile
PROFILE_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "accuracy_issues": {"type": "array", "items": {"type": "string"}},
        "completeness_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "completeness_issues": {"type": "array", "items": {"type": "string"}},
        "coherence_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "coherence_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["accuracy_score", "completeness_score", "coherence_score"],
}


def compute_summary_metrics(verification_results: list[dict]) -> dict:
    """Compute aggregate metrics from summary verification results.

    Args:
        verification_results: List of verification result dicts

    Returns:
        Dict with aggregate metrics
    """
    keyword_scores = []
    summary_scores = []

    for result in verification_results:
        if "error" not in result:
            keyword_scores.append(result.get("keywords_score", 0))
            summary_scores.append(result.get("summary_score", 0))

    return {
        "total_files": len(verification_results),
        "avg_keywords_score": sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0,
        "avg_summary_score": sum(summary_scores) / len(summary_scores) if summary_scores else 0,
        "keywords_score_distribution": {
            score: keyword_scores.count(score) for score in range(1, 6)
        },
        "summary_score_distribution": {
            score: summary_scores.count(score) for score in range(1, 6)
        },
    }


def compute_profile_metrics(verification_result: dict) -> dict:
    """Compute metrics from profile verification result.

    Args:
        verification_result: Profile verification result dict

    Returns:
        Dict with profile metrics
    """
    if "error" in verification_result:
        return {"error": verification_result["error"]}

    accuracy = verification_result.get("accuracy_score", 0)
    completeness = verification_result.get("completeness_score", 0)
    coherence = verification_result.get("coherence_score", 0)

    return {
        "accuracy_score": accuracy,
        "completeness_score": completeness,
        "coherence_score": coherence,
        "overall_score": (accuracy + completeness + coherence) / 3,
        "accuracy_issues": verification_result.get("accuracy_issues", []),
        "completeness_issues": verification_result.get("completeness_issues", []),
        "coherence_issues": verification_result.get("coherence_issues", []),
    }


def find_issues(file_results: dict) -> list[tuple[str, dict]]:
    """Find files with issues (score < 4).

    Args:
        file_results: Dict mapping file paths to verification results

    Returns:
        List of (file_path, result) tuples for files with issues
    """
    issues = []
    for file_path, result in file_results.items():
        if isinstance(result, dict) and "error" not in result:
            if (
                result.get("keywords_score", 5) < 4
                or result.get("summary_score", 5) < 4
            ):
                issues.append((file_path, result))
    return issues


def print_summary_metrics(metrics: dict, file_results: dict):
    """Print formatted summary verification metrics.

    Args:
        metrics: Aggregate metrics dict
        file_results: Dict mapping file paths to verification results
    """
    print("\n" + "-" * 40)
    print("GITHUB SUMMARY VERIFICATION")
    print("-" * 40)

    print(f"\nFiles analyzed: {metrics['total_files']}")
    print(f"Keywords Score: {metrics['avg_keywords_score']:.2f}/5.0")
    print(f"  Distribution: {metrics['keywords_score_distribution']}")
    print(f"Summary Score:  {metrics['avg_summary_score']:.2f}/5.0")
    print(f"  Distribution: {metrics['summary_score_distribution']}")

    overall = (metrics["avg_keywords_score"] + metrics["avg_summary_score"]) / 2
    print(f"Overall: {overall:.2f}/5.0")

    issues = find_issues(file_results)
    if issues:
        print(f"\nFiles with issues ({len(issues)}):")
        for file_path, result in issues[:3]:
            short_path = "/".join(Path(file_path).parts[-3:])
            print(f"  {short_path}")
            print(f"    Keywords: {result.get('keywords_score', '?')}/5")
            print(f"    Summary:  {result.get('summary_score', '?')}/5")


def print_profile_metrics(metrics: dict):
    """Print formatted profile verification metrics.

    Args:
        metrics: Profile metrics dict
    """
    print("\n" + "-" * 40)
    print("PROFILE VERIFICATION")
    print("-" * 40)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"\nAccuracy:    {metrics['accuracy_score']}/5.0")
    if metrics.get("accuracy_issues"):
        for issue in metrics["accuracy_issues"][:3]:
            print(f"  - {issue}")

    print(f"Completeness: {metrics['completeness_score']}/5.0")
    if metrics.get("completeness_issues"):
        for issue in metrics["completeness_issues"][:3]:
            print(f"  - {issue}")

    print(f"Coherence:   {metrics['coherence_score']}/5.0")
    if metrics.get("coherence_issues"):
        for issue in metrics["coherence_issues"][:3]:
            print(f"  - {issue}")

    print(f"\nOverall: {metrics['overall_score']:.2f}/5.0")
