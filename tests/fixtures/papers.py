"""Test papers for integration tests."""

# Papers with clear relevance signals for testing ranking accuracy

ML_PAPERS = {
    "2501.00001": {
        "title": "Scaling Laws for Neural Language Models",
        "summary": "We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training.",
        "expected_relevance": "high",  # For ML researcher
    },
    "2501.00002": {
        "title": "Attention Is All You Need: A Retrospective",
        "summary": "A comprehensive retrospective on the transformer architecture, analyzing its impact on NLP, computer vision, and beyond. We discuss recent improvements and future directions.",
        "expected_relevance": "high",  # For ML researcher
    },
    "2501.00003": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "summary": "We propose Low-Rank Adaptation (LoRA), which freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.",
        "expected_relevance": "high",  # For ML researcher
    },
}

BIOLOGY_PAPERS = {
    "2501.00010": {
        "title": "Single-Cell RNA Sequencing Reveals Novel Cell Types in Human Liver",
        "summary": "Using single-cell RNA sequencing, we identified 12 previously uncharacterized cell populations in human liver tissue, providing new insights into hepatic function and disease.",
        "expected_relevance": "high",  # For biologist
    },
    "2501.00011": {
        "title": "AlphaFold3: Predicting Protein-Ligand Interactions",
        "summary": "We present AlphaFold3, extending protein structure prediction to accurately model protein-ligand, protein-DNA, and protein-RNA interactions.",
        "expected_relevance": "high",  # For biologist
    },
}

UNRELATED_PAPERS = {
    "2501.00020": {
        "title": "Optimizing Crop Yields Using IoT Sensors and Weather Data",
        "summary": "This paper presents a system for predicting optimal irrigation schedules for wheat crops using soil moisture sensors and local weather forecasts.",
        "expected_relevance": "low",  # For ML researcher
    },
    "2501.00021": {
        "title": "A Survey of Traditional Basket Weaving Techniques in Southeast Asia",
        "summary": "We document and categorize traditional basket weaving patterns from 15 indigenous communities across Thailand, Vietnam, and Cambodia.",
        "expected_relevance": "low",  # For everyone
    },
}

# Combined test set for ranking tests
MIXED_PAPERS_FOR_ML_RESEARCHER = {
    **ML_PAPERS,
    **{k: {**v, "expected_relevance": "low"} for k, v in BIOLOGY_PAPERS.items()},
    **UNRELATED_PAPERS,
}


def papers_to_ranking_format(papers: dict) -> dict:
    """Convert test papers to the format expected by ranking agent."""
    return {
        paper_id: {"title": data["title"], "summary": data["summary"]}
        for paper_id, data in papers.items()
    }


def papers_to_db_format(papers: dict, date: str = "2025-01-27") -> list[dict]:
    """Convert test papers to the format returned by PostgresHelper."""
    return [
        {
            "id": paper_id,
            "title": data["title"],
            "summary": data["summary"],
            "authors": [],
            "published_at": None,
            "upvotes": 50,  # Default upvotes
            "date": date,
        }
        for paper_id, data in papers.items()
    ]
