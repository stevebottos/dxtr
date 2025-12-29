"""
Deep Research tools for the main chat agent to delegate to deep_research agent
"""

from ..agent import analyze_paper as _analyze_paper


def deep_research(paper_id: str, question: str, date: str = None) -> dict:
    """
    Answer a specific question about a research paper using RAG.

    Uses retrieval-augmented generation to find relevant sections of the paper
    and provide a detailed, context-aware answer tailored to the user's background.

    Args:
        paper_id: Paper ID (e.g., "2512.12345" or just "12345")
        question: The question to answer about the paper
        date: Date in YYYY-MM-DD format (optional)

    Returns:
        dict with keys:
            - success: bool
            - paper_id: str (the paper analyzed)
            - answer: str (the answer to the question)
            - error: str (if failed)
    """
    try:
        # Normalize paper ID (remove arxiv prefix if present)
        if "/" in paper_id:
            paper_id = paper_id.split("/")[-1]

        # Load user context from CLI
        from ....cli import _load_user_context
        user_context = _load_user_context()

        # Call deep research agent
        answer = _analyze_paper(paper_id, question, user_context, date)

        return {
            "success": True,
            "paper_id": paper_id,
            "answer": answer
        }

    except Exception as e:
        return {
            "success": False,
            "paper_id": paper_id or "unknown",
            "error": str(e)
        }


# Tool definition for Ollama function calling
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "deep_research",
        "description": "Answer a specific question about a research paper using retrieval-augmented generation. Use this when the user asks detailed questions about a paper like: 'What is the methodology?', 'Summarize paper X', 'What are the main contributions?', 'Suggest a project based on paper Y'. Retrieves relevant sections and provides tailored answers based on the user's profile.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper ID (e.g., '2512.12345' or 'arxiv:2512.12345')"
                },
                "question": {
                    "type": "string",
                    "description": "The question to answer about the paper (e.g., 'What is the main contribution?', 'Summarize this paper', 'Suggest a 1-month project based on this work')"
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, will search all dates if not provided)"
                }
            },
            "required": ["paper_id", "question"]
        }
    }
}
