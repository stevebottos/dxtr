"""
Deep Research Agent

Performs comprehensive analysis of research papers using agentic RAG:
1. Generates exploration questions based on user context and abstract
2. Retrieves relevant chunks for each question (deduplicated)
3. Answers user's query using the retrieved chunks

The exploration questions are a retrieval strategy to find relevant sections
beyond what a single query would return. They are NOT answered themselves.
"""

import json
import re
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage

from dxtr.config import config


class DeepResearchAgent:
    """Agent for in-depth research paper analysis using RAG."""

    def __init__(self):
        """Initialize deep research agent."""
        model_config = config.get_model_config("deep_research")
        self.model_name = model_config.name
        self.temperature = model_config.temperature
        self.prompts_dir = Path(__file__).parent / "prompts"

    def analyze_paper(
        self, paper_id: str, user_query: str, user_context: str, date: str = None
    ) -> str:
        """
        Answer a question about a research paper using agentic RAG.

        Workflow:
        1. Generate exploration questions (retrieval strategy)
        2. Retrieve chunks for all questions (deduplicated)
        3. Answer user's query directly with retrieved chunks

        Args:
            paper_id: Paper ID (e.g., "2512.12345")
            user_query: The user's original question/request about the paper
            user_context: User profile and interests
            date: Date in YYYY-MM-DD format (optional)

        Returns:
            Answer based on paper content and user context
        """
        print(f"  [Agent] Finding paper {paper_id}...")
        paper_dir = self._find_paper(paper_id, date)
        if not paper_dir:
            return f"Paper {paper_id} not found. Make sure it has been downloaded."

        print(f"  [Agent] Found at {paper_dir}")

        # Load abstract
        abstract = self._load_abstract(paper_dir)
        if not abstract:
            return f"No metadata found for paper {paper_id}."

        # Load index
        index_dir = paper_dir / "paper.index"
        if not index_dir.exists():
            return f"Index not found for paper {paper_id}. Run 'dxtr get-papers' to build it."

        # TODO: Load index created by docker service (needs embed_model for queries)
        print(f"  [Agent] Loading index...")
        raise NotImplementedError(
            "Index loading needs embed_model - refactor to use docker service"
        )
        total_chunks = len(index.docstore.docs)
        print(f"  [Agent] Index loaded ({total_chunks} chunks)")

        # STEP 1: Generate exploration questions
        print(f"\n  [Step 1/3] Generating exploration questions...")
        exploration_questions = self._generate_exploration_questions(
            abstract, user_context, user_query
        )

        if not exploration_questions:
            return "Failed to generate exploration questions."

        # STEP 2: Retrieve chunks for all questions (deduplicated)
        print(f"\n  [Step 2/3] Retrieving relevant sections...")
        retriever = index.as_retriever(similarity_top_k=3)
        all_chunks = self._retrieve_deduplicated(
            retriever, exploration_questions, total_chunks
        )

        # STEP 3: Answer user's query with retrieved chunks
        print(f"\n  [Step 3/3] Answering your question...\n")
        print(f"[Deep Research Agent]: ", end="", flush=True)
        llm = Ollama(
            model=self.model_name, request_timeout=120.0, temperature=self.temperature
        )
        final_answer = self._answer_with_retrieved_chunks(
            user_query, exploration_questions, all_chunks, user_context, llm
        )

        print("\n")
        return final_answer

    def _load_abstract(self, paper_dir: Path) -> str | None:
        """Load paper abstract from metadata."""
        metadata_file = paper_dir / "metadata.json"
        if not metadata_file.exists():
            return None

        metadata = json.loads(metadata_file.read_text())
        return metadata.get("summary", "")

    def _generate_exploration_questions(
        self, abstract: str, user_context: str, user_query: str
    ) -> list[str]:
        """
        Generate 3-5 exploration questions using the exploration prompt.

        Args:
            abstract: Paper abstract
            user_context: User profile and interests
            user_query: User's original query

        Returns:
            List of exploration questions (3-5 questions)
        """
        # Load exploration prompt
        exploration_prompt = (self.prompts_dir / "exploration.md").read_text()

        # Build user message
        user_message = f"""# User's Background & Interests

{user_context}

---

# Paper Abstract

{abstract}

---

# User's Query

"{user_query}"

---

Generate 3-5 targeted exploration questions for this paper."""

        messages = [
            {"role": "system", "content": exploration_prompt},
            {"role": "user", "content": user_message},
        ]

        # Call LLM (non-streaming for parsing)
        response = chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": self.temperature},
        )

        # Parse questions from response
        questions_text = response.message.content
        questions = self._parse_questions(questions_text)

        # Display questions
        print(f"\n    Generated {len(questions)} exploration questions:")
        for i, q in enumerate(questions, 1):
            print(f"    {i}. {q}")

        return questions

    def _parse_questions(self, text: str) -> list[str]:
        """Parse numbered questions from LLM response."""
        questions = []
        # Match lines starting with "1.", "2.", etc.
        for line in text.strip().split("\n"):
            match = re.match(r"^\s*\d+\.\s*(.+)$", line)
            if match:
                questions.append(match.group(1).strip())
        return questions

    def _retrieve_deduplicated(
        self, retriever, questions: list[str], total_chunks: int
    ) -> list:
        """
        Retrieve chunks for all questions and deduplicate.

        Args:
            retriever: LlamaIndex retriever
            questions: List of exploration questions
            total_chunks: Total chunks in index

        Returns:
            Deduplicated list of chunks
        """
        all_chunks = []
        chunk_ids_seen = set()

        for i, question in enumerate(questions, 1):
            nodes = retriever.retrieve(question)
            new_chunks = 0
            for node in nodes:
                node_id = node.node_id
                if node_id not in chunk_ids_seen:
                    all_chunks.append(node)
                    chunk_ids_seen.add(node_id)
                    new_chunks += 1

            print(f"    Q{i}: Retrieved {len(nodes)} chunks ({new_chunks} new)")

        print(f"\n    Total unique chunks: {len(all_chunks)}/{total_chunks}")
        return all_chunks

    def _answer_with_retrieved_chunks(
        self,
        user_query: str,
        exploration_questions: list[str],
        chunks: list,
        user_context: str,
        llm,
    ) -> str:
        """
        Answer user's query using chunks retrieved via exploration questions.

        Args:
            user_query: User's original question
            exploration_questions: Questions used for retrieval (for context)
            chunks: Retrieved chunks
            user_context: User profile
            llm: LLM instance

        Returns:
            Answer string
        """
        # Build context from all chunks
        chunk_texts = [node.text for node in chunks]
        context_str = "\n\n---\n\n".join(chunk_texts)

        # Build list of exploration questions
        questions_list = "\n".join(
            [f"{i}. {q}" for i, q in enumerate(exploration_questions, 1)]
        )

        # Build prompt
        prompt = f"""{user_context}

-----

You are analyzing a research paper with the user's background in mind.

To find the most relevant sections of this paper, I generated these exploration questions:

{questions_list}

These questions helped retrieve the following relevant sections from the paper:

{context_str}

-----

User's Question: {user_query}

-----

Answer the user's question using the retrieved paper sections above. The exploration questions were just a retrieval strategy - focus on answering the user's actual question. Be specific, cite details from the paper, and tailor your response to the user's background and interests."""

        # Stream answer
        response = llm.stream_complete(prompt)
        answer_text = ""
        for chunk in response:
            delta = chunk.delta
            answer_text += delta
            print(delta, end="", flush=True)

        return answer_text

    def _find_paper(self, paper_id: str, date: str = None) -> Path | None:
        """
        Find paper directory by ID.

        Args:
            paper_id: Paper ID to search for
            date: Optional date to narrow search

        Returns:
            Path to paper directory or None if not found
        """
        papers_root = config.paths.papers_dir

        if date:
            # Search specific date
            date_dir = papers_root / date
            if date_dir.exists():
                paper_dir = date_dir / paper_id
                if paper_dir.exists():
                    return paper_dir
        else:
            # Search all dates
            for date_dir in papers_root.iterdir():
                if date_dir.is_dir():
                    paper_dir = date_dir / paper_id
                    if paper_dir.exists():
                        return paper_dir

        return None


# Global instance for backward compatibility
_agent = DeepResearchAgent()


def analyze_paper(
    paper_id: str, user_query: str, user_context: str, date: str = None
) -> str:
    """
    Answer a question about a research paper using RAG.

    This is a convenience function that delegates to the agent instance.

    Args:
        paper_id: Paper ID (e.g., "2512.12345")
        user_query: The user's original question/request about the paper
        user_context: User profile and interests
        date: Date in YYYY-MM-DD format (optional)

    Returns:
        Answer based on paper content and user context
    """
    return _agent.analyze_paper(paper_id, user_query, user_context, date)
