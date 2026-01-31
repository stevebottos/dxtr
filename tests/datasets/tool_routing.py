"""Test case definitions for tool routing evaluation.

Each test case defines:
- input: User query
- expected_tools: Tools that SHOULD be called
- forbidden_tools: Tools that should NOT be called
- description: What the test validates
- requires_profile: Whether this test needs an existing profile
"""

from dataclasses import dataclass


@dataclass
class ToolRoutingCase:
    """A test case for tool routing evaluation."""
    name: str
    input: str
    expected_tools: list[str]
    forbidden_tools: list[str]
    description: str
    requires_profile: bool = False


# =============================================================================
# Profile Creation Tests
# =============================================================================

PROFILE_CREATION_CASES = [
    ToolRoutingCase(
        name="profile_creation_with_github",
        input="__PROFILE_CONTENT__",  # Replaced at runtime with actual profile content
        expected_tools=["create_github_summary", "call_profile_synthesizer"],
        forbidden_tools=[],
        description="Given full profile info with GitHub links, should summarize GitHub then create profile",
        requires_profile=False,
    ),
]


# =============================================================================
# Paper Query Tests (No Profile Needed)
# =============================================================================

PAPER_QUERY_CASES = [
    ToolRoutingCase(
        name="top_papers_by_upvotes",
        input="Show me the top 5 papers based on upvotes from yesterday.",
        expected_tools=["get_top_papers"],
        forbidden_tools=["call_profile_synthesizer", "rank_daily_papers"],
        description="Upvote-based query should use get_top_papers, not ranking",
        requires_profile=False,
    ),
    ToolRoutingCase(
        name="available_papers",
        input="What papers are available?",
        expected_tools=["get_available_papers"],
        forbidden_tools=["call_profile_synthesizer", "rank_daily_papers"],
        description="Availability query should check what's in database",
        requires_profile=False,
    ),
    ToolRoutingCase(
        name="paper_stats",
        input="How many papers do we have from this week?",
        expected_tools=["get_paper_stats"],
        forbidden_tools=["call_profile_synthesizer"],
        description="Stats query should use aggregate function",
        requires_profile=False,
    ),
]


# =============================================================================
# Profile-Based Ranking Tests (Profile Required)
# =============================================================================

RANKING_CASES = [
    ToolRoutingCase(
        name="rank_papers_yesterday",
        input="Rank papers for me based on my profile from yesterday.",
        # rank_daily_papers is now a result function, recorded as "final_result" by pydantic-ai
        expected_tools=["final_result"],
        forbidden_tools=["call_profile_synthesizer", "create_github_summary"],
        description="With existing profile, should rank without recreating profile",
        requires_profile=True,
    ),
    ToolRoutingCase(
        name="rank_papers_today",
        input="Show me today's papers ranked by my interests.",
        expected_tools=["final_result"],
        forbidden_tools=["call_profile_synthesizer", "create_github_summary"],
        description="Should use existing profile for today's rankings",
        requires_profile=True,
    ),
]


# =============================================================================
# Greeting/Chitchat Tests (No Tools)
# =============================================================================

NO_TOOL_CASES = [
    ToolRoutingCase(
        name="greeting_hello",
        input="Hello!",
        expected_tools=[],
        forbidden_tools=["rank_daily_papers", "call_profile_synthesizer", "create_github_summary"],
        description="Simple greeting should not trigger expensive tools",
        requires_profile=False,
    ),
    ToolRoutingCase(
        name="greeting_hi",
        input="Hi there, how are you?",
        expected_tools=[],
        forbidden_tools=["rank_daily_papers", "call_profile_synthesizer", "create_github_summary"],
        description="Chitchat should not trigger tools",
        requires_profile=False,
    ),
    ToolRoutingCase(
        name="thanks",
        input="Thanks for your help!",
        expected_tools=[],
        forbidden_tools=["rank_daily_papers", "call_profile_synthesizer", "create_github_summary"],
        description="Gratitude should not trigger tools",
        requires_profile=False,
    ),
]


# =============================================================================
# Output Format Tests
# =============================================================================

@dataclass
class OutputFormatCase:
    """A test case for validating output format/content."""
    name: str
    input: str
    description: str
    required_patterns: list[str]  # Regex patterns that must appear in output
    requires_profile: bool = False


RANKING_OUTPUT_CASES = [
    OutputFormatCase(
        name="ranking_includes_links",
        input="Rank today's papers for me based on my profile.",
        description="Ranking output should include HuggingFace and arXiv links for each paper",
        required_patterns=[
            r"https://huggingface\.co/papers/\d+\.\d+",  # HuggingFace paper link
            r"https://arxiv\.org/abs/\d+\.\d+",          # arXiv link
        ],
        requires_profile=True,
    ),
]


# =============================================================================
# All Cases (for parametrized tests)
# =============================================================================

ALL_CASES = PROFILE_CREATION_CASES + PAPER_QUERY_CASES + RANKING_CASES + NO_TOOL_CASES

# Cases that don't need profile (can run independently)
INDEPENDENT_CASES = PAPER_QUERY_CASES + NO_TOOL_CASES

# Cases that need profile (must run after profile creation)
PROFILE_DEPENDENT_CASES = RANKING_CASES
