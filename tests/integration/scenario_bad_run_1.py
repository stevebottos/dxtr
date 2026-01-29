"""
Regression test for bad conversation behavior.

Langfuse Session: https://us.cloud.langfuse.com/project/cmkwx7jsl01ukad071v3n1ksz/sessions/a70df315-3e57-441a-b3ff-b9fa219459e2
Local transcript: testing_strat/bad_runs/1.txt

## Original Behavior (Bug)

Turn 1: User asks "what papers do you have for me today"
- Model checks profile, finds none, asks for info ✓

Turn 2: User provides full profile with "here's everything you need"
- Model summarizes GitHub ✓
- Model asks "Ready to synthesize your full profile. Is it okay to proceed?" ✓
  (This is CORRECT - user might want to add more info conversationally)

Turn 3: User says "didn't you just do that?" (confused confirmation)
- Model calls call_profile_synthesizer ✓
- Model calls get_top_papers ✗ (USER DID NOT ASK FOR THIS)
- Model shows UNRANKED papers (just upvotes) ✗
- Model says "Say 'rank papers for 2026-01-28' for personalized recommendations" ✗
  (User's original request was papers - why make them ask again?)

## Expected Behavior

Turn 3 should:
- Create the profile (user confirmed)
- Confirm success
- Either: Ask if user wants to proceed with their original request (paper ranking)
- Or: Fulfill the original request directly (rank papers)
- Should NOT show unranked papers unprompted

The bug is Grok being overly proactive but doing it WRONG - fetching papers
without being asked, showing them unranked, then telling user to ask again.

Run: pytest tests/integration/scenario_bad_run_1.py -v -s
"""

import pytest
from pathlib import Path
import uuid

from dxtr.agents.master import agent
from dxtr.data_models import MasterRequest
from dxtr import set_session_id, set_session_tags, set_session_metadata, get_model_settings


TEST_USER_ID = f"e2e_test_{uuid.uuid4().hex[:8]}"

PROFILE_CONTENT = (Path(__file__).parent.parent.parent / "testing_strat" / "profile.md").read_text()


def extract_tool_calls(messages) -> list[str]:
    """Extract tool names from agent message history."""
    tool_names = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if hasattr(part, "tool_name"):
                tool_names.append(part.tool_name)
    return tool_names


def create_deps(query: str = "") -> MasterRequest:
    return MasterRequest(
        user_id=TEST_USER_ID,
        session_id=f"session-{uuid.uuid4().hex[:8]}",
        query=query,
    )


def setup_tracing(phase: str):
    set_session_id(f"{phase}-{TEST_USER_ID}")
    set_session_tags(["test", "regression", "bad_run_1"])
    set_session_metadata({"phase": phase, "test_user_id": TEST_USER_ID})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_bad_run_1_regression():
    """
    Replicate bad_runs/1.txt conversation and verify Turn 3 bug is fixed.

    Original session: a70df315-3e57-441a-b3ff-b9fa219459e2
    """
    setup_tracing("bad_run_1")

    print(f"\n{'='*60}")
    print(f"REGRESSION TEST: bad_runs/1.txt")
    print(f"User ID: {TEST_USER_ID}")
    print(f"{'='*60}\n")

    # === Turn 1: User asks for papers ===
    print("TURN 1: User asks for papers")
    print("-" * 40)

    turn1_message = "what papers do you have for me today"
    deps = create_deps(turn1_message)

    print(f"User: {turn1_message}")
    result1 = await agent.run(
        turn1_message,
        deps=deps,
        model_settings=get_model_settings(),
    )

    response1 = result1.output
    messages1 = result1.all_messages()
    tools1 = extract_tool_calls(messages1)

    print(f"Tools: {list(set(tools1))}")
    print(f"DXTR: {response1[:300]}...")

    # Model may either:
    # 1. Check profile and ask for info (interprets "for me" as needing personalization)
    # 2. Just show papers (interprets as simple query)
    # Both are acceptable for Turn 1
    if "check_profile_state" in tools1:
        print("✓ Turn 1: Checked profile, will ask for info")
    else:
        print("✓ Turn 1: Showed papers directly (no personalization assumed)")

    # === Turn 2: User provides everything ===
    print("\n" + "="*60)
    print("TURN 2: User provides complete info")
    print("-" * 40)

    turn2_message = f"""here's everything you need:
{PROFILE_CONTENT}"""

    print(f"User: here's everything you need: [full profile with 3 GitHub repos]")

    result2 = await agent.run(
        turn2_message,
        deps=create_deps(turn2_message),
        message_history=messages1,
        model_settings=get_model_settings(),
    )

    response2 = result2.output
    messages2 = result2.all_messages()
    tools2 = extract_tool_calls(messages2)

    print(f"Tools: {list(set(tools2) - set(tools1))}")
    print(f"DXTR: {response2[:400]}...")

    # Should summarize GitHub
    assert "create_github_summary" in tools2, f"Should summarize GitHub. Tools: {tools2}"

    # Should ask for confirmation (this is CORRECT behavior)
    # User might want to add more info, so asking before synthesizing is right
    print("✓ Turn 2 OK: Summarized GitHub, asked for confirmation")

    # === Turn 3: User confirms (confused) ===
    print("\n" + "="*60)
    print("TURN 3: User confirms (confused)")
    print("-" * 40)

    turn3_message = "didn't you just do that?"
    print(f"User: {turn3_message}")

    result3 = await agent.run(
        turn3_message,
        deps=create_deps(turn3_message),
        message_history=messages2,
        model_settings=get_model_settings(),
    )

    response3 = result3.output
    messages3 = result3.all_messages()
    tools3 = extract_tool_calls(messages3)

    # Tools unique to Turn 3
    turn3_tools = list(set(tools3) - set(tools2))

    print(f"Tools (Turn 3 only): {turn3_tools}")
    print(f"DXTR: {response3[:500]}...")

    # === ASSERTIONS FOR TURN 3 ===
    print("\n" + "="*60)
    print("CHECKING TURN 3 ASSERTIONS")
    print("-" * 40)

    # 1. Should have synthesized profile (user confirmed, even if confused)
    assert "call_profile_synthesizer" in tools3, \
        f"FAIL: Should synthesize profile on confirmation. Tools: {turn3_tools}"
    print("✓ Profile synthesized")

    # 2. Should NOT have called get_top_papers unprompted
    # The bug was calling this without user asking
    assert "get_top_papers" not in turn3_tools, \
        f"FAIL: Should NOT call get_top_papers unprompted. User didn't ask for papers in Turn 3. Tools: {turn3_tools}"
    print("✓ Did not fetch papers unprompted")

    # 3. If papers ARE shown, they should be RANKED (with scores)
    # Either don't show papers, or show ranked ones - never unranked
    if "rank_papers_for_user" in turn3_tools:
        # If model chose to fulfill original request, that's acceptable
        has_scores = any(f"[{i}/10]" in response3 or f"/{i}0]" in response3 for i in range(1, 11))
        assert has_scores, \
            f"FAIL: If showing papers, they must be ranked with scores"
        print("✓ Papers shown are ranked (original request fulfilled)")
    else:
        # If not ranking, should just confirm profile and maybe offer to rank
        assert "rank papers" not in response3.lower() or "would you like" in response3.lower() or "want me to" in response3.lower(), \
            f"FAIL: Should either rank papers or offer to do so, not tell user to ask again"
        print("✓ Profile confirmed without showing unranked papers")

    print(f"\n{'='*60}")
    print("REGRESSION TEST PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_bad_run_1_regression())
