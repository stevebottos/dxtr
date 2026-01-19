import pytest
import httpx

BASE_URL = "http://localhost:8000"


@pytest.mark.integration
def test_health():
    """Verify the server is running."""
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_chat_hello():
    """Send 'hello' and verify we get a response."""
    response = httpx.post(
        f"{BASE_URL}/chat",
        json={
            "user_id": "test-user",
            "session_id": "test-session",
            "query": "hello",
        },
        timeout=60.0,  # LLM calls can be slow
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
