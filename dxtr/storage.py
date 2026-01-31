"""
Session storage abstraction for dxtr.

Provides a clean interface for storing:
- Message history
- Artifacts (rankings, summaries, etc.)
- Session state

Designed to swap from in-memory to Redis without touching business logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage


# =============================================================================
# ARTIFACT TYPES
# =============================================================================

ArtifactType = Literal["rankings", "github_summary", "profile"]


class ArtifactMeta(BaseModel):
    """Metadata stored in session state for system prompt injection."""
    summary: str  # Human-readable: "rankings from 2025-01-30 based on user profile"
    artifact_type: ArtifactType
    created_at: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Artifact:
    """Full artifact with content (stored separately from session state)."""
    id: int
    content: str
    meta: ArtifactMeta


# =============================================================================
# SESSION STATE (extended with artifact registry)
# =============================================================================

class SessionState(BaseModel):
    """User state loaded at the start of each turn.

    The artifact_registry maps integer choice keys to metadata.
    This gets injected into the system prompt so the agent knows
    what artifacts are available without loading their full content.
    """
    # Existing fields
    has_synthesized_profile: bool = False
    has_github_summary: bool = False
    profile_content: str | None = None

    # Artifact tracking
    artifact_registry: dict[int, ArtifactMeta] = {}
    next_artifact_id: int = 1

    # Artifacts to display this turn (populated by display_artifact tool)
    pending_display_artifacts: list[int] = []

    def register_artifact(self, summary: str, artifact_type: ArtifactType) -> int:
        """Register a new artifact, return its choice ID."""
        artifact_id = self.next_artifact_id
        self.artifact_registry[artifact_id] = ArtifactMeta(
            summary=summary,
            artifact_type=artifact_type,
        )
        self.next_artifact_id += 1
        return artifact_id

    def queue_for_display(self, artifact_id: int) -> None:
        """Mark an artifact to be displayed in the response."""
        if artifact_id not in self.pending_display_artifacts:
            self.pending_display_artifacts.append(artifact_id)

    def get_artifact_prompt_section(self) -> str:
        """Generate system prompt section listing available artifacts."""
        if not self.artifact_registry:
            return ""

        lines = ["# Available Artifacts"]
        lines.append("Use display_artifact(choice) to show to user, read_artifact(choice) to load for discussion.")
        lines.append("")
        for artifact_id, meta in sorted(self.artifact_registry.items()):
            lines.append(f"{artifact_id}: {meta.summary}")

        return "\n".join(lines)


# =============================================================================
# STORAGE INTERFACE
# =============================================================================

class SessionStore(Protocol):
    """Abstract interface for session storage.

    Implementations:
    - InMemoryStore: dict-based, for development
    - RedisStore: Redis-based, for production (future)
    """

    async def get_history(self, session_key: str) -> list[ModelMessage]:
        """Get message history for a session."""
        ...

    async def save_history(self, session_key: str, messages: list[ModelMessage]) -> None:
        """Save message history for a session."""
        ...

    async def get_state(self, session_key: str) -> SessionState:
        """Get session state (includes artifact registry)."""
        ...

    async def save_state(self, session_key: str, state: SessionState) -> None:
        """Save session state."""
        ...

    async def get_artifact(self, session_key: str, artifact_id: int) -> Artifact | None:
        """Get artifact content by ID."""
        ...

    async def save_artifact(self, session_key: str, artifact: Artifact) -> None:
        """Save artifact content."""
        ...


# =============================================================================
# IN-MEMORY IMPLEMENTATION
# =============================================================================

class InMemoryStore:
    """In-memory session store for development."""

    def __init__(self):
        self._history: dict[str, list[ModelMessage]] = {}
        self._state: dict[str, SessionState] = {}
        self._artifacts: dict[str, dict[int, Artifact]] = {}  # session_key -> {id -> Artifact}

    async def get_history(self, session_key: str) -> list[ModelMessage]:
        return self._history.get(session_key, [])

    async def save_history(self, session_key: str, messages: list[ModelMessage]) -> None:
        self._history[session_key] = messages

    async def get_state(self, session_key: str) -> SessionState:
        return self._state.get(session_key, SessionState())

    async def save_state(self, session_key: str, state: SessionState) -> None:
        self._state[session_key] = state

    async def get_artifact(self, session_key: str, artifact_id: int) -> Artifact | None:
        session_artifacts = self._artifacts.get(session_key, {})
        return session_artifacts.get(artifact_id)

    async def save_artifact(self, session_key: str, artifact: Artifact) -> None:
        if session_key not in self._artifacts:
            self._artifacts[session_key] = {}
        self._artifacts[session_key][artifact.id] = artifact


# =============================================================================
# GLOBAL STORE INSTANCE
# =============================================================================

# Single instance used throughout the app
# Swap this to RedisStore for production
_store: SessionStore = InMemoryStore()


def get_store() -> SessionStore:
    """Get the global session store."""
    return _store


def set_store(store: SessionStore) -> None:
    """Set the global session store (for testing or switching to Redis)."""
    global _store
    _store = store


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_session_key(user_id: str, session_id: str) -> str:
    """Generate session key from user and session IDs."""
    return f"{user_id}:{session_id}"


async def create_and_store_artifact(
    session_key: str,
    content: str,
    summary: str,
    artifact_type: ArtifactType,
) -> int:
    """
    Create an artifact, register it in session state, and store its content.

    Returns the artifact ID.
    """
    store = get_store()

    # Get current state and register artifact
    state = await store.get_state(session_key)
    artifact_id = state.register_artifact(summary, artifact_type)
    await store.save_state(session_key, state)

    # Store the full artifact content
    artifact = Artifact(
        id=artifact_id,
        content=content,
        meta=state.artifact_registry[artifact_id],
    )
    await store.save_artifact(session_key, artifact)

    return artifact_id
