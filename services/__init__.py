"""Standalone services."""

from services.update_papers import fetch_papers_for_period, fetch_papers_metadata

__all__ = ["fetch_papers_metadata", "fetch_papers_for_period"]
