from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages] | None # Conversation history
    rag_query: str | None                          # Query string for RAG
    rag_docs: List[Any] | None                     # List of retrieved docs


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str | None = field(default=None)  # Final report
