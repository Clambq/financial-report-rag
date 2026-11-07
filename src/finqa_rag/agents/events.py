"""Execution event contracts and Server-Sent Events serialization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EventType = Literal[
    "run_started",
    "node_started",
    "tool_started",
    "tool_finished",
    "node_finished",
    "verification_failed",
    "retrying",
    "answer_chunk",
    "run_finished",
    "run_failed",
]


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    request_id: str
    sequence: int = Field(ge=0)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    node: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


def encode_sse_event(event: AgentEvent) -> str:
    payload = json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {payload}\n\n"
    )
