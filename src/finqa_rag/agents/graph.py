"""Bounded LangGraph orchestration for the FinSight agents."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from time import perf_counter_ns
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from finqa_rag.agents.events import AgentEvent
from finqa_rag.agents.nodes import (
    AgentRuntime,
    quant_node,
    report_node,
    research_node,
    supervisor_node,
    verifier_node,
)
from finqa_rag.agents.state import AgentResult, FinSightState, NodeTrace


NodeCallable = Callable[[FinSightState, AgentRuntime], Awaitable[dict[str, object]]]


def _merge_state(
    state: FinSightState,
    update: dict[str, object],
    *,
    node: str,
    started_ns: int,
    ended_ns: int,
) -> dict[str, object]:
    values = dict(update)
    new_issues = values.pop("issues", [])
    values["issues"] = [*state.issues, *new_issues]
    values["traces"] = [
        *state.traces,
        NodeTrace(
            node=node,
            status="completed",
            latency_ms=max(0.0, (ended_ns - started_ns) / 1_000_000),
        ),
    ]
    return state.model_copy(update=values).model_dump(mode="python")


def build_finsight_graph(
    runtime: AgentRuntime,
    max_retries: int = 1,
    *,
    checkpointer: Any | None = None,
) -> Any:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    graph = StateGraph(dict)

    def wrap(name: str, node: NodeCallable):
        async def run(raw_state: dict[str, object]) -> dict[str, object]:
            state = FinSightState.model_validate(raw_state)
            started = perf_counter_ns()
            update = await node(state, runtime)
            return _merge_state(
                state,
                update,
                node=name,
                started_ns=started,
                ended_ns=perf_counter_ns(),
            )

        return run

    async def run_verifier(raw_state: dict[str, object]) -> dict[str, object]:
        state = FinSightState.model_validate(raw_state)
        started = perf_counter_ns()
        update = await verifier_node(
            state,
            runtime,
            max_retries=max_retries,
        )
        return _merge_state(
            state,
            update,
            node="verifier",
            started_ns=started,
            ended_ns=perf_counter_ns(),
        )

    graph.add_node("supervisor", wrap("supervisor", supervisor_node))
    graph.add_node("research", wrap("research", research_node))
    graph.add_node("quant", wrap("quant", quant_node))
    graph.add_node("report", wrap("report", report_node))
    graph.add_node("verifier", run_verifier)
    graph.add_edge(START, "supervisor")

    def after_supervisor(raw_state: dict[str, object]) -> str:
        state = FinSightState.model_validate(raw_state)
        return END if state.needs_human_review else "research"

    def after_research(raw_state: dict[str, object]) -> str:
        state = FinSightState.model_validate(raw_state)
        if state.needs_human_review:
            return END
        if state.plan and "quant" in state.plan.capabilities:
            return "quant"
        return "report"

    def after_quant(raw_state: dict[str, object]) -> str:
        state = FinSightState.model_validate(raw_state)
        return END if state.needs_human_review else "report"

    def after_report(raw_state: dict[str, object]) -> str:
        state = FinSightState.model_validate(raw_state)
        return "verifier" if state.draft_answer else END

    def after_verifier(raw_state: dict[str, object]) -> str:
        state = FinSightState.model_validate(raw_state)
        if state.verification_passed or state.needs_human_review:
            return END
        return state.repair_target or "report"

    graph.add_conditional_edges("supervisor", after_supervisor)
    graph.add_conditional_edges("research", after_research)
    graph.add_conditional_edges("quant", after_quant)
    graph.add_conditional_edges("report", after_report)
    graph.add_conditional_edges("verifier", after_verifier)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class FinSightAgent:
    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        max_retries: int = 1,
        checkpointer: Any | None = None,
    ) -> None:
        self._graph = build_finsight_graph(
            runtime,
            max_retries=max_retries,
            checkpointer=checkpointer,
        )

    @staticmethod
    def _config(request_id: str) -> dict[str, object]:
        return {"configurable": {"thread_id": request_id}}

    async def ainvoke(self, request: FinSightState) -> AgentResult:
        raw = await self._graph.ainvoke(
            request.model_dump(mode="python"),
            config=self._config(request.request_id),
        )
        state = FinSightState.model_validate(raw)
        return AgentResult(
            request_id=state.request_id,
            final_answer=state.final_answer,
            citation_ids=state.citation_ids,
            needs_human_review=state.needs_human_review,
            human_review_reason=state.human_review_reason,
            issues=state.issues,
            traces=state.traces,
        )

    async def astream(self, request: FinSightState) -> AsyncIterator[AgentEvent]:
        sequence = 0
        yield AgentEvent(
            event_type="run_started",
            request_id=request.request_id,
            sequence=sequence,
        )
        sequence += 1
        final_state: dict[str, object] | None = None
        async for part in self._graph.astream(
            request.model_dump(mode="python"),
            config=self._config(request.request_id),
            stream_mode="updates",
        ):
            for node_name, raw_state in part.items():
                yield AgentEvent(
                    event_type="node_started",
                    request_id=request.request_id,
                    sequence=sequence,
                    node=node_name,
                )
                sequence += 1
                yield AgentEvent(
                    event_type="node_finished",
                    request_id=request.request_id,
                    sequence=sequence,
                    node=node_name,
                )
                sequence += 1
                final_state = raw_state
        if final_state:
            state = FinSightState.model_validate(final_state)
            if state.final_answer:
                yield AgentEvent(
                    event_type="answer_chunk",
                    request_id=request.request_id,
                    sequence=sequence,
                    payload={"text": state.final_answer},
                )
                sequence += 1
        yield AgentEvent(
            event_type="run_finished",
            request_id=request.request_id,
            sequence=sequence,
        )
