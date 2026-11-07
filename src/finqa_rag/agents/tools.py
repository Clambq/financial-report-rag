"""Typed tool contracts and adapters for the FinSight workflow."""

from __future__ import annotations

import inspect
import re
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from finqa_rag.reasoning.program import ProgramError, execute_program
from finqa_rag.retrieval.types import ScoredEvidence


ToolErrorCode = Literal[
    "unknown_tool",
    "validation_error",
    "execution_error",
    "timeout",
]


def _not_blank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be blank")
    return normalized


class ToolFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    code: ToolErrorCode
    message: str
    retryable: bool = False


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    success: bool
    latency_ms: float = Field(ge=0)
    attempt: int = Field(default=1, ge=1)
    error_code: ToolErrorCode | None = None


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    report_id: str
    kind: Literal["text", "table"]
    text: str
    score: float
    rank: int = Field(ge=1)
    method: str


class RetrieveEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    query: str
    top_k: int = Field(default=5, gt=0, le=50)

    _validate_report_id = field_validator("report_id")(_not_blank)
    _validate_query = field_validator("query")(_not_blank)


class RetrieveEvidenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[EvidenceRecord]
    degraded: bool = False


class QueryReportTableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    row_selector: str

    _validate_report_id = field_validator("report_id")(_not_blank)
    _validate_row_selector = field_validator("row_selector")(_not_blank)


class TableRowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=0)
    values: list[str]


class QueryReportTableOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[TableRowRecord]


class ExecuteFinancialProgramInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    program: str

    _validate_report_id = field_validator("report_id")(_not_blank)
    _validate_program = field_validator("program")(_not_blank)


class ExecuteFinancialProgramOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    value_kind: Literal["decimal", "decision"]


class ComposeGroundedReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    evidence: list[EvidenceRecord]
    calculations: list[dict[str, object]]
    allowed_evidence_ids: list[str]

    _validate_question = field_validator("question")(_not_blank)


class ComposeGroundedReportOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citation_ids: list[str]
    allowed_evidence_ids: list[str]

    _validate_answer = field_validator("answer")(_not_blank)

    @model_validator(mode="after")
    def citations_must_be_allowed(self) -> ComposeGroundedReportOutput:
        invalid = sorted(set(self.citation_ids) - set(self.allowed_evidence_ids))
        if invalid:
            raise ValueError(f"citation IDs outside allowed evidence: {invalid}")
        return self


class VerifyGroundingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_answer: str
    citation_ids: list[str]
    allowed_evidence_ids: list[str]
    calculations: list[dict[str, object]]

    _validate_answer = field_validator("draft_answer")(_not_blank)

    @model_validator(mode="after")
    def citations_must_be_allowed(self) -> VerifyGroundingInput:
        invalid = sorted(set(self.citation_ids) - set(self.allowed_evidence_ids))
        if invalid:
            raise ValueError(f"citation IDs outside allowed evidence: {invalid}")
        return self


class VerificationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    recoverable: bool = False


class VerifyGroundingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)


InputModel = TypeVar("InputModel", bound=BaseModel)
OutputModel = TypeVar("OutputModel", bound=BaseModel)
ToolHandler = Callable[[BaseModel], BaseModel | Awaitable[BaseModel]]


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    input_type: type[BaseModel]
    output_type: type[BaseModel]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(
        self,
        name: str,
        input_type: type[InputModel],
        output_type: type[OutputModel],
        handler: Callable[[InputModel], OutputModel | Awaitable[OutputModel]],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = _RegisteredTool(
            input_type=input_type,
            output_type=output_type,
            handler=handler,  # type: ignore[arg-type]
        )

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    async def ainvoke(
        self,
        name: str,
        payload: dict[str, object],
    ) -> BaseModel:
        registered = self._tools.get(name)
        if registered is None:
            return ToolFailure(
                tool_name=name,
                code="unknown_tool",
                message=f"unknown tool: {name}",
            )
        try:
            validated = registered.input_type.model_validate(payload)
        except ValidationError as error:
            return ToolFailure(
                tool_name=name,
                code="validation_error",
                message=str(error),
            )
        result = registered.handler(validated)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolFailure):
            return result
        try:
            return registered.output_type.model_validate(result)
        except ValidationError as error:
            return ToolFailure(
                tool_name=name,
                code="validation_error",
                message=str(error),
            )


class EvidenceRetriever(Protocol):
    def search(self, report_id: str, question: str) -> list[ScoredEvidence]: ...


class RetrievalTool:
    def __init__(self, retriever: EvidenceRetriever) -> None:
        self._retriever = retriever

    def __call__(
        self,
        request: RetrieveEvidenceInput,
    ) -> RetrieveEvidenceOutput | ToolFailure:
        try:
            ranked = self._retriever.search(request.report_id, request.query)
        except (KeyError, ValueError) as error:
            return ToolFailure(
                tool_name="retrieve_evidence",
                code="execution_error",
                message=str(error),
            )
        evidence = [
            EvidenceRecord(
                evidence_id=item.evidence.evidence_id,
                report_id=item.evidence.report_id,
                kind=item.evidence.kind,
                text=item.evidence.text,
                score=item.score,
                rank=item.rank,
                method=item.method,
            )
            for item in ranked[: request.top_k]
        ]
        return RetrieveEvidenceOutput(evidence=evidence)


_WHITESPACE = re.compile(r"\s+")


def _normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", text).strip().casefold()


class ReportTableTool:
    def __init__(
        self,
        tables: Mapping[str, Sequence[Sequence[str]]],
    ) -> None:
        self._tables = {
            report_id: tuple(tuple(cell for cell in row) for row in table)
            for report_id, table in tables.items()
        }

    def __call__(
        self,
        request: QueryReportTableInput,
    ) -> QueryReportTableOutput | ToolFailure:
        table = self._tables.get(request.report_id)
        if table is None:
            return ToolFailure(
                tool_name="query_report_table",
                code="execution_error",
                message=f"unknown report_id: {request.report_id}",
            )
        wanted = _normalize_label(request.row_selector)
        rows = [
            TableRowRecord(row_index=index, values=list(row))
            for index, row in enumerate(table)
            if row and _normalize_label(row[0]) == wanted
        ]
        return QueryReportTableOutput(rows=rows)


class FinancialProgramTool:
    def __init__(
        self,
        tables: Mapping[str, Sequence[Sequence[str]]],
    ) -> None:
        self._tables = {
            report_id: tuple(tuple(cell for cell in row) for row in table)
            for report_id, table in tables.items()
        }

    def __call__(
        self,
        request: ExecuteFinancialProgramInput,
    ) -> ExecuteFinancialProgramOutput | ToolFailure:
        table = self._tables.get(request.report_id)
        if table is None:
            return ToolFailure(
                tool_name="execute_financial_program",
                code="execution_error",
                message=f"unknown report_id: {request.report_id}",
            )
        try:
            value = execute_program(request.program, table=table)
        except ProgramError as error:
            return ToolFailure(
                tool_name="execute_financial_program",
                code="execution_error",
                message=str(error),
            )
        return ExecuteFinancialProgramOutput(
            value=str(value),
            value_kind="decimal" if isinstance(value, Decimal) else "decision",
        )
