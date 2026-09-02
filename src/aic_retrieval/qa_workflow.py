"""In-memory Phase-7 workflow orchestration; persistence is added in P7.5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from aic_retrieval.qa_answering import EvidenceFirstAnswerer
from aic_retrieval.qa_confidence import require_confirmable
from aic_retrieval.qa_evidence import build_evidence_pack
from aic_retrieval.qa_normalization import AnswerNormalizer
from aic_retrieval.qa_question_router import QuestionRoute, RuleBasedQuestionRouter
from aic_retrieval.qa_schema import AnswerDraft, EvidencePack, QaRequest, QaReview, QaSession, ReviewDecision
from aic_retrieval.qa_store import QaStore


@dataclass
class QaWorkspaceState:
    session: QaSession
    pack: EvidencePack
    route: QuestionRoute
    drafts: dict[str, AnswerDraft]
    reviews: list[QaReview]


class QaWorkflow:
    """Coordinates a reviewable Q&A session without making external calls."""

    def __init__(self, router: RuleBasedQuestionRouter | None = None, answerer: EvidenceFirstAnswerer | None = None, store: QaStore | None = None) -> None:
        self.router = router or RuleBasedQuestionRouter()
        self.answerer = answerer or EvidenceFirstAnswerer()
        self.normalizer = AnswerNormalizer()
        self.store = store
        self._states: dict[str, QaWorkspaceState] = {}

    def prepare(
        self,
        request: QaRequest,
        retrieval_response: dict[str, Any],
        **evidence_kwargs: Any,
    ) -> QaWorkspaceState:
        pack = build_evidence_pack(request, retrieval_response, **evidence_kwargs)
        route = self.router.route(request.question)
        session = QaSession(
            session_id=f"qa-{uuid4().hex}",
            request=request,
            retrieval_context=pack.retrieval_context,
            created_at=_now(),
            selected_video_id=request.selected_video_id,
            selected_frame_id=request.selected_frame_id,
        )
        state = QaWorkspaceState(session, pack, route, {}, [])
        self._states[session.session_id] = state
        if self.store is not None:
            self.store.save_session(session, pack, route)
        return state

    def draft(
        self,
        session_id: str,
        selected_evidence_ids: tuple[str, ...] = (),
        answerer: Any | None = None,
    ) -> tuple[QaWorkspaceState, AnswerDraft]:
        state = self.get(session_id)
        if selected_evidence_ids:
            _require_single_video_evidence(state.pack, selected_evidence_ids)
        # The answerer creates a deterministic content identity. A persisted
        # draft is an audit event, so it must be unique even when a reviewer
        # submits the same evidence in this or another session.
        draft = replace(
            (answerer or self.answerer).draft(state.pack, state.route, selected_evidence_ids),
            draft_id=f"draft-{uuid4().hex}",
        )
        if draft.evidence_refs:
            _require_single_video_evidence(state.pack, draft.evidence_refs)
        state.drafts[draft.draft_id] = draft
        if self.store is not None:
            self.store.save_draft(session_id, draft)
        return state, draft

    def review(
        self,
        session_id: str,
        draft_id: str,
        decision: ReviewDecision,
        final_answer: str | None,
        selected_evidence_ids: tuple[str, ...],
        reviewer: str,
    ) -> tuple[QaWorkspaceState, QaReview]:
        state = self.get(session_id)
        draft = state.drafts.get(draft_id)
        if draft is None:
            raise ValueError("draft_id was not created in this session")
        if decision is not ReviewDecision.REJECTED:
            _require_single_video_evidence(state.pack, selected_evidence_ids)
            require_confirmable(draft, state.pack, selected_evidence_ids)
            answer = (final_answer or draft.raw_answer or "").strip()
            if not answer:
                raise ValueError("final_answer is required for confirmation")
            normalized = self.normalizer.normalize(answer)
            final_answer = normalized.raw_answer
        else:
            normalized = None
            final_answer = final_answer.strip() if final_answer else None
        review = QaReview(
            review_id=f"review-{uuid4().hex}",
            decision=decision,
            final_answer=final_answer,
            selected_evidence_refs=selected_evidence_ids,
            reviewer=reviewer,
            timestamp=_now(),
            draft_id=draft_id,
            normalized_final_answer=normalized.normalized_answer if normalized else None,
            alternative_answers=normalized.alternative_answers if normalized else (),
            normalization_rules_version=normalized.rules_version if normalized else None,
        )
        state.reviews.append(review)
        if self.store is not None:
            self.store.save_review(session_id, review)
        return state, review

    def get(self, session_id: str) -> QaWorkspaceState:
        state = self._states.get(session_id)
        if state is None:
            if self.store is None:
                raise ValueError("Q&A session was not found")
            stored = self.store.load_state(session_id)
            state = QaWorkspaceState(stored.session, stored.pack, stored.route, {item.draft_id: item for item in stored.drafts}, list(stored.reviews))
            self._states[session_id] = state
        return state

    def create_export(self, session_id: str, review_id: str):
        if self.store is None:
            raise ValueError("persistent Q&A store is not configured")
        return self.store.create_export(session_id, review_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.store.list_sessions() if self.store is not None else []

    def payload(self, state: QaWorkspaceState) -> dict[str, Any]:
        return state_payload(state, persistence="sqlite" if self.store is not None else "memory_until_p7_5")


def state_payload(state: QaWorkspaceState, persistence: str = "memory_until_p7_5") -> dict[str, Any]:
    return {
        "session": jsonable(state.session),
        "question_route": jsonable(state.route),
        "evidence_pack": jsonable(state.pack),
        "drafts": [jsonable(item) for item in state.drafts.values()],
        "reviews": [jsonable(item) for item in state.reviews],
        "persistence": persistence,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(jsonable(key)): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_single_video_evidence(pack: EvidencePack, evidence_ids: tuple[str, ...]) -> None:
    """Reject unknown or cross-video evidence before an answer can be reviewed."""
    evidence_by_id = {item.evidence_id: item for item in pack.evidence_refs}
    missing = [evidence_id for evidence_id in evidence_ids if evidence_id not in evidence_by_id]
    if missing:
        raise ValueError(f"selected evidence is not present in this Q&A session: {missing[0]}")
    video_ids = {evidence_by_id[evidence_id].video_id for evidence_id in evidence_ids}
    if len(video_ids) != 1:
        raise ValueError("Q&A selected evidence must belong to exactly one video")
