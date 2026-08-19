"""Local-only orchestration for TRAKE planning, retrieval and alignment."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from aic_retrieval.trake_alignment import AlignmentResult, ChainEvent, TemporalAligner, TrakeChain, validate_chain
from aic_retrieval.trake_candidates import EventCandidatePool, TrakeCandidate, TrakeRetrievalAdapter, VideoCandidate, group_candidates_by_video
from aic_retrieval.trake_decomposition import DecompositionResult, RuleBasedTrakeDecomposer
from aic_retrieval.trake_schema import Availability, TrakeEvent, TrakeRequest, request_from_dict
from aic_retrieval.trake_scoring import ScoreBreakdown, score_chain
from aic_retrieval.trake_store import TrakeStore


WORKFLOW_VERSION = "phase8-workflow-v1"


@dataclass
class TrakeWorkflowState:
    session_id: str
    request: TrakeRequest
    decomposition_warnings: tuple[str, ...]
    created_at_epoch: float
    pools: tuple[EventCandidatePool, ...] = ()
    videos: tuple[VideoCandidate, ...] = ()
    alignments: tuple[AlignmentResult, ...] = ()
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    plan_revision: int = 1
    locked_event_ids: tuple[str, ...] = ()
    manual_chain: TrakeChain | None = None


class TrakeWorkflow:
    def __init__(
        self,
        event_retriever: Callable[[TrakeEvent, int], list[dict[str, Any]]],
        *,
        modality_availability: dict[str, Availability] | None = None,
        store: TrakeStore | None = None,
    ) -> None:
        self.decomposer = RuleBasedTrakeDecomposer()
        self.adapter = TrakeRetrievalAdapter(event_retriever)
        self.aligner = TemporalAligner()
        self.modality_availability = dict(modality_availability or {"clip": Availability.AVAILABLE})
        self.sessions: dict[str, TrakeWorkflowState] = {}
        self.store = store

    def plan(
        self,
        query_id: str,
        query: str,
        *,
        manual_events: list[str] | None = None,
        constraints: dict[str, Any] | None = None,
        group: str = "L21",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = (
            self.decomposer.manual_override(query_id, query, manual_events, group=group, constraints=constraints)
            if manual_events is not None
            else self.decomposer.decompose(query_id, query, group=group)
        )
        request = result.request
        if constraints and manual_events is None:
            payload = request.to_dict(); payload["constraints"] = constraints
            request = request_from_dict(payload)
        session_id = f"trake-{uuid4().hex}"
        state = TrakeWorkflowState(session_id, request, result.warnings, time.time())
        state.stage_timings_ms["plan"] = round((time.perf_counter() - started) * 1000, 3)
        self.sessions[session_id] = state
        return self._persist(state)

    def search(self, session_id: str, *, event_pool_size: int = 60, max_per_video: int = 8, video_pool_size: int = 20) -> dict[str, Any]:
        state = self._state(session_id)
        started = time.perf_counter()
        state.pools = self.adapter.retrieve_request(
            state.request,
            pool_size=event_pool_size,
            max_per_video=max_per_video,
            availability=self.modality_availability,
        )
        state.videos = group_candidates_by_video(state.request, state.pools, video_pool_size=video_pool_size)
        state.alignments = ()
        state.stage_timings_ms["search"] = round((time.perf_counter() - started) * 1000, 3)
        return self._persist(state)

    def align(self, session_id: str, *, top_k_per_video: int = 5, top_videos: int = 20) -> dict[str, Any]:
        state = self._state(session_id)
        if not state.videos:
            raise ValueError("TRAKE search must run before alignment")
        started = time.perf_counter()
        state.alignments = self.aligner.align_videos(state.request, state.videos, top_k_per_video=top_k_per_video, top_videos=top_videos)
        state.stage_timings_ms["align"] = round((time.perf_counter() - started) * 1000, 3)
        return self._persist(state)

    def update_plan(self, session_id: str, events: list[dict[str, Any]], constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self._state(session_id)
        payload = state.request.to_dict()
        payload["events"] = events
        if constraints is not None: payload["constraints"] = constraints
        state.request = request_from_dict(payload)
        state.plan_revision += 1
        state.pools = (); state.videos = (); state.alignments = (); state.manual_chain = None; state.locked_event_ids = ()
        return self._persist(state)

    def replace_candidate(self, session_id: str, chain_id: str, event_id: str, candidate_id: str, *, lock_event: bool = True) -> dict[str, Any]:
        state = self._state(session_id)
        source = state.manual_chain if state.manual_chain and state.manual_chain.chain_id == chain_id else next((chain for result in state.alignments for chain in result.chains if chain.chain_id == chain_id), None)
        if source is None: raise KeyError(f"unknown TRAKE chain: {chain_id}")
        replacement = next((item for pool in state.pools for item in pool.candidates if item.candidate_id == candidate_id and item.event_id == event_id), None)
        if replacement is None: raise KeyError(f"unknown candidate for event {event_id}: {candidate_id}")
        selected: list[TrakeCandidate | None] = []
        events: list[ChainEvent] = []
        for entry in source.events:
            candidate = replacement if entry.event_id == event_id else entry.candidate
            selected.append(candidate)
            events.append(ChainEvent(entry.event_id, entry.required, candidate, "user_replacement" if entry.event_id == event_id else entry.selection))
        required = tuple(event.required for event in state.request.events)
        breakdown = score_chain(tuple(selected), required, 0, self.aligner.score_config)
        manual = TrakeChain(f"{source.chain_id}:manual:{uuid4().hex[:8]}", source.video_id, tuple(events), breakdown, (), manual=True)
        valid, failures = validate_chain(state.request, manual, allow_same_frame=bool(state.request.constraints.get("allow_same_frame", False)))
        if not valid: raise ValueError("manual replacement violates TRAKE invariants: " + ", ".join(failures))
        state.manual_chain = manual
        if lock_event:
            state.locked_event_ids = tuple(dict.fromkeys((*state.locked_event_ids, event_id)))
        return self._persist(state)

    def review(self, session_id: str, chain_id: str, decision: str, reviewer: str) -> dict[str, Any]:
        state = self._state(session_id)
        if decision not in {"confirmed", "rejected"}: raise ValueError("decision must be confirmed or rejected")
        chain = state.manual_chain if state.manual_chain and state.manual_chain.chain_id == chain_id else next((item for result in state.alignments for item in result.chains if item.chain_id == chain_id), None)
        if chain is None: raise KeyError(f"unknown TRAKE chain: {chain_id}")
        valid, failures = validate_chain(state.request, chain, allow_same_frame=bool(state.request.constraints.get("allow_same_frame", False)))
        if decision == "confirmed" and not valid: raise ValueError("cannot confirm invalid chain: " + ", ".join(failures))
        review = {"review_id": f"trake-review-{uuid4().hex}", "session_id": session_id, "chain_id": chain_id, "decision": decision, "reviewer": reviewer.strip() or "local-reviewer", "created_at": _now(), "plan_revision": state.plan_revision, "valid_at_review": valid}
        if self.store: self.store.append_review(review)
        return {"review": review, **self._persist(state)}

    def export(self, session_id: str, review_id: str) -> dict[str, Any]:
        state = self._state(session_id)
        reviews = self.store.reviews(session_id) if self.store else []
        review = next((item for item in reviews if item["review_id"] == review_id), None)
        if not review or review["decision"] != "confirmed": raise ValueError("internal export requires a confirmed review")
        chain = state.manual_chain if state.manual_chain and state.manual_chain.chain_id == review["chain_id"] else next((item for result in state.alignments for item in result.chains if item.chain_id == review["chain_id"]), None)
        if chain is None: raise ValueError("reviewed chain is unavailable")
        valid, failures = validate_chain(state.request, chain, allow_same_frame=bool(state.request.constraints.get("allow_same_frame", False)))
        if not valid: raise ValueError("cannot export invalid chain: " + ", ".join(failures))
        if not self.store: raise ValueError("TRAKE persistence is required for export")
        algorithm_chain_id=chain.chain_id.split(":manual:",1)[0]
        algorithm_chain=next((item for result in state.alignments for item in result.chains if item.chain_id==algorithm_chain_id),chain)
        algorithm_by_event={item.event_id:item.candidate for item in algorithm_chain.events}
        refinements=self.store.refinements(session_id)
        event_records=[]
        for item in chain.events:
            if item.candidate is None: continue
            algorithm=algorithm_by_event.get(item.event_id)
            event_records.append({
                "event_id":item.event_id,"candidate_id":item.candidate.candidate_id,"keyframe_id":item.candidate.keyframe_id,
                "frame_idx":item.candidate.frame_idx,"pts_time":item.candidate.pts_time,"keyframe_path":item.candidate.keyframe_path,
                "mapping":{"source":"validated_index_ref","frame_idx":item.candidate.frame_idx,"pts_time":item.candidate.pts_time,"fps":item.candidate.evidence.get("fps") or item.candidate.evidence.get("raw_result",{}).get("fps")},
                "evidence":item.candidate.evidence,"provenance":item.candidate.provenance,"retrieval_method":item.candidate.retrieval_method,
                "algorithm_selection":algorithm.to_dict() if algorithm else None,"final_selection":item.candidate.to_dict(),
                "manually_replaced":bool(algorithm and algorithm.candidate_id!=item.candidate.candidate_id),
                "refinements":[value for value in refinements if value.get("event_id")==item.event_id],
            })
        record={"export_id":f"trake-export-{uuid4().hex}","session_id":session_id,"review_id":review_id,"created_at":_now(),"schema":"phase8-internal-record-v1","not_final_btc_format":True,"query_id":state.request.query_id,"plan_revision":state.plan_revision,"video_id":chain.video_id,"same_video":all(item.candidate is None or item.candidate.video_id==chain.video_id for item in chain.events),"temporally_ordered":valid,"score":jsonable(chain.score),"events":event_records,"confirmed_by":review["reviewer"],"confirmed_at":review["created_at"],"human_confirmed":True}
        self.store.append_export(record)
        return {"export": record}

    def get(self, session_id: str) -> dict[str, Any]:
        payload = state_payload(self._state(session_id))
        payload["reviews"] = self.store.reviews(session_id) if self.store else []
        return payload

    def list_sessions(self) -> dict[str, Any]:
        if self.store: return {"sessions": self.store.list_sessions()}
        items = sorted(self.sessions.values(), key=lambda item: (-item.created_at_epoch, item.session_id))
        return {"sessions": [{"session_id": item.session_id, "query_id": item.request.query_id, "original_query": item.request.original_query, "plan_revision": item.plan_revision} for item in items]}

    def _state(self, session_id: str) -> TrakeWorkflowState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            payload = self.store.load_state(session_id) if self.store else None
            if payload is not None:
                state = _state_from_payload(payload); self.sessions[session_id] = state; return state
            raise KeyError(f"unknown TRAKE session: {session_id}") from exc

    def _persist(self, state: TrakeWorkflowState) -> dict[str, Any]:
        payload = state_payload(state)
        if self.store: self.store.save_state(payload)
        return payload


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def state_payload(state: TrakeWorkflowState) -> dict[str, Any]:
    return {"workflow_version": WORKFLOW_VERSION, "state": jsonable(state)}


def _candidate(payload: dict[str, Any]) -> TrakeCandidate:
    values=dict(payload)
    for name in ("provenance","warnings"): values[name]=tuple(values.get(name,()))
    return TrakeCandidate(**values)


def _chain(payload: dict[str, Any]) -> TrakeChain:
    events=tuple(ChainEvent(item["event_id"],item["required"],_candidate(item["candidate"]) if item.get("candidate") else None,item.get("selection","algorithm")) for item in payload["events"])
    score=ScoreBreakdown(**payload["score"])
    return TrakeChain(payload["chain_id"],payload["video_id"],events,score,tuple(payload.get("warnings",())),payload.get("valid",True),payload.get("manual",False),payload.get("version","phase8-temporal-dp-v1"))


def _state_from_payload(payload: dict[str, Any]) -> TrakeWorkflowState:
    request=request_from_dict(payload["request"]); event_by_id={item.event_id:item for item in request.events}
    pools=tuple(EventCandidatePool(event_by_id[item["event"]["event_id"]],tuple(_candidate(candidate) for candidate in item["candidates"]),{key:Availability(value) for key,value in item.get("modality_availability",{}).items()},tuple(item.get("warnings",()))) for item in payload.get("pools",()))
    videos=tuple(VideoCandidate(item["video_id"],{key:tuple(_candidate(candidate) for candidate in values) for key,values in item["event_candidates"].items()},item["required_coverage"],item["required_total"],item["optional_coverage"],item["optional_total"],item["complete"],tuple(item.get("warnings",()))) for item in payload.get("videos",()))
    alignments=tuple(AlignmentResult(item["video_id"],tuple(_chain(chain) for chain in item.get("chains",())),tuple(item.get("failures",()))) for item in payload.get("alignments",()))
    return TrakeWorkflowState(payload["session_id"],request,tuple(payload.get("decomposition_warnings",())),payload["created_at_epoch"],pools,videos,alignments,dict(payload.get("stage_timings_ms",{})),payload.get("plan_revision",1),tuple(payload.get("locked_event_ids",())),_chain(payload["manual_chain"]) if payload.get("manual_chain") else None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
