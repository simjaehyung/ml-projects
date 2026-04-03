from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Protocol


class WorkflowState(str, Enum):
    INIT = "INIT"
    DATA_READY = "DATA_READY"
    AUG_DONE = "AUG_DONE"
    BASE_TRAIN_DONE = "BASE_TRAIN_DONE"
    INFER_3FPS_DONE = "INFER_3FPS_DONE"
    REPORT_DONE = "REPORT_DONE"
    DONE = "DONE"
    FAILED = "FAILED"


TRANSITIONS = {
    WorkflowState.INIT: WorkflowState.DATA_READY,
    WorkflowState.DATA_READY: WorkflowState.AUG_DONE,
    WorkflowState.AUG_DONE: WorkflowState.BASE_TRAIN_DONE,
    WorkflowState.BASE_TRAIN_DONE: WorkflowState.INFER_3FPS_DONE,
    WorkflowState.INFER_3FPS_DONE: WorkflowState.REPORT_DONE,
    WorkflowState.REPORT_DONE: WorkflowState.DONE,
}


class WorkflowHook(Protocol):
    """Optional hook interface for future upper-layer states/extensions."""

    def before_state(self, state: WorkflowState, context: Dict[str, Any]) -> None:
        ...

    def after_state(self, state: WorkflowState, context: Dict[str, Any]) -> None:
        ...


@dataclass
class TransitionRecord:
    at: str
    from_state: str
    to_state: str
    ok: bool
    note: str = ""


@dataclass
class WorkflowJournal:
    records: List[TransitionRecord] = field(default_factory=list)

    def add(self, from_state: WorkflowState, to_state: WorkflowState, ok: bool, note: str = "") -> None:
        self.records.append(
            TransitionRecord(
                at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                from_state=from_state.value,
                to_state=to_state.value,
                ok=ok,
                note=note,
            )
        )

    def write(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.__dict__ for r in self.records]
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class StateMachine:
    def __init__(self, hooks: List[WorkflowHook] | None = None) -> None:
        self.state = WorkflowState.INIT
        self.journal = WorkflowJournal()
        self.hooks = hooks or []

    def next_state(self) -> WorkflowState:
        if self.state not in TRANSITIONS:
            raise RuntimeError(f"No transition defined for state={self.state}")
        return TRANSITIONS[self.state]

    def run_step(self, context: Dict[str, Any], handler) -> None:
        nxt = self.next_state()
        try:
            for h in self.hooks:
                h.before_state(nxt, context)
            handler(nxt, context)
            for h in self.hooks:
                h.after_state(nxt, context)
            self.journal.add(self.state, nxt, ok=True)
            self.state = nxt
        except Exception as exc:
            self.journal.add(self.state, WorkflowState.FAILED, ok=False, note=str(exc))
            self.state = WorkflowState.FAILED
            raise
