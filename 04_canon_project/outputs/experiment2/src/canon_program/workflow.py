from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from run_experiment2 import (
    build_paths,
    build_seed_inventory,
    evaluate_3fps,
    generate_augmented_sets,
    set_seed,
    split_manifest,
    train_one_task,
    write_report,
)

from .state_machine import StateMachine, WorkflowState


@dataclass
class ProgramConfig:
    experiment_root: Path
    gate_per_class: int = 1000
    classify_per_class: int = 400
    val_ratio: float = 0.2
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-4
    gate_threshold: float = 0.5
    seed: int = 42


class BaselineWorkflowProgram:
    """Baseline workflow only (active learning excluded by design)."""

    def __init__(self, cfg: ProgramConfig) -> None:
        self.cfg = cfg
        self.sm = StateMachine()
        self.context: Dict[str, Any] = {
            "config": asdict(cfg),
            "gate_metrics": None,
            "classify_metrics": None,
        }

    def run(self) -> None:
        set_seed(self.cfg.seed)
        while self.sm.state not in {WorkflowState.DONE, WorkflowState.FAILED}:
            self.sm.run_step(self.context, self._handle_state)

        paths = build_paths(self.cfg.experiment_root)
        self.sm.journal.write(paths.reports_root / "workflow_state_journal.json")
        (paths.reports_root / "workflow_state_snapshot.json").write_text(
            json.dumps(
                {
                    "state": self.sm.state.value,
                    "context": self.context,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _handle_state(self, state: WorkflowState, context: Dict[str, Any]) -> None:
        cfg = self.cfg
        paths = build_paths(cfg.experiment_root)

        if state == WorkflowState.DATA_READY:
            paths.artifacts_root.mkdir(parents=True, exist_ok=True)
            paths.reports_root.mkdir(parents=True, exist_ok=True)
            seed_inventory = build_seed_inventory(paths)
            context["seed_count_by_class"] = {k: len(v) for k, v in seed_inventory.items()}
            return

        if state == WorkflowState.AUG_DONE:
            seed_inventory = build_seed_inventory(paths)
            gate_df, cls_df = generate_augmented_sets(
                paths=paths,
                seed_inventory=seed_inventory,
                gate_per_class=cfg.gate_per_class,
                cls_per_class=cfg.classify_per_class,
                image_size=cfg.image_size,
                seed=cfg.seed,
            )
            gate_train, gate_val = split_manifest(gate_df, val_ratio=cfg.val_ratio, seed=cfg.seed)
            cls_train, cls_val = split_manifest(cls_df, val_ratio=cfg.val_ratio, seed=cfg.seed)

            gate_train.to_csv(paths.artifacts_root / "gate_manifest_train.csv", index=False)
            gate_val.to_csv(paths.artifacts_root / "gate_manifest_val.csv", index=False)
            cls_train.to_csv(paths.artifacts_root / "classify_manifest_train.csv", index=False)
            cls_val.to_csv(paths.artifacts_root / "classify_manifest_val.csv", index=False)
            return

        if state == WorkflowState.BASE_TRAIN_DONE:
            gate_train = paths.artifacts_root / "gate_manifest_train.csv"
            gate_val = paths.artifacts_root / "gate_manifest_val.csv"
            cls_train = paths.artifacts_root / "classify_manifest_train.csv"
            cls_val = paths.artifacts_root / "classify_manifest_val.csv"

            import pandas as pd

            gate_metrics = train_one_task(
                task_name="gate",
                train_df=pd.read_csv(gate_train),
                val_df=pd.read_csv(gate_val),
                num_classes=2,
                paths=paths,
                image_size=cfg.image_size,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                lr=cfg.lr,
                seed=cfg.seed,
            )
            cls_metrics = train_one_task(
                task_name="classify",
                train_df=pd.read_csv(cls_train),
                val_df=pd.read_csv(cls_val),
                num_classes=4,
                paths=paths,
                image_size=cfg.image_size,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                lr=cfg.lr,
                seed=cfg.seed,
            )
            context["gate_metrics"] = gate_metrics
            context["classify_metrics"] = cls_metrics
            return

        if state == WorkflowState.INFER_3FPS_DONE:
            eval_df = evaluate_3fps(
                paths=paths,
                gate_ckpt=paths.artifacts_root / "gate_best.pt",
                cls_ckpt=paths.artifacts_root / "classify_best.pt",
                image_size=cfg.image_size,
                gate_threshold=cfg.gate_threshold,
            )
            context["eval_3fps_count"] = int(len(eval_df))
            context["gate_pass_count"] = int((eval_df["gate_pred"] == 1).sum())
            return

        if state == WorkflowState.REPORT_DONE:
            if context.get("gate_metrics") is None or context.get("classify_metrics") is None:
                raise RuntimeError("Training metrics missing before report generation.")
            import pandas as pd

            eval_df = pd.read_csv(paths.artifacts_root / "eval_3fps_predictions.csv")
            write_report(
                paths=paths,
                gate_metrics=context["gate_metrics"],
                cls_metrics=context["classify_metrics"],
                eval_df=eval_df,
                gate_per_class=cfg.gate_per_class,
                cls_per_class=cfg.classify_per_class,
            )
            return

        if state == WorkflowState.DONE:
            return

        raise RuntimeError(f"Unhandled state: {state}")
