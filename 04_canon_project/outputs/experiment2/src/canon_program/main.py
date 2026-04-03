from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_root = Path(__file__).resolve().parents[1]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


_bootstrap_src_path()

from canon_program.workflow import BaselineWorkflowProgram, ProgramConfig


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Deployable baseline state-machine workflow (without active learning).")
    parser.add_argument("--config", type=Path, default=default_root / "configs" / "baseline_program.json")
    return parser.parse_args()


def load_config(config_path: Path) -> ProgramConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment_root"] = Path(payload["experiment_root"])
    return ProgramConfig(**payload)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    app = BaselineWorkflowProgram(cfg)
    app.run()
    print("Baseline workflow completed.")


if __name__ == "__main__":
    main()
