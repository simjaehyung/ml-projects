# experiment2

Deployable baseline state-machine program (active learning excluded for now).

## Quick Start

```powershell
./bin/run_program.ps1
```

## Program Layout

- `bin/run_program.ps1`: single launcher script
- `configs/baseline_program.json`: runtime config
- `src/canon_program/main.py`: entrypoint
- `src/canon_program/state_machine.py`: finite-state engine + transition journal
- `src/canon_program/workflow.py`: baseline workflow implementation
- `src/run_experiment2.py`: reused training/inference functions

## State Outputs

- `reports/workflow_state_journal.json`
- `reports/workflow_state_snapshot.json`

## Main Artifacts

- `artifacts/*_manifest_*.csv`
- `artifacts/*_metrics.json`
- `artifacts/*_best.pt`
- `artifacts/eval_3fps_predictions.csv`
- `reports/EXPERIMENT2_REPORT.md`
