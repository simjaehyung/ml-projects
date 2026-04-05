# Program Structure (Deployable Baseline)

This structure excludes active learning by default and keeps extension points for upper-layer orchestration.

## Tree

```text
experiment3/
  bin/
    run_program.ps1                # one-command launcher
  configs/
    baseline_program.json          # runtime config
  src/
    run_experiment3.py             # existing pipeline functions (reused)
    canon_program/
      __init__.py
      state_machine.py             # finite-state engine + hooks
      workflow.py                  # baseline workflow implementation
      main.py                      # CLI entrypoint
  artifacts/
  reports/
```

## States

- INIT
- DATA_READY
- AUG_DONE
- BASE_TRAIN_DONE
- INFER_3FPS_DONE
- REPORT_DONE
- DONE
- FAILED

## Extension Concept

- `WorkflowHook` is reserved for future upper-layer states/events.
- Active learning is intentionally excluded from this baseline program.
