# Experiment3 GitHub Upload Report

## 1. What Was Done
This upload introduces a new experiment package (`outputs/experiment3`) based on experiment2, with the gate model switched to EfficientNet-B family while keeping the classifier fixed as ResNet18.

## 2. Main Changes
- New pipeline entrypoint: `src/run_experiment3.py`
- Gate backbone options added:
  - `efficientnet_b0` (default)
  - `efficientnet_b1`
  - `efficientnet_b2`
  - `efficientnet_b3`
- Classifier remains ResNet18 for fair comparison.
- State-machine workflow connected to experiment3 and `gate_backbone` config.
- New docs for handoff and multi-persona process.
- New easy web interface (Streamlit) for non-expert execution.

## 3. Added User Interface
- `src/web_ui_experiment3.py`: Streamlit-based runner
- `run_web_ui.ps1`: one-command launcher
- `requirements.txt`: dependency install list

With this interface, collaborators can run:
1) Direct pipeline with selectable gate backbone and run hyperparameters.
2) State-machine config runs (`smoke_program.json` or `baseline_program.json`).
3) Latest report preview.

## 4. Multi-Persona Process Included
- Source of truth:
  - `GATE_OPTIMIZATION_MULTI_AGENT_HARNESS.md`
- Personas:
  1. Nari (efficiency benchmark)
  2. Jaehyung (recall safety)
  3. Yujeong (final decision)
- Supporting smoke decision note:
  - `reports/PERSONA_DECISION_SMOKE.md` (runtime-generated, not tracked by git)

## 5. How Others Can Run Quickly
From `outputs/experiment3`:

```powershell
pip install -r requirements.txt
./run_web_ui.ps1
```

Or CLI:

```powershell
python src/run_experiment3.py --gate-per-class 40 --classify-per-class 20 --train-ratio 0.6 --epochs 1 --batch-size 16 --gate-backbone efficientnet_b0
```

## 6. Notes on Tracked vs Generated
- Tracked: source/config/docs/scripts
- Ignored: `artifacts/`, `generated/`, `reports/`, `__pycache__/`

This keeps the repository clean while allowing reproducible re-runs.
