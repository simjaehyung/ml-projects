# experiment3

Deployable baseline pipeline for gate/classification experiment.

- Gate backbone: EfficientNet-B family
- Classifier backbone: ResNet18 (fixed)
- Evaluation policy: augmented holdout split (train_ratio=0.6) + separate 3fps video sanity check

## 1) Quick Start (UI)

```powershell
cd outputs/experiment3
pip install -r requirements.txt
./run_web_ui.ps1
```

Then open the Streamlit URL shown in terminal (usually http://localhost:8501).

## 2) Quick Start (CLI)

Direct pipeline:

```powershell
python src/run_experiment3.py --gate-per-class 40 --classify-per-class 20 --train-ratio 0.6 --epochs 1 --batch-size 16 --gate-backbone efficientnet_b0
```

State-machine smoke:

```powershell
python src/canon_program/main.py --config configs/smoke_program.json
```

## 3) GitHub Sharing Checklist

1. Commit and push the folder:

```powershell
git add outputs/experiment3
git commit -m "Add experiment3 with EfficientNet-B gate and easy runner UI"
git push
```

2. In repository README (or PR description), point users to this folder and this README.
3. Ask collaborators to install Python and run the Quick Start UI section.

## 4) Program Layout

- `run_web_ui.ps1`: one-command launcher for Streamlit interface
- `requirements.txt`: dependencies for easy setup
- `configs/baseline_program.json`: production-like run config
- `configs/smoke_program.json`: lightweight smoke config
- `src/web_ui_experiment3.py`: web interface
- `src/run_experiment3.py`: core training/inference pipeline
- `src/canon_program/main.py`: state-machine entrypoint
- `src/canon_program/state_machine.py`: finite-state engine + transition journal
- `src/canon_program/workflow.py`: baseline workflow implementation

## 5) Main Outputs

- `artifacts/*_manifest_*.csv`
- `artifacts/*_metrics.json`
- `artifacts/*_best.pt`
- `artifacts/eval_3fps_predictions.csv`
- `reports/EXPERIMENT3_REPORT.md`
- `reports/workflow_state_journal.json`
- `reports/workflow_state_snapshot.json`
