# HANDOFF FOR NEXT CHAT (EXPERIMENT3)

## 1) Source of Truth
- Always follow: outputs/experiment3/GATE_OPTIMIZATION_MULTI_AGENT_HARNESS.md
- Experiment3 fixed policy:
  - Gate backbone: EfficientNet-B family (default efficientnet_b0)
  - Classifier backbone: ResNet18 fixed
  - Holdout split: train_ratio=0.6, test_ratio=0.4
  - Video evaluation: separate 3fps sanity track

## 2) Core Run Commands
- Direct pipeline:
  - python outputs/experiment3/src/run_experiment3.py --gate-per-class 1000 --classify-per-class 400 --train-ratio 0.6 --epochs 3 --batch-size 32 --gate-backbone efficientnet_b0
- State-machine baseline:
  - python outputs/experiment3/src/canon_program/main.py --config outputs/experiment3/configs/baseline_program.json

## 3) Main Files
- outputs/experiment3/src/run_experiment3.py
- outputs/experiment3/src/canon_program/workflow.py
- outputs/experiment3/src/canon_program/main.py
- outputs/experiment3/configs/baseline_program.json
- outputs/experiment3/configs/smoke_program.json
- outputs/experiment3/run_experiment3.ps1

## 4) Output Targets
- artifacts/gate_manifest_train.csv
- artifacts/gate_manifest_test.csv
- artifacts/classify_manifest_train.csv
- artifacts/classify_manifest_test.csv
- artifacts/gate_metrics.json
- artifacts/classify_metrics.json
- artifacts/eval_3fps_predictions.csv
- reports/EXPERIMENT3_REPORT.md

## 5) Persona Workflow Reminder
1. Nari: benchmark/efficiency table.
2. Jaehyung: recall safety and threshold risks.
3. Yujeong: final trade-off decision and deploy recommendation.
