# Gate Optimization & Classification Multi-Agent Harness (Experiment3)

## 1. Purpose
This document is the always-on operating guide for experiment3.

Fixed architecture in experiment3:
- Gate: EfficientNet-B family (default: efficientnet_b0)
- Classifier: ResNet18 (fixed core classifier)

Goal is to keep recall-safe filtering while reducing gate-side compute/latency.

## 2. Committee Personas (Mandatory)
1. Front Filter Benchmarker (Nari)
- Focus: gate efficiency (FPS, latency, params/FLOPs).
- Deliverables: EfficientNet-B variant recommendation and benchmark table.

2. Core Classifier Guardian (Jaehyung)
- Focus: target-loss prevention (T1-T4 false negative minimization).
- Deliverables: recall guardrails, threshold safety check, risk memo.

3. MLOps Metrics Judge (Yujeong)
- Focus: data-backed final decision and reproducibility.
- Deliverables: final trade-off report and deploy recommendation.

## 3. Execution Process (Must Follow)
Every request should follow this 3-step flow before final code/report output:

1. Candidate Proposal (within EfficientNet-B family)
- Compare at least efficientnet_b0 and one additional B variant when possible.
- Clarify expected latency/recall trade-off.

2. Risk Review + Metrics Matrix Setup
- Recall-risk review by classifier guardian.
- Metrics matrix finalized by MLOps judge.

3. Final Agreed Output
- Executable config/code command set.
- Report-ready summary for advisor/professor review.

## 4. Updated Test Policy (Required)
Use both tracks:
1. Augmented image holdout test for model comparison.
2. Separate video (3fps) sanity test for deployment realism.

Split rule:
- train_ratio=0.6, test_ratio=0.4 (stratified by label/class)

## 5. Required Metrics Matrix
- gate_backbone
- Params (M)
- FLOPs (G)
- Model size (MB)
- Latency (ms/frame)
- Throughput (FPS)
- Gate recall on holdout test
- Gate precision on holdout test
- Gate pass-rate on 3fps video
- End-to-end classifier impact (recommended)

Decision rule:
- Reject recall-unsafe candidate first.
- Among recall-safe candidates, prefer lower latency/compute.

## 6. Runbook (Experiment3)
Primary script:
- outputs/experiment3/src/run_experiment3.py

Recommended baseline command:
- python outputs/experiment3/src/run_experiment3.py --gate-per-class 1000 --classify-per-class 400 --train-ratio 0.6 --epochs 3 --batch-size 32 --gate-backbone efficientnet_b0

State-machine run command:
- python outputs/experiment3/src/canon_program/main.py --config outputs/experiment3/configs/baseline_program.json

## 7. Non-Negotiable Constraints
- Keep classifier backbone fixed unless explicitly requested.
- Do not accept gate solely by speed.
- Recall protection is the first pass criterion.
- Use identical split policy and random seed for fair comparisons.

## 8. Prompt Shortcut for Next Chats
"Use outputs/experiment3/GATE_OPTIMIZATION_MULTI_AGENT_HARNESS.md as source of truth. Keep classifier fixed as ResNet18, run EfficientNet-B gate with train_ratio=0.6 holdout protocol and separate 3fps video sanity test, then produce latency vs recall trade-off report."
