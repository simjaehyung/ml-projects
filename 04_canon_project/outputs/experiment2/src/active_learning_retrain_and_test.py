import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_experiment2 import (
    CLASS_NAMES,
    ImagePathDataset,
    build_paths,
    create_model,
    evaluate_3fps,
    set_seed,
    split_manifest,
    train_one_task,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


def find_latest_video(project_root: Path) -> Path:
    videos = [p for p in project_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        raise RuntimeError("No video files found in project.")
    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0]


def extract_video_to_fps(video_path: Path, out_dir: Path, target_fps: float = 3.0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = 30.0

    step = max(1, int(round(native_fps / target_fps)))
    idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            saved += 1
            out_name = f"{video_path.stem}_frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_dir / out_name), frame)
        idx += 1

    cap.release()
    return saved


def load_active_labels(label_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(label_csv)

    if "ox_target" in df.columns:
        ox = df["ox_target"].astype(str).str.strip().str.upper()
    else:
        ox = pd.Series([""] * len(df))

    if "note" in df.columns:
        note = df["note"].astype(str).str.strip().str.upper()
    else:
        note = pd.Series([""] * len(df))

    resolved = ox.where(ox.isin(["O", "X"]), note)
    resolved = resolved.where(resolved.isin(["O", "X"]), "")

    out = df.copy()
    out["resolved_ox"] = resolved
    out = out[out["resolved_ox"].isin(["O", "X"])].copy()

    out["label_id"] = out["resolved_ox"].map({"X": 0, "O": 1})
    out["label"] = out["resolved_ox"].map({"X": "non_target", "O": "target"})
    out["origin"] = "active_labeled"

    keep_cols = ["path", "label", "label_id", "origin"]
    for c in keep_cols:
        if c not in out.columns:
            raise RuntimeError(f"Required column missing after processing: {c}")
    return out[keep_cols].drop_duplicates(subset=["path"]).reset_index(drop=True)


def oversample_active(df: pd.DataFrame, pos_repeat: int, neg_repeat: int) -> pd.DataFrame:
    pos = df[df["label_id"] == 1]
    neg = df[df["label_id"] == 0]

    parts = [df]
    for _ in range(max(0, pos_repeat - 1)):
        parts.append(pos)
    for _ in range(max(0, neg_repeat - 1)):
        parts.append(neg)

    return pd.concat(parts, ignore_index=True)


def evaluate_folder(gate_ckpt: Path, cls_ckpt: Path, eval_root: Path, image_size: int, gate_threshold: float) -> pd.DataFrame:
    from run_experiment2 import load_model, preprocess_eval, read_image, list_images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_model = load_model(gate_ckpt, num_classes=2, device=device)
    cls_model = load_model(cls_ckpt, num_classes=4, device=device)

    frames = list_images(eval_root)
    rows = []

    with torch.no_grad():
        for p in frames:
            img = read_image(p)
            x = preprocess_eval(img, image_size=image_size).to(device)

            gate_logits = gate_model(x)
            gate_prob = torch.softmax(gate_logits, dim=1)[0, 1].item()
            gate_pred = 1 if gate_prob >= gate_threshold else 0

            cls_pred = "non_target"
            cls_conf = 0.0
            if gate_pred == 1:
                cls_logits = cls_model(x)
                cls_prob = torch.softmax(cls_logits, dim=1)[0]
                cls_idx = int(torch.argmax(cls_prob).item())
                cls_pred = CLASS_NAMES[cls_idx]
                cls_conf = float(cls_prob[cls_idx].item())

            rows.append(
                {
                    "path": str(p),
                    "gate_prob_target": gate_prob,
                    "gate_pred": gate_pred,
                    "cls_pred": cls_pred,
                    "cls_conf": cls_conf,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    default_experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Apply active labels, retrain gate, test on 3fps video frames")
    parser.add_argument("--experiment-root", type=Path, default=default_experiment_root)
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=default_experiment_root / "artifacts" / "labeling" / "ambiguous_candidates_for_ox_labeling.csv",
    )
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--extract-fps", type=float, default=3.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-pos-repeat", type=int, default=4)
    parser.add_argument("--active-neg-repeat", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    paths = build_paths(args.experiment_root)
    paths.artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.reports_root.mkdir(parents=True, exist_ok=True)

    gate_train = pd.read_csv(paths.artifacts_root / "gate_manifest_train.csv")
    gate_val = pd.read_csv(paths.artifacts_root / "gate_manifest_val.csv")

    active_df = load_active_labels(args.label_csv)
    active_df_aug = oversample_active(active_df, pos_repeat=args.active_pos_repeat, neg_repeat=args.active_neg_repeat)

    gate_train2 = pd.concat([gate_train, active_df_aug], ignore_index=True)

    gate_metrics = train_one_task(
        task_name="gate_active",
        train_df=gate_train2,
        val_df=gate_val,
        num_classes=2,
        paths=paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    cls_ckpt = paths.artifacts_root / "classify_best.pt"
    gate_ckpt = paths.artifacts_root / "gate_active_best.pt"

    if args.video_path is None:
        video_path = find_latest_video(paths.project_root)
    else:
        video_path = args.video_path

    eval_3fps_dir = paths.artifacts_root / "eval_video_3fps"
    if eval_3fps_dir.exists():
        for p in eval_3fps_dir.glob("*"):
            if p.is_file():
                p.unlink()

    n_saved = extract_video_to_fps(video_path=video_path, out_dir=eval_3fps_dir, target_fps=args.extract_fps)

    eval_df = evaluate_folder(
        gate_ckpt=gate_ckpt,
        cls_ckpt=cls_ckpt,
        eval_root=eval_3fps_dir,
        image_size=args.image_size,
        gate_threshold=args.gate_threshold,
    )

    pred_csv = paths.artifacts_root / "eval_video_3fps_predictions_active.csv"
    eval_df.to_csv(pred_csv, index=False)

    summary = {
        "video_path": str(video_path),
        "saved_3fps_frames": int(n_saved),
        "active_labels_count": int(len(active_df)),
        "active_O_count": int((active_df["label_id"] == 1).sum()),
        "active_X_count": int((active_df["label_id"] == 0).sum()),
        "gate_threshold": args.gate_threshold,
        "gate_pass_count": int((eval_df["gate_pred"] == 1).sum()),
        "eval_frame_count": int(len(eval_df)),
        "class_distribution": eval_df["cls_pred"].value_counts().to_dict(),
        "gate_metrics": {
            "best_val_f1_macro": gate_metrics["best_val_f1_macro"],
            "final_val_acc": gate_metrics["final_val_acc"],
            "final_val_f1_macro": gate_metrics["final_val_f1_macro"],
        },
    }

    summary_path = paths.reports_root / "ACTIVE_LEARNING_TEST_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# Active Learning Retrain + New Video 3fps Test",
        "",
        f"- Video: {video_path}",
        f"- 3fps extracted frames: {n_saved}",
        f"- Active labels used: {len(active_df)} (O={(active_df['label_id'] == 1).sum()}, X={(active_df['label_id'] == 0).sum()})",
        f"- Gate threshold: {args.gate_threshold}",
        "",
        "## Gate Retrain Metrics",
        f"- best_val_f1_macro: {gate_metrics['best_val_f1_macro']:.4f}",
        f"- final_val_acc: {gate_metrics['final_val_acc']:.4f}",
        f"- final_val_f1_macro: {gate_metrics['final_val_f1_macro']:.4f}",
        "",
        "## New Video 3fps Inference",
        f"- eval frames: {len(eval_df)}",
        f"- gate pass frames: {(eval_df['gate_pred'] == 1).sum()}",
        "- class distribution:",
    ]
    for k, v in eval_df["cls_pred"].value_counts().to_dict().items():
        md_lines.append(f"  - {k}: {v}")

    md_lines += [
        "",
        f"- predictions csv: {pred_csv}",
        f"- summary json: {summary_path}",
    ]

    report_md = paths.reports_root / "ACTIVE_LEARNING_TEST_SUMMARY.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("Active-learning retrain and test completed.")
    print(f"Report: {report_md}")


if __name__ == "__main__":
    main()
