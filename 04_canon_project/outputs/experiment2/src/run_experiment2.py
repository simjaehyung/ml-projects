import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["t1", "t2", "t3", "t4"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        last_err = None
        for _ in range(5):
            try:
                shutil.rmtree(path)
                last_err = None
                break
            except OSError as err:
                last_err = err
                time.sleep(0.25)
        if last_err is not None:
            raise last_err
    path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def gate_transform(size: int) -> A.Compose:
    return A.Compose(
        [
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 11), p=1.0),
                ],
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.RandomGamma(gamma_limit=(75, 130), p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.06), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.5,
            ),
            A.ImageCompression(quality_range=(35, 90), p=0.5),
            A.Perspective(scale=(0.02, 0.08), keep_size=True, p=0.45),
            A.Affine(scale=(0.9, 1.08), translate_percent=(-0.08, 0.08), rotate=(-8, 8), p=0.5),
            A.RandomResizedCrop(size=(size, size), scale=(0.88, 1.0), ratio=(0.93, 1.07), p=0.7),
            A.Resize(size, size),
        ]
    )


def classify_transform(size: int) -> A.Compose:
    return A.Compose(
        [
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.25,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.55),
            A.GaussNoise(std_range=(0.005, 0.025), p=0.25),
            A.Perspective(scale=(0.01, 0.035), keep_size=True, p=0.2),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-4, 4), p=0.35),
            A.RandomResizedCrop(size=(size, size), scale=(0.93, 1.0), ratio=(0.97, 1.03), p=0.5),
            A.Resize(size, size),
        ]
    )


@dataclass
class Paths:
    project_root: Path
    experiment_root: Path
    target_root: Path
    negative_root: Path
    fps1_root: Path
    fps3_root: Path
    generated_root: Path
    artifacts_root: Path
    reports_root: Path


class ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = read_image(Path(row["path"]))
        x = self.transform(image)
        y = int(row["label_id"])
        return x, y


def build_paths(experiment_root: Path) -> Paths:
    project_root = experiment_root.parents[1]
    return Paths(
        project_root=project_root,
        experiment_root=experiment_root,
        target_root=project_root / "target",
        negative_root=project_root / "dataset_video1" / "not_similar_1fps",
        fps1_root=project_root / "dataset_video1" / "extracted_1fps",
        fps3_root=project_root / "dataset_video1" / "extracted_3fps",
        generated_root=experiment_root / "generated",
        artifacts_root=experiment_root / "artifacts",
        reports_root=experiment_root / "reports",
    )


def build_seed_inventory(paths: Paths) -> Dict[str, List[Path]]:
    inventory = {}
    for cls in CLASS_NAMES:
        cls_dir = paths.target_root / cls
        imgs = list_images(cls_dir)
        if len(imgs) == 0:
            raise RuntimeError(f"No seed images found in {cls_dir}")
        inventory[cls] = imgs
    rows = []
    for cls, imgs in inventory.items():
        for p in imgs:
            rows.append({"class": cls, "path": str(p)})
    pd.DataFrame(rows).to_csv(paths.artifacts_root / "target_seed_inventory.csv", index=False)
    return inventory


def generate_augmented_sets(
    paths: Paths,
    seed_inventory: Dict[str, List[Path]],
    gate_per_class: int,
    cls_per_class: int,
    image_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    ensure_clean_dir(paths.generated_root)

    gate_aug = gate_transform(image_size)
    cls_aug = classify_transform(image_size)

    gate_rows = []
    cls_rows = []

    gate_positive_dir = paths.generated_root / "gate" / "target"
    cls_root = paths.generated_root / "classify"

    for cls_id, cls in enumerate(CLASS_NAMES):
        seeds = seed_inventory[cls]
        for i in range(gate_per_class):
            src = rng.choice(seeds)
            img = read_image(src)
            out = gate_aug(image=img)["image"]
            out_path = gate_positive_dir / cls / f"{cls}_g_{i:04d}.jpg"
            write_image(out_path, out)
            gate_rows.append(
                {
                    "path": str(out_path),
                    "label": "target",
                    "label_id": 1,
                    "source_class": cls,
                    "origin": "target_aug",
                }
            )

        for i in range(cls_per_class):
            src = rng.choice(seeds)
            img = read_image(src)
            out = cls_aug(image=img)["image"]
            out_path = cls_root / cls / f"{cls}_c_{i:04d}.jpg"
            write_image(out_path, out)
            cls_rows.append(
                {
                    "path": str(out_path),
                    "label": cls,
                    "label_id": cls_id,
                    "source_class": cls,
                    "origin": "target_aug",
                }
            )

    neg_candidates = list_images(paths.negative_root)
    if len(neg_candidates) == 0:
        raise RuntimeError(f"No negative images found in {paths.negative_root}")
    required_neg = gate_per_class * len(CLASS_NAMES)
    gate_negative_dir = paths.generated_root / "gate" / "non_target"
    for i in range(required_neg):
        src = rng.choice(neg_candidates)
        img = read_image(src)
        out = gate_aug(image=img)["image"]
        out_path = gate_negative_dir / f"neg_g_{i:04d}.jpg"
        write_image(out_path, out)
        gate_rows.append(
            {
                "path": str(out_path),
                "label": "non_target",
                "label_id": 0,
                "source_class": "non_target",
                "origin": "negative_aug",
            }
        )

    gate_df = pd.DataFrame(gate_rows)
    cls_df = pd.DataFrame(cls_rows)

    gate_df.to_csv(paths.artifacts_root / "gate_manifest_all.csv", index=False)
    cls_df.to_csv(paths.artifacts_root / "classify_manifest_all.csv", index=False)
    return gate_df, cls_df


def split_manifest(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=df["label_id"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def create_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_task(
    task_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_classes: int,
    paths: Paths,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes, device)

    train_ds = ImagePathDataset(train_df, image_size=image_size)
    val_ds = ImagePathDataset(val_df, image_size=image_size)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = -1.0
    history = []
    ckpt_path = paths.artifacts_root / f"{task_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                y_pred.extend(pred)
                y_true.extend(y.numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": acc, "val_f1_macro": f1})

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, ckpt_path)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    result = {
        "task": task_name,
        "device": str(device),
        "best_val_f1_macro": best_f1,
        "final_val_acc": float(history[-1]["val_acc"]),
        "final_val_f1_macro": float(history[-1]["val_f1_macro"]),
        "history": history,
        "classification_report": report,
        "checkpoint": str(ckpt_path),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
    }
    with open(paths.artifacts_root / f"{task_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def load_model(ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = create_model(num_classes, device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess_eval(image: np.ndarray, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tfm(image).unsqueeze(0)


def evaluate_3fps(
    paths: Paths,
    gate_ckpt: Path,
    cls_ckpt: Path,
    image_size: int,
    gate_threshold: float,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_model = load_model(gate_ckpt, num_classes=2, device=device)
    cls_model = load_model(cls_ckpt, num_classes=4, device=device)

    frames = list_images(paths.fps3_root)
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

    df = pd.DataFrame(rows)
    df.to_csv(paths.artifacts_root / "eval_3fps_predictions.csv", index=False)
    return df


def write_report(
    paths: Paths,
    gate_metrics: Dict,
    cls_metrics: Dict,
    eval_df: pd.DataFrame,
    gate_per_class: int,
    cls_per_class: int,
) -> None:
    pred_dist = eval_df["cls_pred"].value_counts().to_dict()
    gate_pass = int((eval_df["gate_pred"] == 1).sum())
    total = int(len(eval_df))
    gate_pass_ratio = (gate_pass / total * 100.0) if total else 0.0

    top_gate = eval_df.sort_values("gate_prob_target", ascending=False).head(20)
    top_gate.to_csv(paths.artifacts_root / "eval_3fps_top20_gate.csv", index=False)

    report_lines = [
        "# Experiment2 End-to-End Report",
        "",
        "## 1. Target File Location Check",
        f"- Target root: {paths.target_root}",
        f"- Seed inventory CSV: {paths.artifacts_root / 'target_seed_inventory.csv'}",
        "- Class folders used: t1, t2, t3, t4",
        "",
        "## 2. Offline Augmentation Strategy",
        "- Tool: Albumentations",
        f"- Gate augmentation samples per class: {gate_per_class}",
        f"- Classification augmentation samples per class: {cls_per_class}",
        "- Gate uses wider real-world perturbations; Classification uses conservative perturbations.",
        "",
        "## 3. Training Results",
        f"- Gate best val macro-F1: {gate_metrics['best_val_f1_macro']:.4f}",
        f"- Gate final val acc: {gate_metrics['final_val_acc']:.4f}",
        f"- Classification best val macro-F1: {cls_metrics['best_val_f1_macro']:.4f}",
        f"- Classification final val acc: {cls_metrics['final_val_acc']:.4f}",
        "",
        "## 4. 3fps Evaluation",
        f"- 3fps frames evaluated: {total}",
        f"- Gate passed frames: {gate_pass} ({gate_pass_ratio:.2f}%)",
        "- Predicted class distribution:",
    ]

    for k, v in sorted(pred_dist.items()):
        report_lines.append(f"  - {k}: {v}")

    report_lines.extend(
        [
            "",
            "## 5. Output Artifacts",
            f"- Gate metrics: {paths.artifacts_root / 'gate_metrics.json'}",
            f"- Classification metrics: {paths.artifacts_root / 'classify_metrics.json'}",
            f"- 3fps prediction CSV: {paths.artifacts_root / 'eval_3fps_predictions.csv'}",
            f"- Top gate frames CSV: {paths.artifacts_root / 'eval_3fps_top20_gate.csv'}",
            "",
            "## 6. Notes",
            "- This pipeline was built from scratch in experiment2 and does not import any legacy training code.",
            "- If needed, you can increase epochs for stronger convergence once the strategy is validated.",
        ]
    )

    out_path = paths.reports_root / "EXPERIMENT2_REPORT.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Experiment2: offline augmentation + gate/classify training + 3fps eval")
    parser.add_argument("--experiment-root", type=Path, default=default_experiment_root)
    parser.add_argument("--gate-per-class", type=int, default=1000)
    parser.add_argument("--classify-per-class", type=int, default=400)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    paths = build_paths(args.experiment_root)
    paths.artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.reports_root.mkdir(parents=True, exist_ok=True)

    seed_inventory = build_seed_inventory(paths)

    gate_df, cls_df = generate_augmented_sets(
        paths=paths,
        seed_inventory=seed_inventory,
        gate_per_class=args.gate_per_class,
        cls_per_class=args.classify_per_class,
        image_size=args.image_size,
        seed=args.seed,
    )

    gate_train, gate_val = split_manifest(gate_df, val_ratio=args.val_ratio, seed=args.seed)
    cls_train, cls_val = split_manifest(cls_df, val_ratio=args.val_ratio, seed=args.seed)

    gate_train.to_csv(paths.artifacts_root / "gate_manifest_train.csv", index=False)
    gate_val.to_csv(paths.artifacts_root / "gate_manifest_val.csv", index=False)
    cls_train.to_csv(paths.artifacts_root / "classify_manifest_train.csv", index=False)
    cls_val.to_csv(paths.artifacts_root / "classify_manifest_val.csv", index=False)

    gate_metrics = train_one_task(
        task_name="gate",
        train_df=gate_train,
        val_df=gate_val,
        num_classes=2,
        paths=paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    cls_metrics = train_one_task(
        task_name="classify",
        train_df=cls_train,
        val_df=cls_val,
        num_classes=4,
        paths=paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    eval_df = evaluate_3fps(
        paths=paths,
        gate_ckpt=paths.artifacts_root / "gate_best.pt",
        cls_ckpt=paths.artifacts_root / "classify_best.pt",
        image_size=args.image_size,
        gate_threshold=args.gate_threshold,
    )

    write_report(
        paths=paths,
        gate_metrics=gate_metrics,
        cls_metrics=cls_metrics,
        eval_df=eval_df,
        gate_per_class=args.gate_per_class,
        cls_per_class=args.classify_per_class,
    )

    print("Experiment2 pipeline completed.")
    print(f"Report: {paths.reports_root / 'EXPERIMENT2_REPORT.md'}")


if __name__ == "__main__":
    main()
