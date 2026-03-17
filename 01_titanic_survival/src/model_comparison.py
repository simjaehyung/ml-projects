"""
Titanic5 — 3-Model Comparison with Visual Report
=====================================================
모델: Logistic Regression / Random Forest / SVM
데이터: Titanic5 + 인간관계 파생 변수
출력: reports/figures/ 에 PNG 2장 저장
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── 색상 팔레트 ──────────────────────────────────────────────────
PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#DD8452",
    "SVM":                 "#55A868",
}
BG_COLOR   = "#F8F9FA"
GRID_COLOR = "#E9ECEF"


# ══════════════════════════════════════════════════════════════════
# 1. 데이터 파이프라인
# ══════════════════════════════════════════════════════════════════

def build_dataset():
    """Titanic5 다운로드 → 전처리 → 인간관계 특성 추가"""

    print("[1] Titanic5 loading...")
    try:
        df = pd.read_csv("https://hbiostat.org/data/repo/titanic3.csv")
    except Exception:
        # 로컬 fallback
        df = pd.read_csv("../data/raw_titanic.csv")

    # 컬럼명 정리
    df.columns = (df.columns
                  .str.lower()
                  .str.replace(".", "_", regex=False)
                  .str.strip())

    # ── Title 추출 ───────────────────────────────────────────────
    df["Title"] = df["name"].str.extract(r", (\w+)\.")
    df["Title"].replace({
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Countess": "Royalty", "Sir": "Royalty", "Lady": "Royalty",
        "Don": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
        "Dr": "Officer", "Rev": "Officer",
        "Col": "Officer", "Major": "Officer", "Capt": "Officer",
    }, inplace=True)

    # ── 누락값 처리 ──────────────────────────────────────────────
    df["age"]      = df.groupby("Title")["age"].transform(
                         lambda x: x.fillna(x.median()))
    df["age"]      = df["age"].fillna(df["age"].median())
    df["fare"]     = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # ── 가족 변수 ────────────────────────────────────────────────
    df["Surname"]    = df["name"].str.split(",").str[0].str.strip()
    df["FamilySize"] = df["sibsp"] + df["parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["FamilyCategory"] = pd.cut(
        df["FamilySize"], bins=[0, 1, 4, 20],
        labels=["Alone", "Small", "Large"])

    # ── 티켓 변수 ────────────────────────────────────────────────
    df["TicketGroupSize"] = df.groupby("ticket")["name"].transform("count")
    df["FarePerPerson"]   = df["fare"] / df["TicketGroupSize"]

    # NOTE: FamilySurvivalRate / TicketSurvivalRate 는 'survived' 컬럼을 참조하므로
    # 전체 데이터에서 계산하면 데이터 누출(data leakage)이 발생한다.
    # 여기서는 제외하고, 대신 SocialConnectionStrength 계산 시 0.5(불확실) 로 대체한다.

    # ── 가족 위치 점수 ───────────────────────────────────────────
    t_score = {"Master": 5, "Mrs": 4, "Miss": 3,
               "Royalty": 2, "Officer": 1, "Mr": -2}

    def _fps(row):
        s = t_score.get(row["Title"], 0)
        if pd.notna(row["age"]):
            if   row["age"] <  5: s += 3
            elif row["age"] < 13: s += 2
            elif row["age"] >= 65: s += 1
        if   row["FamilySize"] == 1: s -= 1
        elif row["FamilySize"] >= 5: s -= 2
        return s

    df["FamilyPositionScore"] = df.apply(_fps, axis=1)

    # SocialConnectionStrength: 생존율 항은 0.5(중립)로 대체
    df["SocialConnectionStrength"] = (
        df["FamilySize"]      * 0.3 +
        df["TicketGroupSize"] * 0.2 +
        0.5                   * 0.3 +   # FamilySurvivalRate 대체
        0.5                   * 0.2     # TicketSurvivalRate 대체
    )

    # ── 갑판 위치 ────────────────────────────────────────────────
    def _deck(row):
        if pd.notna(row.get("cabin")):
            return str(row["cabin"])[0]
        return {1: "B", 2: "E", 3: "G"}.get(row["pclass"], "G")

    df["DeckLevel"] = df.apply(_deck, axis=1).map(
        {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}).fillna(3)

    # ── 인코딩 ───────────────────────────────────────────────────
    df = pd.get_dummies(df,
        columns=["sex", "embarked", "Title", "FamilyCategory"],
        drop_first=True)

    # ── 정규화 ───────────────────────────────────────────────────
    scale_cols = ["age", "fare", "FarePerPerson", "FamilySize",
                  "FamilyPositionScore", "SocialConnectionStrength", "DeckLevel"]
    existing = [c for c in scale_cols if c in df.columns]
    df[existing] = StandardScaler().fit_transform(df[existing])

    # ── 최종 컬럼 선택 ───────────────────────────────────────────
    drop_raw = ["name", "ticket", "cabin", "boat", "body",
                "home_dest", "Surname", "passengerid"]
    df_model = df.drop(columns=[c for c in drop_raw if c in df.columns])
    df_model = df_model.dropna(subset=["survived"])

    X = df_model.drop(columns=["survived"])
    y = df_model["survived"].astype(int)

    print(f"   Dataset ready: {X.shape[0]} rows x {X.shape[1]} cols\n")
    return X, y


# ══════════════════════════════════════════════════════════════════
# 2. 모델 학습 및 평가
# ══════════════════════════════════════════════════════════════════

def train_and_evaluate(X, y, random_state=42):
    """5-Fold Stratified CV + 단일 테스트셋 평가"""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_split=5, random_state=random_state),
        "SVM": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    results   = {}   # 테스트셋 지표
    cv_scores = {}   # CV 분포
    roc_data  = {}   # ROC 곡선용

    for name, model in models.items():
        print(f"[Training] {name}...")

        # 5-fold CV
        cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_scores[name] = cv_acc

        # 테스트셋 평가
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_prob  = model.predict_proba(X_te)[:, 1]

        results[name] = {
            "Accuracy":  accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, zero_division=0),
            "Recall":    recall_score(y_te, y_pred, zero_division=0),
            "F1":        f1_score(y_te, y_pred, zero_division=0),
            "AUC-ROC":   roc_auc_score(y_te, y_prob),
            "cm":        confusion_matrix(y_te, y_pred),
            "y_pred":    y_pred,
            "y_prob":    y_prob,
        }

        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_data[name] = (fpr, tpr, results[name]["AUC-ROC"])

        print(f"   -> Acc={results[name]['Accuracy']:.3f}  "
              f"F1={results[name]['F1']:.3f}  "
              f"AUC={results[name]['AUC-ROC']:.3f}")

    # Feature importance (Random Forest)
    rf_model = models["Random Forest"]
    feat_imp = pd.Series(
        rf_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False).head(15)

    print()
    return results, cv_scores, roc_data, feat_imp, y_te


# ══════════════════════════════════════════════════════════════════
# 3. 시각화 — 대시보드 (Figure 1)
# ══════════════════════════════════════════════════════════════════

def plot_dashboard(results, cv_scores, roc_data, y_te, save_path):
    fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)
    fig.suptitle(
        "Titanic5 — 3-Model Comparison Dashboard",
        fontsize=20, fontweight="bold", y=0.98, color="#212529"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.92, bottom=0.07)

    model_names = list(results.keys())
    colors      = [PALETTE[n] for n in model_names]

    # ── (0,0) 지표 비교 막대 ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    x      = np.arange(len(metrics))
    width  = 0.22

    for i, (name, col) in enumerate(zip(model_names, colors)):
        vals = [results[name][m] for m in metrics]
        bars = ax0.bar(x + (i - 1) * width, vals,
                       width=width, label=name, color=col,
                       alpha=0.88, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax0.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.008,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=7.5, color="#333333")

    ax0.set_xticks(x)
    ax0.set_xticklabels(metrics, fontsize=9)
    ax0.set_ylim(0.5, 1.05)
    ax0.set_ylabel("Score", fontsize=10)
    ax0.set_title("① 성능 지표 비교", fontsize=12, fontweight="bold", pad=10)
    ax0.legend(fontsize=8, loc="lower right")
    ax0.set_facecolor(BG_COLOR)
    ax0.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax0.set_axisbelow(True)

    # ── (0,1) ROC 곡선 ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random Chance")

    for name, col in zip(model_names, colors):
        fpr, tpr, auc = roc_data[name]
        ax1.plot(fpr, tpr, lw=2.2, color=col,
                 label=f"{name}  (AUC = {auc:.3f})")
        ax1.fill_between(fpr, tpr, alpha=0.07, color=col)

    ax1.set_xlabel("False Positive Rate", fontsize=9)
    ax1.set_ylabel("True Positive Rate", fontsize=9)
    ax1.set_title("② ROC 곡선", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_facecolor(BG_COLOR)
    ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax1.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax1.set_axisbelow(True)

    # ── (0,2) CV 안정성 박스플롯 ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    cv_data  = [cv_scores[n] for n in model_names]
    bp = ax2.boxplot(cv_data, patch_artist=True, widths=0.45,
                     medianprops=dict(color="black", linewidth=2))

    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    for whisker in bp["whiskers"]:
        whisker.set(color="#888888", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="#888888", linewidth=1.2)

    # 각 모델의 평균값 표시
    for i, (name, col) in enumerate(zip(model_names, colors), 1):
        mean = np.mean(cv_scores[name])
        ax2.scatter(i, mean, color="white", edgecolor=col,
                    s=60, zorder=5, linewidths=2)
        ax2.text(i, mean + 0.005, f"{mean:.3f}",
                 ha="center", va="bottom", fontsize=8, color="#333333")

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8.5)
    ax2.set_ylabel("Accuracy (5-Fold CV)", fontsize=9)
    ax2.set_title("③ 교차 검증 안정성", fontsize=12, fontweight="bold", pad=10)
    ax2.set_facecolor(BG_COLOR)
    ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax2.set_axisbelow(True)

    # ── (1,0~2) 혼동 행렬 3개 ────────────────────────────────────
    label_map = {0: "사망", 1: "생존"}
    tick_labels = ["사망 (0)", "생존 (1)"]
    cmap_list = ["Blues", "Oranges", "Greens"]

    for j, (name, cmap) in enumerate(zip(model_names, cmap_list)):
        ax = fig.add_subplot(gs[1, j])
        cm = results[name]["cm"]
        total = cm.sum()

        # 퍼센트 어노테이션
        annot = np.array([[f"{v}\n({v/total*100:.1f}%)" for v in row] for row in cm])

        sns.heatmap(cm, annot=annot, fmt="", cmap=cmap,
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    linewidths=0.5, linecolor="#cccccc",
                    cbar=True, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})

        ax.set_xlabel("예측값", fontsize=9)
        ax.set_ylabel("실제값", fontsize=9)
        ax.set_title(
            f"④-{j+1}  {name}\n"
            f"Acc {results[name]['Accuracy']:.3f}  "
            f"F1 {results[name]['F1']:.3f}",
            fontsize=10.5, fontweight="bold", pad=8
        )
        ax.tick_params(labelsize=8.5)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    print(f"[Saved] Dashboard -> {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# 4. 시각화 — 특성 중요도 (Figure 2)
# ══════════════════════════════════════════════════════════════════

def plot_feature_importance(feat_imp, save_path):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # 색상: 인간관계 변수는 강조
    relationship_vars = {
        "FamilyPositionScore", "SocialConnectionStrength",
        "FamilySurvivalRate", "TicketSurvivalRate",
        "FamilySize", "IsAlone", "FarePerPerson",
        "TicketGroupSize", "DeckLevel"
    }
    bar_colors = [
        "#DD8452" if any(rv in c for rv in relationship_vars)
        else "#4C72B0"
        for c in feat_imp.index
    ]

    bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                   color=bar_colors[::-1],
                   alpha=0.88, edgecolor="white", linewidth=0.6)

    for bar, v in zip(bars, feat_imp.values[::-1]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9, color="#333333")

    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Random Forest — 특성 중요도 Top 15\n"
                 "■ 주황: 인간관계 파생 변수   ■ 파랑: 원본 변수",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DD8452", label="인간관계 파생 변수"),
        Patch(facecolor="#4C72B0", label="원본 변수"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    print(f"[Saved] Feature Importance -> {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# 5. 텍스트 결과 요약
# ══════════════════════════════════════════════════════════════════

def print_summary(results, cv_scores):
    print("\n" + "=" * 60)
    print("  Final Performance Summary")
    print("=" * 60)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    header = f"{'Model':<28}" + "".join(f"{m:>10}" for m in metrics)
    print(header)
    print("-" * 78)
    for name, res in results.items():
        row = f"{name:<28}" + "".join(f"{res[m]:>10.4f}" for m in metrics)
        print(row)

    print("\n5-Fold CV Mean +/- Std")
    print("-" * 40)
    for name, scores in cv_scores.items():
        print(f"  {name:<28} {scores.mean():.4f} +/- {scores.std():.4f}")

    best = max(results, key=lambda n: results[n]["AUC-ROC"])
    print(f"\n[Best AUC-ROC] {best}  ({results[best]['AUC-ROC']:.4f})")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 출력 폴더 생성
    fig_dir = os.path.join("..", "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. 데이터
    X, y = build_dataset()

    # 2. 학습 & 평가
    results, cv_scores, roc_data, feat_imp, y_te = train_and_evaluate(X, y)

    # 3. 시각화
    plot_dashboard(
        results, cv_scores, roc_data, y_te,
        save_path=os.path.join(fig_dir, "model_comparison_dashboard.png")
    )
    plot_feature_importance(
        feat_imp,
        save_path=os.path.join(fig_dir, "feature_importance.png")
    )

    # 4. 텍스트 요약
    print_summary(results, cv_scores)

    print("Done. Figures saved to reports/figures/")
