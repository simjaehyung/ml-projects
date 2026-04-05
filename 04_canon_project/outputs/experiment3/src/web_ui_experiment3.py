from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st


def run_command(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def read_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return f"Not found: {path}"
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... (truncated)"
    return text


def main() -> None:
    st.set_page_config(page_title="Experiment3 Runner", layout="wide")
    st.title("Experiment3 Easy Runner")
    st.caption("EfficientNet-B gate + ResNet18 classifier")

    experiment_root = Path(__file__).resolve().parents[1]
    src_root = experiment_root / "src"
    run_script = src_root / "run_experiment3.py"
    sm_main = src_root / "canon_program" / "main.py"

    st.sidebar.header("Environment")
    python_exec = st.sidebar.text_input("Python executable", value=sys.executable)

    tab1, tab2 = st.tabs(["Direct Pipeline", "State Machine"])

    with tab1:
        st.subheader("Direct Pipeline Run")
        col1, col2, col3 = st.columns(3)
        with col1:
            gate_per_class = st.number_input("gate_per_class", min_value=1, value=40, step=1)
            classify_per_class = st.number_input("classify_per_class", min_value=1, value=20, step=1)
        with col2:
            train_ratio = st.slider("train_ratio", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
            epochs = st.number_input("epochs", min_value=1, value=1, step=1)
        with col3:
            batch_size = st.number_input("batch_size", min_value=1, value=16, step=1)
            gate_backbone = st.selectbox(
                "gate_backbone",
                options=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"],
                index=0,
            )

        run_direct = st.button("Run Direct Pipeline", type="primary")
        if run_direct:
            cmd = [
                python_exec,
                str(run_script),
                "--experiment-root",
                str(experiment_root),
                "--gate-per-class",
                str(gate_per_class),
                "--classify-per-class",
                str(classify_per_class),
                "--train-ratio",
                str(train_ratio),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--gate-backbone",
                gate_backbone,
            ]
            st.code(" ".join(cmd))
            with st.spinner("Running pipeline..."):
                code, out, err = run_command(cmd, cwd=experiment_root)
            st.write(f"Exit code: {code}")
            st.text_area("stdout", out, height=180)
            st.text_area("stderr", err, height=120)

    with tab2:
        st.subheader("State-Machine Run")
        config_name = st.selectbox("config", options=["smoke_program.json", "baseline_program.json"], index=0)
        config_path = experiment_root / "configs" / config_name
        if st.button("Run State-Machine"):
            cmd = [python_exec, str(sm_main), "--config", str(config_path)]
            st.code(" ".join(cmd))
            with st.spinner("Running state-machine..."):
                code, out, err = run_command(cmd, cwd=experiment_root)
            st.write(f"Exit code: {code}")
            st.text_area("stdout", out, height=180)
            st.text_area("stderr", err, height=120)

    st.divider()
    st.subheader("Latest Report Preview")
    report_path = experiment_root / "reports" / "EXPERIMENT3_REPORT.md"
    st.write(f"Report path: {report_path}")
    st.code(read_text(report_path), language="markdown")


if __name__ == "__main__":
    main()
