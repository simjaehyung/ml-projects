# Titanic Machine Learning Assignment

This repository contains an end-to-end Machine Learning pipeline applied to the famous Titanic dataset. It covers data preprocessing, baseline model training, AutoML (using TPOT), and validation testing.

## Project Structure
- `data/`: Raw and processed Titanic datasets.
- `src/`: Modular Python scripts for preprocessing, manual modeling, AutoML, and validation.
- `과제_제출_보고서.md`: **제출용 보고서** (메인)
- `reports/`: 상세 보고서 (모델 비교, 데이터셋 구축)

## Setup Instructions
We recommend using [Miniconda](https://docs.anaconda.com/free/miniconda/) or Anaconda.

1. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate titanic_env
   ```
   *(Alternatively, use `pip install -r requirements.txt`)*

2. Run the pipeline:
   ```bash
   cd src
   python main.py
   ```
   This script will orchestrate data loading, preprocessing, model training, AutoML optimization (limited to 10 minutes), validation testing, and generate figures in `reports/figures/`.

## Features
- **Data Preprocessing**: Handles missing values (`Age` using median, `Embarked` using mode) and One-Hot Encoding for categorical features. Numeric features are Standard Scaled.
- **Manual Baseline Models**: Evaluates Logistic Regression, Support Vector Machine (SVC), and Random Forest.
- **AutoML (TPOT)**: Automatically searches for the best machine learning pipeline over 5 cross-validation folds.
- **Robustness Tests**: Evaluates the stability of the model using Seed Testing and Split Testing.
