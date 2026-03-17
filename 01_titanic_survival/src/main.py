import time
from data_preprocessing import get_train_test_data
from manual_modeling import evaluate_manual_models
from validation_tests import run_seed_test, run_split_test
from automl_modeling import run_automl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def generate_notion_report(manual_results, automl_pipeline, automl_score, seed_mean, seed_var, split_results, best_manual_name):
    report = f"""
# Titanic ML Final Report

## Models Performance Overview
**Manual Models:**
- Logistic Regression Baseline Accuracy: {manual_results.get('Logistic Regression', 0):.4f}
- SVM Baseline Accuracy: {manual_results.get('Support Vector Machine (SVM)', 0):.4f}
- Random Forest Baseline Accuracy: {manual_results.get('Random Forest', 0):.4f}

**AutoML (TPOT) Result:**
- Best Pipeline Score (CV=5, 10 min limit): {automl_score:.4f}
- Best Pipeline Steps: 
```python
{str(automl_pipeline.steps) if automl_pipeline else "AutoML Pipeline Generation Failed"}
```

## Stability & Validation Tests (Best Manual Model: {best_manual_name})
**Seed Test (Seeds: 42, 100, 2026):**
- Mean Accuracy: {seed_mean:.4f}
- Variance: {seed_var:.6f}

**Split Test (80/20 vs 70/30):**
- 80/20 Split Accuracy: {split_results.get('Test_0.2', 0):.4f}
- 70/30 Split Accuracy: {split_results.get('Test_0.3', 0):.4f}

## Core Questions
**Which model performed best?**
The model that performed best overall was {"AutoML" if automl_score > manual_results[best_manual_name] else best_manual_name} with an accuracy of {max(automl_score, manual_results[best_manual_name]):.4f}.

**Which was most stable?**
The Seed test showed a variance of {seed_var:.6f}, indicating {"low" if seed_var < 0.001 else "moderate" if seed_var < 0.01 else "high"} variance across random seeds. The difference between the 80/20 and 70/30 split was {abs(split_results.get('Test_0.2', 0) - split_results.get('Test_0.3', 0)):.4f}, showing that data splitting strategy clearly impacts performance. 

**Are they the same?**
Usually, the highest performing model (especially an ensemble like Random Forest or an AutoML pipeline) may also be quite stable, but models with very high complexity can risk overfitting, making them less stable on new data. 

**Can this be used in a real service? Why/Why not?**
Since this model operates on static, structured tabular data, it is excellent for offline analytics or batch prediction. However, to be used in a real web service, you would need to implement an API (using FastAPI or Flask) to serve the trained model artifact (e.g., via `pickle` or `joblib`) to handle real-time HTTP prediction requests. Additionally, real-world data would require ensuring robust handling of missing features online and monitoring for data drift.

## Educational Goal: 5 Complex Concepts Explained
1. **One-Hot Encoding**: A process used to turn categorical data (like 'Male'/'Female') into binary numbers (0s and 1s) because Machine Learning algorithms only understand numbers.
2. **Cross-Validation (CV)**: A technique to evaluate a model by splitting the data into multiple folds (e.g., 5-fold CV). The model trains on 4 folds and tests on the 1 remaining fold, repeating this 5 times to ensure the score isn't just a lucky coincidence.
3. **Random Seed (random_state)**: Computers generate 'pseudo-random' numbers using an initial seed value. Freezing the seed (e.g., 42) ensures you get the exact same random split every time for reproducibility.
4. **Data Normalization / Standard Scaling**: Adjusting numeric features (like Age or Fare) so they share a common scale (usually a mean of 0 and standard deviation of 1). This stops large numbers from dominating the model logic.
5. **Variance in Seed Tests**: Running the exact same algorithm code multiple times can lead to different results purely based on how data is randomly shuffled. Variance measures how "jumpy" or unstable those scores are.
"""
    with open("../reports/notion_report.md", "w", encoding="utf-8") as file:
        file.write(report)
    print("Report successfully saved to reports/notion_report.md")

if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Data Preprocessing
    X, y = get_train_test_data()
    
    # 2. Manual Modeling
    manual_results = evaluate_manual_models(X, y)
    
    # Find best manual model class
    best_manual_name = max(manual_results, key=manual_results.get)
    print(f"Best manual model determined to be: {best_manual_name}")
    
    model_mapping = {
        "Logistic Regression": LogisticRegression,
        "Support Vector Machine (SVM)": SVC,
        "Random Forest": RandomForestClassifier
    }
    best_model_class = model_mapping[best_manual_name]
    
    # 3. Validation Tests on Best Manual Model
    seed_mean, seed_var = run_seed_test(X, y, best_model_class)
    split_results = run_split_test(X, y, best_model_class)
    
    # 4. AutoML Implementation (10 min limit)
    try:
        automl_pipeline, automl_score = run_automl(X, y, time_limit_mins=10)
    except Exception as e:
        print(f"AutoML failed: {e}. Falling back to a dummy result for the report.")
        automl_pipeline = None
        automl_score = 0.0

    # 5. Final Report Preparation
    generate_notion_report(
        manual_results, automl_pipeline, automl_score, 
        seed_mean, seed_var, split_results, best_manual_name
    )
    
    end_time = time.time()
    print(f"Entire pipeline executed in {(end_time - start_time)/60:.2f} minutes.")
