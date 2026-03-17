
# Titanic ML Final Report

## Models Performance Overview
**Manual Models:**
- Logistic Regression Baseline Accuracy: 0.8045
- SVM Baseline Accuracy: 0.8324
- Random Forest Baseline Accuracy: 0.8156

**AutoML (TPOT) Result:**
- Best Pipeline Score (CV=5, 10 min limit): 0.0000
- Best Pipeline Steps: 
```python
AutoML Pipeline Generation Failed
```

## Stability & Validation Tests (Best Manual Model: Support Vector Machine (SVM))
**Seed Test (Seeds: 42, 100, 2026):**
- Mean Accuracy: 0.8361
- Variance: 0.000090

**Split Test (80/20 vs 70/30):**
- 80/20 Split Accuracy: 0.8324
- 70/30 Split Accuracy: 0.8246

## Core Questions
**Which model performed best?**
The model that performed best overall was Support Vector Machine (SVM) with an accuracy of 0.8324.

**Which was most stable?**
The Seed test showed a variance of 0.000090, indicating low variance across random seeds. The difference between the 80/20 and 70/30 split was 0.0078, showing that data splitting strategy clearly impacts performance. 

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
