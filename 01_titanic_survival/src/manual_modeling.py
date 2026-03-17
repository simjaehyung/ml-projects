import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_preprocessing import get_train_test_data

def evaluate_manual_models(X, y, test_size=0.2, random_state=42):
    """
    Trains and evaluates exactly 3 models: Logistic Regression, SVM, and Random Forest.
    Records their baseline accuracy.
    """
    print(f"\n--- Manual Modeling Validation (Test Size: {test_size}) ---")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Support Vector Machine (SVM)": SVC(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate Accuracy
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
    print("--- Manual Modeling Complete ---\n")
    return results

if __name__ == "__main__":
    X, y = get_train_test_data()
    evaluate_manual_models(X, y)
