import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_preprocessing import get_train_test_data

def run_seed_test(X, y, model_class, seeds=[42, 100, 2026], test_size=0.2):
    """
    Runs the given model class 3 times with different random seeds.
    Calculates the accuracy for each run and the score variance.
    """
    print(f"\n--- Running Seed Test with seeds: {seeds} ---")
    
    accuracies = []
    
    for seed in seeds:
        print(f"Testing with seed: {seed}")
        # Use a fixed data split seed to only test model instability, 
        # but often it's more comprehensive to change both split and model random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        # Instantiate model with the specific seed
        model = model_class(random_state=seed)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Accuracy (seed={seed}): {acc:.4f}")
        
    variance = np.var(accuracies)
    mean_acc = np.mean(accuracies)
    
    print(f"Seed Test Results -> Mean: {mean_acc:.4f}, Variance: {variance:.6f}")
    print("--- Seed Test Complete ---\n")
    return mean_acc, variance

def run_split_test(X, y, model_class, splits=[0.2, 0.3], random_state=42):
    """
    Compares model results using two different train/test splits (e.g., 80/20 vs 70/30).
    splits argument takes the 'test_size' values.
    """
    print(f"\n--- Running Split Test with test_sizes: {splits} ---")
    
    results = {}
    
    for test_size in splits:
        print(f"Testing split ({1-test_size:.2f} / {test_size:.2f})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        model = model_class(random_state=random_state)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[f"Test_{test_size}"] = acc
        print(f"Accuracy (test_size={test_size}): {acc:.4f}")
        
    print("--- Split Test Complete ---\n")
    return results

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    X, y = get_train_test_data()
    # Assume RandomForest is best for sanity test
    run_seed_test(X, y, RandomForestClassifier)
    run_split_test(X, y, RandomForestClassifier)
