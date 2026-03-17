import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

from data_preprocessing import get_train_test_data

def run_automl(X, y, time_limit_mins=10, random_state=42):
    """
    Uses TPOT to automatically find the best machine learning pipeline.
    Uses 5-fold CV and a user-specified time limit (10 mins default).
    """
    print(f"\n--- Starting AutoML with TPOT ({time_limit_mins} mins limit) ---")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Initialize TPOT
    # We set max_time_mins instead of generations/population_size to directly enforce the 10 min limit
    # cv=5 for 5-fold cross-validation
    tpot = TPOTClassifier(
        max_time_mins=time_limit_mins,
        cv=5,
        random_state=random_state,
        n_jobs=1  # Prevent Dask hanging on Windows
    )
    
    print("Fitting TPOT (This will take up to the specified time limit...)")
    tpot.fit(X_train, y_train)
    
    # Evaluate on test set
    score = tpot.score(X_test, y_test)
    print(f"\nTPOT Test Accuracy: {score:.4f}")
    
    # Export best pipeline to a python script
    export_path = '../src/tpot_best_pipeline.py'
    try:
        tpot.export(export_path)
        print(f"Exported best pipeline to: {export_path}")
    except Exception as e:
        print(f"Warning: Failed to export pipeline: {e}")
    
    print("--- AutoML Complete ---\n")
    
    return tpot.fitted_pipeline_, score

if __name__ == "__main__":
    X, y = get_train_test_data()
    run_automl(X, y, time_limit_mins=2) # 2 mins just for sanity test when run directly
