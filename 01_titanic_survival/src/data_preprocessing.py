import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_data(url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", data_dir="../data/"):
    """
    Downloads the Titanic dataset from a public URL and saves it locally.
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "raw_titanic.csv")
    
    if not os.path.exists(file_path):
        print("Downloading Titanic dataset...")
        df = pd.read_csv(url)
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print(f"Loading data from local file: {file_path}")
        df = pd.read_csv(file_path)
    
    return df

def preprocess_data(df):
    """
    Cleans the data and performs preprocessing.
    - Handling missing values
    - Encoding categorical variables
    - Feature scaling
    """
    print("\n--- Starting Data Preprocessing ---")
    print(f"Initial shape: {df.shape}")
    
    # 1. Drop columns that are mostly unique or not immediately useful for simple models
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_cleaned = df.drop(columns=drop_cols, errors='ignore')
    print(f"Dropped columns: {drop_cols} (Cabin has too many missing values)")
    
    # 2. Handle missing values
    # Age: fill with median
    age_median = df_cleaned['Age'].median()
    df_cleaned['Age'] = df_cleaned['Age'].fillna(age_median)
    print(f"Filled missing 'Age' values with median: {age_median}")
    
    # Embarked: fill with mode
    embarked_mode = df_cleaned['Embarked'].mode()[0]
    df_cleaned['Embarked'] = df_cleaned['Embarked'].fillna(embarked_mode)
    print(f"Filled missing 'Embarked' values with mode: '{embarked_mode}'")
    
    # Fare: fill missing with median (just in case)
    fare_median = df_cleaned['Fare'].median()
    df_cleaned['Fare'] = df_cleaned['Fare'].fillna(fare_median)
    
    # 3. Encode categorical variables using One-Hot Encoding
    categorical_cols = ['Sex', 'Embarked']
    df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
    print(f"One-Hot Encoded categorical attributes: {categorical_cols}")
    
    # 4. Feature scaling
    # We will scale 'Age' and 'Fare'
    scaler = StandardScaler()
    num_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    print("Scaled numerical features: Age, Fare, Pclass, SibSp, Parch using StandardScaler")
    
    print(f"Final shape: {df_encoded.shape}")
    print("--- Preprocessing Complete ---\n")
    
    return df_encoded

def get_train_test_data(data_dir="../data/"):
    """
    Convenience function to load and preprocess data, then return X and y.
    """
    df = load_data(data_dir=data_dir)
    df_processed = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop(columns=['Survived'])
    y = df_processed['Survived']
    
    return X, y

if __name__ == "__main__":
    X, y = get_train_test_data()
    print("X.head():")
    print(X.head())
