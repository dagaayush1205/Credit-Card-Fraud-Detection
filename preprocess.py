import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess(filepath="data/creditcard.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna()

    # Features and Labels
    X = df.drop(['Class'], axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y

if __name__ == "__main__":
    X, X_scaled, y = preprocess()
    print("✅ Data preprocessed.")
