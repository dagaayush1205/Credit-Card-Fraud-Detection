
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Load and scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # X = your features

# 2. Fit the Isolation Forest
iso_model = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42)
iso_model.fit(X_scaled)

# 3. Get anomaly scores (higher = more anomalous)
iso_score = -iso_model.decision_function(X_scaled)  # Flip sign so higher = more anomalous

# 4. Evaluate AUPRC
from sklearn.metrics import average_precision_score
print("📈 Isolation Forest AUPRC:", average_precision_score(y_true, iso_score))  # y_true = 0/1 fraud labels

