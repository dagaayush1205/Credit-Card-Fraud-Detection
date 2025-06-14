import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from xgboost import XGBClassifier
from utils.evaluate import plot_pr_curve

# Load data
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# --- 1. 80% Train, 10% Val, 10% Test ---
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42)
# (0.1111 of 90% ≈ 10%, rest is 80%)

print(f"📦 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# --- 2. Train XGBoost with Best Parameters ---
best_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)

best_model.fit(X_train, y_train)

# --- 3. Evaluate on Validation Set ---
val_pred = best_model.predict(X_val)
val_proba = best_model.predict_proba(X_val)[:, 1]

print("\n🔍 Validation Report:")
print(classification_report(y_val, val_pred, digits=4))
print("\n📉 Validation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print(f"\n📈 Validation AUPRC: {average_precision_score(y_val, val_proba):.4f}")
plot_pr_curve(y_val, val_proba, label='XGBoost (Validation)')

# --- 4. Final Test Evaluation ---
test_pred = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)[:, 1]

print("\n🔍 Test Report:")
print(classification_report(y_test, test_pred, digits=4))
print("\n📉 Test Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print(f"\n📈 Test AUPRC: {average_precision_score(y_test, test_proba):.4f}")
plot_pr_curve(y_test, test_proba, label='XGBoost (Test)')

