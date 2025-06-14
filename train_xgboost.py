import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from utils.evaluate import plot_pr_curve

# Load dataset
df = pd.read_csv('data/creditcard.csv')
X = df.drop(['Class'], axis=1)
y = df['Class']

# Train/Test split (stratified due to imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(y == 0).sum() / (y == 1).sum())

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

# Grid search with 3-fold CV
grid = GridSearchCV(xgb, param_grid, scoring='average_precision', cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
print("✅ Best Parameters:", grid.best_params_)

# Evaluate
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auprc = average_precision_score(y_test, y_proba)
print(f"\n📈 AUPRC Score: {auprc:.4f}")

# Plot PR Curve
plot_pr_curve(y_test, y_proba, label='XGBoost')

