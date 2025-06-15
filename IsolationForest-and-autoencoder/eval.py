# Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score

# For comparison only on labels where Class is available
iso_labels = (iso_preds == -1).astype(int)
ae_labels = (mse > threshold).astype(int)

print("\n📊 Isolation Forest")
print("Precision:", precision_score(y, iso_labels))
print("Recall:", recall_score(y, iso_labels))
print("F1 Score:", f1_score(y, iso_labels))

print("\n📊 Autoencoder")
print("Precision:", precision_score(y, ae_labels))
print("Recall:", recall_score(y, ae_labels))
print("F1 Score:", f1_score(y, ae_labels))

