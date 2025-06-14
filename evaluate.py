from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# Isolation Forest metrics
iso_score = -isoforest.decision_function(X_scaled)
iso_scores = -iso_scores  # Flip sign for anomaly scores (higher = more anomalous)
iso_auprc = average_precision_score(y, iso_scores)
print(f"📈 Isolation Forest AUPRC: {iso_auprc:.4f}")

# Autoencoder metrics
ae_auprc = average_precision_score(y, mse)
print(f"📈 Autoencoder AUPRC: {ae_auprc:.4f}")

# Plot PR curve
def plot_pr_curve(y_true, scores, label):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.plot(recall, precision, label=f"{label} (AUPRC={average_precision_score(y_true, scores):.4f})")

plt.figure(figsize=(8, 6))
plot_pr_curve(y, iso_scores, "Isolation Forest")
plot_pr_curve(y, mse, "Autoencoder")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

