import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import save_model
import joblib
from preprocess import preprocess
import os

X, X_scaled, y = preprocess()

# ---- Isolation Forest ----
isoforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
isoforest.fit(X_scaled)
joblib.dump(isoforest, "models/isolation_forest.pkl")
print("✅ Isolation Forest trained and saved.")

# ---- Autoencoder ----
input_dim = X_scaled.shape[1]
inputs = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(inputs)
decoded = Dense(input_dim, activation="linear")(encoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=64, shuffle=True, validation_split=0.1)
autoencoder.save("models/autoencoder.h5")
print("✅ Autoencoder trained and saved.")

