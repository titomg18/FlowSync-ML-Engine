import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Lokasi file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "transport_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "kmeans_transport.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Contoh dataset (kalau belum ada CSV)
data = pd.DataFrame({
    "travel_time": [10, 15, 20, 30, 45, 60, 90],
    "frequency":   [1, 2, 3, 4, 5, 6, 7],
    "cost":        [2000, 2500, 3000, 3500, 4500, 6000, 9000]
})

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Simpan model & scaler
with open(MODEL_PATH, "wb") as f:
    pickle.dump(kmeans, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model & scaler berhasil dilatih dan disimpan")
