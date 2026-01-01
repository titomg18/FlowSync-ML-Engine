import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import os

df = pd.read_csv("data/flowsync_commuter_survey.csv")

df = df[['respondent_id', 'route', 'current_commute_min']]
df.rename(columns={'current_commute_min': 'travel_time'}, inplace=True)

np.random.seed(42)
df['frequency'] = np.random.randint(3, 7, size=len(df))
df['cost'] = df['travel_time'] * 100

X = df[['travel_time', 'frequency', 'cost']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.title("Elbow Method - Transport Usage Clustering")
plt.savefig("elbow.png")
plt.close()

# Train final model
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# SAVE MODEL (INI PENTING)
with open("kmeans_transport.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model & scaler berhasil disimpan")
