import sys
import pickle
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "kmeans_transport.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

travel_time = float(sys.argv[1])
frequency = float(sys.argv[2])
cost = float(sys.argv[3])

X = pd.DataFrame([{
    'travel_time': travel_time,
    'frequency': frequency,
    'cost': cost
}])

X_scaled = scaler.transform(X)
cluster = model.predict(X)

print(cluster[0])
