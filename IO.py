# Ucitavanje prvog skupa podataka

# Biblioteke
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Učitavanje podataka
data = pd.read_csv("data_1.csv", header=None)
print(data)

# Broj primera i odlika
n = data.shape[1] - 1
print("Broj odlika:", n)
m = data.shape[0]
print("Broj primera:", m)

# Matrica prediktora
X = data.iloc[:, 0:13].values
print("Prikaz prediktora\n", X)
print("Dimenzije matrice prediktora:", X.shape)

# Ciljna promenljiva
y = data.iloc[:, -1].values
print("Ciljna promenljiva\n", y)
print("Dimenzije vektora y:", y.shape)

# Dodavanje kolone sa jedinicama za računanje w i b zajedno
X = np.hstack((X, np.ones((X.shape[0], 1))))

# Podela obučavajućeg skupa na test i training skupove
split_index = int(0.7 * len(data))

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Ispis train i test promenljivih
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)

