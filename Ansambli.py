# Biblioteke
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Tacnost
def calculate_accuracy(model, X, y):
    predictions = model.predict(X)
    correct_predictions = (predictions == y).sum()
    total_samples = len(y)
    accuracy = correct_predictions / total_samples
    return accuracy

# Učitavanje podataka
data = pd.read_csv("data_2.csv", header=None)

# Broj primera i odlika
n = data.shape[1] - 1
m = data.shape[0]

# Matrica prediktora
X = data.iloc[:, 0:6].values

# Ciljna promenljiva
y = data.iloc[:, -1].values

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

# Definisanje listi za hiper-parametre koje želimo da analiziramo
ensemble_sizes = [10, 50, 100, 200]
learning_rates = [0.1, 0.05, 0.01]

# Analiza Random Forest algoritma
rf_scores = []
for size in ensemble_sizes:
    rf = RandomForestClassifier(n_estimators=size, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = calculate_accuracy(rf, X_test, y_test)
    rf_scores.append(accuracy)

# Crtanje grafika za Random Forest
plt.plot(ensemble_sizes, rf_scores)
plt.xlabel('Size of Ensemble')
plt.ylabel('Accuracy')
plt.title('Random Forest: Impact of Ensemble Size')
plt.show()

# Analiza Gradient Boosting algoritma
gb_scores = []
for rate in learning_rates:
    gb = GradientBoostingClassifier(learning_rate=rate, random_state=42)
    gb.fit(X_train, y_train)
    accuracy = calculate_accuracy(gb, X_test, y_test)
    gb_scores.append(accuracy)

# Crtanje grafika za Gradient Boosting
plt.plot(learning_rates, gb_scores)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting: Impact of Learning Rate')
plt.show()

# Prikaz značajnosti odlika za Gradient Boosting algoritam
gb = GradientBoostingClassifier(learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

feature_importance = gb.feature_importances_
num_predictors = X_train.shape[1] - 1  # Broj odlika
feature_names = [f"Odlika_{i}" for i in range(1, num_predictors + 1)]

sorted_indices = np.argsort(feature_importance)[::-1]

# Prikaz svih značajnih prediktora ako ih ima dovoljno
num_features_to_show = min(6, len(feature_importance))
if num_features_to_show > 0:
    sorted_indices = np.argsort(feature_importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_features_to_show), feature_importance[sorted_indices][:num_features_to_show], tick_label=[f"Odlika_{i+1}" for i in sorted_indices[:num_features_to_show]])
    plt.xlabel('Odlika')
    plt.ylabel('Importance')
    plt.title('Gradient Boosting: Feature Importance')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Nema dovoljno značajnih prediktora za prikaz.")

# Izračunavanje tačnosti na trening i test skupu koristeći funkciju calculate_accuracy
train_accuracy_rf = calculate_accuracy(rf, X_train, y_train)
test_accuracy_rf = calculate_accuracy(rf, X_test, y_test)

train_accuracy_gb = calculate_accuracy(gb, X_train, y_train)
test_accuracy_gb = calculate_accuracy(gb, X_test, y_test)

# Ispis tačnosti na trening i test skupu
print("Random Forest:")
print(f"Train Accuracy: {train_accuracy_rf * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy_rf * 100:.2f}%")

print("\nGradient Boosting:")
print(f"Train Accuracy: {train_accuracy_gb * 100}")
print(f"Test Accuracy: {test_accuracy_gb * 100}")
