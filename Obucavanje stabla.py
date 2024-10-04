# Obucavanje stabla
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib.colors import ListedColormap

# Učitavanje podataka
data = pd.read_csv("data_1.csv", header=None)
column_names = [f'odlika{i}' for i in range(1, 14)] + ['label']
data.columns = column_names

# Korelacija izmedju odlika
correlation_matrix = data.corr()

# Vizualizacija korelacije
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# Vizualizacija odabranih prediktora
sns.pairplot(data, vars=['odlika6', 'odlika7'], hue='label')
plt.show()

# Izbor para prediktora i ciljne promenljive
X = data[['odlika6', 'odlika7']]
y = data['label']

# Podela podataka na train i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treniranje klasifikacionog stabla sa različitim maksimalnim dubinama
depths = [1, 3, 5, 10, None]
models = []
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    models.append(clf)

# Evaluacija modela
for depth, model in zip(depths, models):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Maksimalna dubina stabla: {depth}, Tacnost: {accuracy * 100:.2f}%")

# Vizualizacija granica odlučivanja
x_min, x_max = X_train['odlika6'].min() - 1, X_train['odlika6'].max() + 1
y_min, y_max = X_train['odlika7'].min() - 1, X_train['odlika7'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# DataFrame za grid sa odgovarajućim imenima karakteristika
grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['odlika6', 'odlika7'])

f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))

for idx, depth, model in zip(range(len(depths)), depths, models):
    Z = model.predict(grid_df)  # Koristimo DataFrame sa imenima odlika za predikcije
    Z = Z.reshape(xx.shape)

    axarr[idx // 3, idx % 3].contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'blue')))
    axarr[idx // 3, idx % 3].scatter(X_train['odlika6'], X_train['odlika7'], c=y_train, edgecolors='k', marker='o', cmap=ListedColormap(('red', 'blue')))
    axarr[idx // 3, idx % 3].set_title(f'Max Depth: {depth}')

plt.show()
