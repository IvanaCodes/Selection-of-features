# Biblioteke
from sklearn.linear_model import LogisticRegression
from IO import *

# Klasa FeatureSelector
class FeatureSelector:
    def __init__(self, data):
        # Konstruktor klase FeatureSelector koji inicijalizuje podatke, prediktore, ciljnu promenljivu,
        # skalirane prediktore, koeficijente korelacije i odabrane prediktore
        self.data = data
        self.X = self.data.iloc[:, :-1]  # Uzimanje svih kolona osim poslednje kao prediktora
        self.y = self.data.iloc[:, -1]   # Uzimanje poslednje kolone kao ciljne promenljive
        self.scaled_X = None             # Inicijalizacija skloiranih prediktora
        self.correlation_coeffs = None   # Inicijalizacija koeficijenata korelacije
        self.selected_features = []      # Inicijalizacija odabranih prediktora

# Metoda za skaliranje podataka prediktora
    def scale_data(self):
        self.scaled_X = self.X.apply(lambda x: (x - x.mean()) / x.std())
        return self.scaled_X

# Metoda za izračunavanje koeficijenta korelacije između dva niza
    def correlation_coef(self, x, y):
        return np.corrcoef(x, y)[0, 1]

# Metoda za izračunavanje koeficijenata korelacije između svakog prediktora i ciljne promenljive
    def calculate_correlations(self):
        self.correlation_coeffs = np.array([self.correlation_coef(self.scaled_X.iloc[:, i], self.y) for i in range(len(self.X.columns))])
        return self.correlation_coeffs

# Metoda za prikazivanje apsolutnih vrednosti koeficijenata korelacije na grafiku
    def plot_correlations(self):
        plt.bar(range(len(self.correlation_coeffs)), np.abs(self.correlation_coeffs))
        plt.xlabel('Prediktori')
        plt.ylabel('Koeficijenti korelacije (Apsolutne vrednosti)')
        plt.title('Koeficijenti korelacije prediktora')
        plt.show()

# Metoda za izračunavanje greške klasifikacije
    def error(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

# Metoda za odabir prediktora pomoću metode omotaca (wrapper method)
    def wrapper_method(self, no_of_features_to_keep=2, k=5):
        model = LogisticRegression(penalty=None)
        X = self.scaled_X.to_numpy()  # niz X sadrzi skalirane prediktore
        y = self.y.to_numpy().flatten()

        # Petlja za izbor prediktora koje treba zadrzati
        for _ in range(no_of_features_to_keep):
            errors = []  # Lista koja cuva greske za svaki prediktor
            # Petlja kroz sve prediktore
            for i in range(X.shape[1]):
                # selected_features je lista trenutno odabranih prediktora sa dodatim trenutnim prediktorom i
                selected_features = self.selected_features + [i] if self.selected_features else [i]
                X_current = X[:, selected_features] # sadrzi podatke samo za trenutno odabrane prediktore

                # K-struka unakrsna validacija
                fold_errors = []
                for j in range(k):
                    mask = np.ones(len(X), dtype=bool)  # mask se koristi za kreiranje trening i validacionih skupova
                    mask[j * len(X) // k:(j + 1) * len(X) // k] = False
                    # X_train i X_val sadrže podatke za trening i validaciju za trenutno odabrane prediktore
                    # Model predviđa ciljne promenljive za validacioni skup (X_val).
                    X_train = X_current[mask]
                    X_val = X_current[~mask]
                    y_train = y[mask]
                    y_val = y[~mask]

                    # Model se trenira na X_train i y_train
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    fold_errors.append(self.error(y_val, y_pred))

                # Greška se računa i dodaje u listu fold_errors
                errors.append(np.mean(fold_errors))

            min_error_index = np.argmin(errors)
            self.selected_features.append(min_error_index)
        # Funkcija vraća listu odabranih prediktora
        return self.selected_features

# Metoda za prikazivanje prosečnih vrednosti odabranih prediktora na grafiku
    def plot_selected_features(self):
        # Metoda za prikazivanje prosečnih vrednosti odabranih prediktora na grafiku
        mean_values = self.scaled_X.iloc[:, self.selected_features].mean()
        plt.bar([f'Prediktor {idx + 1}' for idx in self.selected_features], mean_values)
        plt.xlabel('Prediktori')
        plt.ylabel('Srednje vrednosti')
        plt.title('Srednje vrednosti odabranih prediktora')
        plt.show()

 # Metoda za dobijanje imena odabranih prediktora
    def get_selected_features_names(self):
        return [self.X.columns[idx] for idx in self.selected_features]

 # Metoda za izračunavanje tačnosti modela na dati skup podataka
    def calculate_accuracy(self, model, X, y):
        predictions = model.predict(X)
        correct_predictions = (predictions == y).sum()
        total_samples = len(y)
        accuracy = correct_predictions / total_samples
        return accuracy

# Instanca klase FeatureSelector
selector = FeatureSelector(data)

# Skaliranje podataka
selector.scale_data()

# Izračunavanje i prikazivanje koeficijenata korelacije
selector.calculate_correlations()
selector.plot_correlations()

# Odabir prediktora pomoću omotac metode
selector.wrapper_method()
selector.plot_selected_features()

# Deljenje skupa podataka
split_index = int(0.7 * len(selector.data))
X_train = selector.scaled_X.iloc[:split_index]
y_train = selector.y.iloc[:split_index]
X_test = selector.scaled_X.iloc[split_index:]
y_test = selector.y.iloc[split_index:]

# Ispisivanje trening i test podataka
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)

# Treniranje modela
model = LogisticRegression(penalty=None)
model.fit(X_train, y_train)

# Izračunavanje i ispisivanje tačnosti
train_accuracy = selector.calculate_accuracy(model, X_train, y_train)
test_accuracy = selector.calculate_accuracy(model, X_test, y_test)

print("Train Accuracy:", train_accuracy * 100)
print("Test Accuracy:", test_accuracy * 100 )

# Ispisivanje odabranih prediktora
selected_features_names = selector.get_selected_features_names()
print("Selected Features:", selected_features_names)
