from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from IO import *

class FeatureSelector:
    def __init__(self, data):
        self.data = data
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]
        self.scaled_X = None
        self.correlation_coeffs = None
        self.selected_features = []

    def scale_data(self):
        self.scaled_X = self.X.apply(lambda x: (x - x.mean()) / x.std())
        return self.scaled_X

    def correlation_coef(self, x, y):
        return np.corrcoef(x, y)[0, 1]

    def calculate_correlations(self):
        self.correlation_coeffs = np.array([self.correlation_coef(self.scaled_X.iloc[:, i], self.y) for i in range(len(self.X.columns))])
        return self.correlation_coeffs

    def plot_correlations(self):
        plt.bar(range(len(self.correlation_coeffs)), np.abs(self.correlation_coeffs))
        plt.xlabel('Features')
        plt.ylabel('Correlation Coefficients (Absolute Values)')
        plt.title('Feature Correlation Coefficients')
        plt.show()

    def error(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    def wrapper_method(self, no_of_features_to_keep=2, k=5):
        model = LogisticRegression(penalty=None)
        X = self.scaled_X.to_numpy()
        y = self.y.to_numpy().flatten()

        for _ in range(no_of_features_to_keep):
            errors = []
            for i in range(X.shape[1]):
                selected_features = self.selected_features + [i] if self.selected_features else [i]
                X_current = X[:, selected_features]

                fold_errors = []
                for j in range(k):
                    mask = np.ones(len(X), dtype=bool)
                    mask[j * len(X) // k:(j + 1) * len(X) // k] = False

                    X_train = X_current[mask]
                    X_val = X_current[~mask]
                    y_train = y[mask]
                    y_val = y[~mask]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    fold_errors.append(self.error(y_val, y_pred))

                errors.append(np.mean(fold_errors))

            min_error_index = np.argmin(errors)
            self.selected_features.append(min_error_index)

        return self.selected_features

    def plot_selected_features(self):
        mean_values = self.scaled_X.iloc[:, self.selected_features].mean()
        plt.bar([f'Predictor {idx + 1}' for idx in self.selected_features], mean_values)
        plt.xlabel('Predictors')
        plt.ylabel('Mean Values')
        plt.title('Mean Values of Selected Predictors')
        plt.show()

    def get_selected_features_names(self):
        return [self.X.columns[idx] for idx in self.selected_features]

    def calculate_accuracy(self, model, X, y):
        predictions = model.predict(X)
        correct_predictions = (predictions == y).sum()
        total_samples = len(y)
        accuracy = correct_predictions / total_samples
        return accuracy

# Usage
# Load your data into a DataFrame called 'data'
# Example: data = pd.read_csv("your_data.csv", header=None)
# Then create an instance of FeatureSelector using your data
selector = FeatureSelector(data)

# Scaling data
selector.scale_data()

# Calculate and plot correlations
selector.calculate_correlations()
selector.plot_correlations()

# Feature selection using wrapper method
selector.wrapper_method()
selector.plot_selected_features()

# Splitting the dataset
split_index = int(0.7 * len(selector.data))
X_train = selector.scaled_X.iloc[:split_index]
y_train = selector.y.iloc[:split_index]
X_test = selector.scaled_X.iloc[split_index:]
y_test = selector.y.iloc[split_index:]

# Print train and test variables
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)

# Train the model
model = LogisticRegression(penalty=None)
model.fit(X_train, y_train)

# Calculate and print accuracy
train_accuracy = selector.calculate_accuracy(model, X_train, y_train)
test_accuracy = selector.calculate_accuracy(model, X_test, y_test)

print("Train Accuracy:", train_accuracy * 100)
print("Test Accuracy:", test_accuracy * 100 )

# Print selected features
selected_features_names = selector.get_selected_features_names()
print("Selected Features:", selected_features_names)
