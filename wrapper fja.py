def wrapper_method(self, no_of_features_to_keep=2, k=5):
    model = LogisticRegression(penalty=None)  # Inicijalizacija modela logističke regresije bez penalizacije
    X = self.scaled_X.to_numpy()  # Konverzija skaliranih prediktora u numpy niz
    y = self.y.to_numpy().flatten()  # Konverzija ciljne promenljive u numpy niz i flattenovanje

    for _ in range(no_of_features_to_keep):  # Petlja za odabir broja prediktora koji treba zadržati
        errors = []  # Lista za skladištenje grešaka za svaki prediktor
        for i in range(X.shape[1]):  # Iteracija kroz sve prediktore
            selected_features = self.selected_features + [i] if self.selected_features else [i]  # Odabir trenutnog prediktora
            X_current = X[:, selected_features]  # Izbor podataka za trenutne odabrane prediktore

            fold_errors = []  # Lista za skladištenje grešaka za svaku iteraciju k-struke unakrsne validacije
            for j in range(k):  # Petlja za k-struku unakrsnu validaciju
                mask = np.ones(len(X), dtype=bool)  # Kreiranje maske sa sve `True` vrednostima
                mask[j * len(X) // k:(j + 1) * len(X) // k] = False  # Postavljanje validacionog dela na `False`

                X_train = X_current[mask]  # Kreiranje trening skupa
                X_val = X_current[~mask]  # Kreiranje validacionog skupa
                y_train = y[mask]  # Kreiranje ciljne promenljive za trening skup
                y_val = y[~mask]  # Kreiranje ciljne promenljive za validacioni skup

                model.fit(X_train, y_train)  # Treniranje modela
                y_pred = model.predict(X_val)  # Predikcija na validacionom skupu
                fold_errors.append(self.error(y_val, y_pred))  # Računanje i skladištenje greške

            errors.append(np.mean(fold_errors))  # Skladištenje prosečne greške za trenutni prediktor

        min_error_index = np.argmin(errors)  # Pronalaženje indeksa prediktora sa najmanjom greškom
        self.selected_features.append(min_error_index)  # Dodavanje prediktora sa najmanjom greškom u listu odabranih prediktora

    return self.selected_features  # Vraćanje liste odabranih predikt
