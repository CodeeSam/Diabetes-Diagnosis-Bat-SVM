import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import random

# 1. THE CLASS DEFINITION (This must be here!)
class BatAlgorithm:
    def __init__(self, X, y, n_bats=20, n_iter=30):
        self.X = X
        self.y = y
        self.n_bats = n_bats
        self.n_iter = n_iter
        self.n_features = X.shape[1]
        self.bats = [np.random.randint(0,2,self.n_features) for _ in range(n_bats)]
        self.fitness = np.zeros(n_bats)
        self.best_bat = None
        self.best_fitness = 0

    def evaluate(self, bat):
        selected_features = [i for i in range(self.n_features) if bat[i]==1]
        if len(selected_features)==0:
            return 0
        X_sel = self.X[:, selected_features]
        clf = SVC(kernel='rbf', gamma='scale')
        # Using 3-fold cross-validation to find the best feature set
        score = cross_val_score(clf, X_sel, self.y, cv=3).mean()
        return score

    def optimize(self):
        for i, bat in enumerate(self.bats):
            self.fitness[i] = self.evaluate(bat)
        self.best_bat = self.bats[np.argmax(self.fitness)]
        self.best_fitness = np.max(self.fitness)
        for t in range(self.n_iter):
            for i in range(self.n_bats):
                new_bat = self.bats[i].copy()
                flip = random.randint(0, self.n_features-1)
                new_bat[flip] = 1 - new_bat[flip]
                new_fitness = self.evaluate(new_bat)
                if new_fitness > self.fitness[i]:
                    self.bats[i] = new_bat
                    self.fitness[i] = new_fitness
                    if new_fitness > self.best_fitness:
                        self.best_bat = new_bat
                        self.best_fitness = new_fitness
        return self.best_bat

# 2. LOAD & PREPROCESS DATA
df = pd.read_csv("diabetes.csv")
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, df[col].mean())

X = df.drop("Outcome", axis=1)
y = df["Outcome"].values
feature_names = X.columns.tolist()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. RUN OPTIMIZATION
print("Optimizing features using Bat Algorithm... please wait.")
bat = BatAlgorithm(X_train, y_train, n_bats=20, n_iter=30)
best_bat_binary = bat.optimize()
selected_indices = [i for i, val in enumerate(best_bat_binary) if val == 1]
selected_feature_names = [feature_names[i] for i in selected_indices]

# 4. TRAIN FINAL MODEL WITH SELECTED FEATURES
X_train_sel = X_train[:, selected_indices]
clf = SVC(kernel='rbf', gamma='scale', probability=True)
clf.fit(X_train_sel, y_train)

# 5. SAVE ASSETS
model_assets = {
    "model": clf,
    "scaler": scaler,
    "selected_indices": selected_indices,
    "feature_names": feature_names,
    "selected_features": selected_feature_names
}

joblib.dump(model_assets, "diabetes_model_assets.pkl")
print("--- Training Complete ---")
print(f"Features Selected: {selected_feature_names}")
print("Saved as: diabetes_model_assets.pkl")