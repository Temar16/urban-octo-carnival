import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis", "Mean Radius", "Mean Texture", "Mean Perimeter", 
                "Mean Area", "Mean Smoothness", "Mean Compactness", "Mean Concavity", 
                "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension", 
                "SE Radius", "SE Texture", "SE Perimeter", "SE Area", "SE Smoothness", 
                "SE Compactness", "SE Concavity", "SE Concave Points", "SE Symmetry", 
                "SE Fractal Dimension", "Worst Radius", "Worst Texture", "Worst Perimeter", 
                "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity", 
                "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"]
data = pd.read_csv(url, names=column_names)

# Preprocessing
X = data.drop(["ID", "Diagnosis"], axis=1)  # Features
y = data["Diagnosis"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Splitting data
scaler = StandardScaler()  # Standardizing features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)  # Training
y_pred = knn_classifier.predict(X_test_scaled)  # Predictions

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis", "Mean Radius", "Mean Texture", "Mean Perimeter", 
                "Mean Area", "Mean Smoothness", "Mean Compactness", "Mean Concavity", 
                "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension", 
                "SE Radius", "SE Texture", "SE Perimeter", "SE Area", "SE Smoothness", 
                "SE Compactness", "SE Concavity", "SE Concave Points", "SE Symmetry", 
                "SE Fractal Dimension", "Worst Radius", "Worst Texture", "Worst Perimeter", 
                "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity", 
                "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"]
data = pd.read_csv(url, names=column_names)

# Preprocessing
X = data.drop(["ID", "Diagnosis"], axis=1)  # Features
y = data["Diagnosis"]  # Target variable

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Classifier with k-fold cross-validation
knn_classifier = KNeighborsClassifier(n_neighbors=5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn_classifier, X_train_scaled, y_train, cv=kfold)

# Report cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())

# Training the classifier on the entire training set
knn_classifier.fit(X_train_scaled, y_train)

# Evaluating on the test set
y_pred = knn_classifier.predict(X_test_scaled)
print("\nTest Set Performance:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis", "Mean Radius", "Mean Texture", "Mean Perimeter", 
                "Mean Area", "Mean Smoothness", "Mean Compactness", "Mean Concavity", 
                "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension", 
                "SE Radius", "SE Texture", "SE Perimeter", "SE Area", "SE Smoothness", 
                "SE Compactness", "SE Concavity", "SE Concave Points", "SE Symmetry", 
                "SE Fractal Dimension", "Worst Radius", "Worst Texture", "Worst Perimeter", 
                "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity", 
                "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"]
data = pd.read_csv(url, names=column_names)

# Preprocessing
X = data.drop(["ID", "Diagnosis"], axis=1)  # Features
y = data["Diagnosis"]  # Target variable

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define KNN Classifier
knn_classifier = KNeighborsClassifier()

# Define hyperparameters grid
param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Best parameters found
print("Best parameters found:")
print(grid_search.best_params_)

# Evaluate the classifier on the test set using best parameters
best_knn_classifier = grid_search.best_estimator_
y_pred = best_knn_classifier.predict(X_test_scaled)

# Report performance metrics
print("\nTest Set Performance:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


