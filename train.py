"""
Trains the classifier.
Loads dataset.csv, puts 20% of it aside for testing and trains a decision
tree on the rest to tell cars from pedestrians. I only let the tree go 3
levels deep so I can actually read the rules it comes up with. It prints
those rules, how well it does on the 20% it never saw, and a confusion
matrix, then saves the model to model.joblib so the app can use it.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

# The data from make_dataset.py
df = pd.read_csv("dataset.csv")

# X is what the model gets to look at, y is the answer it should give
X = df[["length", "width", "height", "num_points"]]
y = df["label"]

# Hide 20% away so we can check the model on stuff it's never seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Keep the tree small so the rules stay readable
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("Training examples:", len(X_train))
print("Test examples:", len(X_test))
print()

# The nice thing about a decision tree: you can just read what it learned
print("The rules the model learned:")
print(export_text(model, feature_names=list(X.columns)))

# Now see how it does on the held-back 20%
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Rows are what it really was, columns are what the model guessed (car, pedestrian)
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=["car", "pedestrian"]))

# Save it so the app can load it later
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")
