import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Separate features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest model
model = RandomForestClassifier(random_state=42)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

from sklearn.model_selection import cross_val_score

# Apply 5-Fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Print cross-validation results
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", cv_scores.mean())

from sklearn.model_selection import GridSearchCV

# Set up hyperparameters to try
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

# Set up GridSearch
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

# Train with different combinations
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
