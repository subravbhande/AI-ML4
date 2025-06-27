import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay
)
# Load the dataset
df = pd.read_csv("ADSI_Table_1A.2.csv")

# Create the binary target column
df["HighFatality"] = df["Total Traffic Accidents - Died"].apply(lambda x: 1 if x > 5000 else 0)

# Drop irrelevant columns
df = df.drop(columns=["Sl. No.", "State/UT/City"])  # Not numerical

# Define input features (X) and target (y)
X = df.drop(columns=["HighFatality"])
y = df["HighFatality"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Create and train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for ROC
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Precision, Recall, F1-score
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ROC AUC Score
roc_score = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_score)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title("ROC Curve")
plt.grid()
plt.show()
# Custom threshold
custom_threshold = 0.3
y_custom = (y_prob >= custom_threshold).astype(int)

print("Classification report at 0.3 threshold:\n")
print(classification_report(y_test, y_custom))

import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_importance = pd.Series(model.coef_[0], index=X.columns)
print("Feature Importance:\n", feature_importance.sort_values(ascending=False))
