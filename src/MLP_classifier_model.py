import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data into a pandas DataFrame
df = pd.read_csv('../splits/train(visual).csv')

# Separate the target variable (y) and features (X)
y = df.iloc[:, 0]  # Assuming the target variable is in the first column
X = df.iloc[:, 1:]  # Assuming the features start from the second column

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))

# Create an MLPClassifier with two hidden layers, a smaller learning rate, and L2 regularization
model = MLPClassifier(
    hidden_layer_sizes=(6, 4),
    activation='logistic',
    learning_rate_init=0.5,
    max_iter=50000,
    solver='adam',
    alpha=0.0001,  # L2 regularization strength
    random_state=42
)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Evaluate accuracy, precision, recall, and F1 score on the validation set
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions)
val_recall = recall_score(y_val, val_predictions)
val_f1 = f1_score(y_val, val_predictions)

print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1)

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Evaluate accuracy, precision, recall, and F1 score on the test set
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

print("\nTest Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)


# confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a confusion matrix on the validation set predictions
cm = confusion_matrix(y_val, val_predictions)

# Normalize the confusion matrix
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create the labels
labels = ['0', '1']

# Plot the heatmap
fig = plt.figure(figsize=(4, 4))
sns.heatmap(cmn, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

# Create a confusion matrix on the test set predictions
cm = confusion_matrix(y_test, test_predictions)

# Normalize the confusion matrix
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create the labels
labels = ['0', '1']

# Plot the heatmap
fig = plt.figure(figsize=(4, 4))
sns.heatmap(cmn, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

# Get the predicted probabilities
train_probs = model.predict_proba(X_train)
test_probs = model.predict_proba(X_test)

# Keep only the positive class
train_probs = train_probs[:, 1]
test_probs = test_probs[:, 1]

# Calculate the ROC curve
train_fpr, train_tpr, thresholds = roc_curve(y_train, train_probs)
test_fpr, test_tpr, thresholds = roc_curve(y_test, test_probs)

# Plot the ROC curve
plt.plot(train_fpr, train_tpr, label="train")
plt.plot(test_fpr, test_tpr, label="test")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Calculate the ROC AUC
roc_auc_score(y_test, test_probs)


