# -*- coding: utf-8 -*-
"""KNN Algorithm K-fold cross validation (balanced data) - Data Mining project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EbDyryA5zN29jRwePgbFOOWP28wfIPhO
"""

#Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Load the data:
train_data_KNN = pd.read_csv("census-income.data.csv")
test_data_KNN = pd.read_csv("census-income.test.csv")

# Trim spaces and remove periods from the target column to standardize labels
train_data_KNN['income'] = train_data_KNN['income'].str.strip().str.replace('.', '')
test_data_KNN['income'] = test_data_KNN['income'].str.strip().str.replace('.', '')

# Changing income values to binary values
train_data_KNN['income'] = train_data_KNN['income'].replace({"<=50K": 0, ">50K": 1})
test_data_KNN['income'] = test_data_KNN['income'].replace({"<=50K": 0, ">50K": 1})

# Replacing the " ?" to "?" in the columns: 'work-class', 'occupation', 'native-country'
columns_to_replace = ['work-class', 'occupation', 'native-country']
for column in columns_to_replace:
    train_data_KNN[column] = train_data_KNN[column].replace(' ?', '?')
    test_data_KNN[column] = test_data_KNN[column].replace(' ?', '?')

# Concatenate train and test data
combined_data = pd.concat([train_data_KNN, test_data_KNN])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# List of categorical columns
categorical_columns = ['work-class', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country']

# Apply label encoding to combined data
for column in categorical_columns:
    combined_data[column] = label_encoder.fit_transform(combined_data[column])

# Split the combined data back into train and test
train_data_KNN_encoded = combined_data[:len(train_data_KNN)]
test_data_KNN_encoded = combined_data[len(train_data_KNN):]

# Extract features and target variables
X_train = train_data_KNN_encoded.drop(columns=['income'])
y_train = train_data_KNN_encoded['income']
X_test = test_data_KNN_encoded.drop(columns=['income'])
y_test = test_data_KNN_encoded['income']

# Apply Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Define the range of k values to try
k_range = range(1, 51)

# Initialize lists to store mean accuracies for each k
mean_accuracies = []

# Perform 10-fold cross-validation for each value of k
for k in k_range:
    # Initialize the KNN classifier with the current value of k
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = cross_val_score(knn_classifier, X_train_resampled, y_train_resampled, cv=kf, scoring='accuracy')

    # Calculate the mean accuracy across all folds
    mean_accuracy = np.mean(accuracies)

    # Append the mean accuracy to the list
    mean_accuracies.append(mean_accuracy)

# Get the index of the highest mean accuracy
best_index = np.argmax(mean_accuracies)

# Get the best k value
best_k = k_range[best_index]

# Initialize the KNN classifier with the best k value
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)

# Train the classifier on the entire resampled training data
best_knn_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the training data using the best classifier
y_pred_train = best_knn_classifier.predict(X_train_resampled)

# Calculate the accuracy on the training data
accuracy_train = accuracy_score(y_train_resampled, y_pred_train)

# Calculate confusion matrix for the training data
conf_matrix_train = confusion_matrix(y_train_resampled, y_pred_train)

# Print the best k value, its corresponding mean accuracy, and the training accuracy
print(f"Best k value: {best_k}, Mean Accuracy: {mean_accuracies[best_index]:.4f}, Training Accuracy: {accuracy_train:.4f}")

# Print confusion matrix for the training data
print("\nConfusion Matrix (Training Data):")
print(conf_matrix_train)

# Print classification report for the training data
print("\nClassification Report (Training Data):")
print(classification_report(y_train_resampled, y_pred_train, target_names=["<=50K", ">50K"]))

# Make predictions on the test data using the best classifier
y_pred_best = best_knn_classifier.predict(X_test_scaled)

# Map predicted labels to their corresponding income categories
predicted_income = ["<=50K" if label == 0 else ">50K" for label in y_pred_best]

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)

# Calculate precision, recall, and F1-Score
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)

# Print the best k value, its corresponding mean accuracy, and the test accuracy
print(f"\nBest k value: {best_k}, Mean Accuracy: {mean_accuracies[best_index]:.4f}, Test Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=["<=50K", ">50K"]))

# Calculate the total number of predictions
total_predictions = len(predicted_income)

# Count the predictions for each income category
predictions_count = {"<=50K": predicted_income.count("<=50K"), ">50K": predicted_income.count(">50K")}

# Print predictions count with percentages
print("\nPredictions Count:")
for income_category, count in predictions_count.items():
    percentage = (count / total_predictions) * 100
    print(f"{income_category}: {count} ({percentage:.2f}%)")