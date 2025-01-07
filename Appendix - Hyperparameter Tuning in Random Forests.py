import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
''''
File that calculates the best Hyperparameter and trains the model. Takes approximately 3.3 hrs in a Macbook M2.
'''
# Function to read data from CSV files
def load_data(training_file, testing_file):
    # Load the training and testing datasets
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(testing_file)

    return train_data, test_data

# Function to preprocess data
def preprocess_data(data, transformer=None, is_train=True):

    income_column = 'income'
    if income_column not in data.columns:
        raise ValueError(f"The column '{income_column}' does not exist in the dataset.")

    if not is_train:
        data[income_column] = data[income_column].str.strip().str.replace('.', '')
        print("Adjusted 'income' for test data:", data[income_column].unique()) 

    # Identify and store categorical features excluding 'income'
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    if income_column in categorical_features:
        categorical_features.remove(income_column)
    print("Categorical features before encoding:", categorical_features)  # Debug statement

    # Apply ColumnTransformer only to categorical features excluding 'income'
    if transformer is None:
        transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough')
        X = transformer.fit_transform(data.drop(columns=[income_column]))
    else:
        X = transformer.transform(data.drop(columns=[income_column]))

    # Encoding 'income'
    y = LabelEncoder().fit_transform(data[income_column])
    print("Encoded 'income':", np.unique(y))

    return X, y, transformer

# Function to train the best Random Forest model using Grid Search Hyperparameter Tuning and SMOTE

def train_best_random_forest_with_smote(X, y):
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Apply Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_smote, y_smote)

    print(f'Best Hyperparameters: {grid_search.best_params_}')
    print(f'Best Training Accuracy: {grid_search.best_score_}')

    # Return the best model found by Grid Search
    return grid_search.best_estimator_

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

# Main execution function
def main():

    training_file = "./census-income.data.csv"
    testing_file = "./census-income.test.csv"
    train_data, test_data = load_data(training_file, testing_file)

    # Preprocess training data
    X_train, y_train, transformer = preprocess_data(train_data, is_train=True)

    # Train the best model using Grid Search and SMOTE
    best_model = train_best_random_forest_with_smote(X_train, y_train)

    # Preprocess testing data using the same transformer
    X_test, y_test, _ = preprocess_data(test_data, transformer, is_train=False)

    # Evaluate the test model
    accuracy, report = evaluate_model(best_model, X_test, y_test)
    print(f'Accuracy test: {accuracy}')
    print(f'Classification Report test:\n{report}')

    # Evaluate the training model
    accuracy_train, report_train = evaluate_model(best_model, X_train, y_train)
    print(f'Accuracy train: {accuracy_train}')
    print(f'Classification Report train:\n{report_train}')

if __name__ == '__main__':
    main()
