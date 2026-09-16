import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TELCO_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "telco"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Fixture to set up the Telco Churn model
@pytest.fixture
def telco_churn_model():
    # Load the dataset
    data = pd.read_csv(TELCO_CSV)

    # Remove leading and trailing spaces in all columns
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert TotalCharges to numeric, coerce invalid values to NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Define categorical and numerical columns
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Remove 'TotalCharges' from numerical columns since it contains non-numeric data
    numerical_cols = ['tenure', 'MonthlyCharges']

    # Split the data into training and testing sets
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Define preprocessing steps for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Create a column transformer with separate transformers for numeric and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the classifier
    classifier = LogisticRegression(random_state=42)

    # Create a pipeline that includes preprocessing and classification
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', classifier)])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    return pipeline

# Rest of the test functions (accuracy, classification report, confusion matrix, etc.) remain the same

# Test accuracy of the model
def test_accuracy(telco_churn_model):
    # Get the test data
    data = pd.read_csv(TELCO_CSV)
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Predict with the model using the test data
    y_pred = telco_churn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.75  # Set your desired accuracy threshold

# Test classification report
def test_classification_report(telco_churn_model):
    # Get the test data
    data = pd.read_csv(TELCO_CSV)
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Predict with the model using the test data
    y_pred = telco_churn_model.predict(X_test)
    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(classification_rep)
    # You can perform further assertions or checks on the classification report if needed

# Test confusion matrix
def test_confusion_matrix(telco_churn_model):
    # Get the test data
    data = pd.read_csv(TELCO_CSV)
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Predict with the model using the test data
    y_pred = telco_churn_model.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion)

