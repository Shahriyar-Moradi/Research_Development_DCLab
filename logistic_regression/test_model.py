import unittest
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TestTelcoChurnModel(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        telco_csv = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "telco"
            / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
        self.data = pd.read_csv(telco_csv)

        # Remove leading and trailing spaces in all columns
        self.data = self.data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Convert TotalCharges to numeric, coerce invalid values to NaN
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')

        # Define categorical and numerical columns
        self.categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        self.numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        # Split the data into training and testing sets
        self.X = self.data.drop('Churn', axis=1)
        self.y = self.data['Churn']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42, stratify=self.y)

        # Define preprocessing steps for numerical and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        # Combine transformers using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])

        # Define the classifier
        self.classifier = LogisticRegression(random_state=42)

        # Create a pipeline that includes preprocessing and classification
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                         ('classifier', self.classifier)])

        # Fit the pipeline on the training data
        self.pipeline.fit(self.X_train, self.y_train)

    def test_accuracy(self):
        # Predict and evaluate
        y_pred = self.pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.75)  # Adjust the threshold as needed

    def test_confusion_matrix(self):
        # Predict and evaluate
        y_pred = self.pipeline.predict(self.X_test)
        confusion = confusion_matrix(self.y_test, y_pred)
        self.assertTrue(confusion.shape == (2, 2))

    def test_classification_report(self):
        # Predict and evaluate
        y_pred = self.pipeline.predict(self.X_test)
        classification_rep = classification_report(self.y_test, y_pred)
        self.assertFalse(classification_rep == '')

if __name__ == '__main__':
    unittest.main()
