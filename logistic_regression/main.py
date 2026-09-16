from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TELCO_CSV = REPO / "data" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = HERE / "models"


def load_and_preprocess_data(file_path):
    global categorical_cols
    global numerical_cols
    data = pd.read_csv(file_path)
    # Load the dataset
    # data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Remove leading and trailing spaces in all columns
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                        'PaymentMethod']
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Convert TotalCharges to numeric, coerce invalid values to NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Get some information of the data
    print('top 5 row of data\n', data.head())
    print('statistics of data\n', data.describe())  # Summary statistics
    # Missing Values
    print('check missing values\n', data.isnull().sum())

    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    print('Check missing values after filling\n', data.isnull().sum())

    return data

def train_and_evaluate_model(data, target_col='Churn', test_size=0.3, random_state=42):
    # Split the data into features and labels
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

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
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the classifier
    classifier = LogisticRegression(random_state=random_state)

    # Create a pipeline that includes preprocessing and classification
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier)])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    model = pipeline
    # Save the model to a file
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    # Calculate precision
    # Calculate precision for the 'Yes' class
    precision_yes = precision_score(y_test, y_pred, pos_label='Yes')

    # Calculate recall for the 'Yes' class
    recall_yes = recall_score(y_test, y_pred, pos_label='Yes')

    # Calculate F1-score for the 'Yes' class
    f1_yes = f1_score(y_test, y_pred, pos_label='Yes')

    # Calculate the confusion matrix
    confusion = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])

    # Print the evaluation metrics for the 'Yes' class
    print(f'classifier: {classifier}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision for "Yes" class: {precision_yes:.2f}')
    print(f'Recall for "Yes" class: {recall_yes:.2f}')
    print(f'F1-score for "Yes" class: {f1_yes:.2f}')
    print('Confusion Matrix:')
    print(confusion)
    print(f'report: {classification_rep}')

    # Make predictions on the subset of test data using the trained model
    X_test_subset = X_test.iloc[0:5]
    y_pred_subset = pipeline.predict(X_test_subset)
    print("Predicted values for the subset:")
    print(y_pred_subset)

    #  Visual representation of your model's performance (e.g., ROC curve, confusion matrix).
    label_mapping = {'No': 0, 'Yes': 1}
    y_test_binary = [label_mapping[label] for label in y_test]
    y_pred_binary = [label_mapping[label] for label in y_pred]
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve and area
    precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_binary)
    pr_auc = average_precision_score(y_test_binary, y_pred_binary)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    data = load_and_preprocess_data(TELCO_CSV)
    results = train_and_evaluate_model(data)
