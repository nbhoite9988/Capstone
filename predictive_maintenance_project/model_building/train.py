# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# for mlflow ui hosting
import subprocess
import mlflow

# Set the tracking URL for MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("engine-prediction-maintenance-training-experiment")

# Initialize Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Store the repo and branch details 
repo = "nbhoite9988/predictive-maintenance"
storage_options = {"revision": "main"}

# Load datasets from Hugging Face Hub
Xtrain_path = "hf://datasets/nbhoite9988/predictive-maintenance/Xtrain.csv"
Xtest_path = "hf://datasets/nbhoite9988/predictive-maintenance/Xtest.csv"
ytrain_path = "hf://datasets/nbhoite9988/predictive-maintenance/ytrain.csv"
ytest_path = "hf://datasets/nbhoite9988/predictive-maintenance/ytest.csv"

# Load the datasets
Xtrain = pd.read_csv(Xtrain_path, storage_options=storage_options)
Xtest = pd.read_csv(Xtest_path, storage_options=storage_options)
ytrain = pd.read_csv(ytrain_path, storage_options=storage_options)
ytest = pd.read_csv(ytest_path, storage_options=storage_options)

# Get numeric and categorical features
numeric_features = Xtrain.columns.tolist()  # As all features are numeric in this dataset, we can directly take all columns as numeric features
categorical_features = []  # No categorical features in this dataset


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    ('passthrough', numeric_features)
)

# Define base model: Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=1)

# Define the parameter grid
param_grid = {
    'gradientboostingclassifier__n_estimators': [100, 200, 300],
    'gradientboostingclassifier__learning_rate': [0.01, 0.05, 0.1],
    'gradientboostingclassifier__max_depth': [2, 3, 4],
    'gradientboostingclassifier__min_samples_split': [2, 5, 10],
    'gradientboostingclassifier__min_samples_leaf': [1, 2, 4],
    'gradientboostingclassifier__subsample': [0.8, 1.0]
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, gb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    rs = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_grid,
        n_iter=50,
        scoring='recall',     # Key: optimize recall!
        cv=5,
        n_jobs=-1,
        random_state=1
    )
    rs.fit(Xtrain, ytrain.values.ravel()) #using .values.ravel() to convert to 1-D array to fix dataconversion warning

    # Log all parameter combinations and their mean test scores
    results = rs.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(rs.best_params_)

    # Store and evaluate the best model
    best_model = rs.best_estimator_
    
    #Adjusting the classification threshold to optimize recall and minimize false negatives.
    classification_threshold = 0.5 

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_predictive_maintenance_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "nbhoite9988/predictive-maintenance"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Step 2: Upload the model file to the repository
    api.upload_file(
        path_or_fileobj="best_predictive_maintenance_model_v1.joblib",
        path_in_repo="best_predictive_maintenance_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
