# titanic-nlp-ai/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import logging
import os

# --- Configuration ---
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_OUTPUT_PATH = "titanic_rf_pipeline.pkl" # Output path for the pipeline
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = 'Survived'
NUMERIC_FEATURES = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
CATEGORICAL_FEATURES = ['Sex', 'Embarked']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES # All features before transformation

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(url: str) -> pd.DataFrame:
    """Loads data from a URL."""
    logging.info(f"Loading data from URL: {url}...")
    try:
        df = pd.read_csv(url)
        logging.info("Data loaded successfully from URL.")
        if df.empty:
            logging.error("Downloaded data is empty.")
            raise ValueError("Downloaded data is empty.")
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"No data found at URL (pandas EmptyDataError): {url}")
        raise
    except ConnectionError as e:
        logging.error(f"Connection error while trying to load data from URL: {url}. Details: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from URL {url}: {e}")
        raise

def create_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Creates a ColumnTransformer for preprocessing."""
    logging.info("Creating preprocessor...")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # drop='first' helps with multicollinearity
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # In case other columns are present and needed later (not used here)
    )
    logging.info("Preprocessor created.")
    return preprocessor

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    """Trains the Random Forest model within a pipeline."""
    logging.info("Starting model training...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    logging.info("Model training completed.")
    return pipeline

def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, data_name: str = "Test"):
    """Evaluates the model and prints metrics."""
    logging.info(f"Evaluating model on {data_name} data...")
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"🎯 {data_name} Accuracy: {accuracy:.4f}")
    print(f"\n--- {data_name} Data Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    return accuracy

def save_pipeline(pipeline: Pipeline, file_path: str):
    """Saves the trained pipeline to a file."""
    logging.info(f"Saving pipeline to {file_path}...")
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir): # Ensure directory exists
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        with open(file_path, "wb") as f:
            pickle.dump(pipeline, f)
        logging.info(f"Pipeline saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving pipeline: {e}")
        raise

def main():
    """Main function to run the model training and evaluation process."""
    logging.info("--- Starting Model Training Pipeline ---")

    try:
        df = load_data(DATA_URL)
    except Exception as e:
        logging.critical(f"Failed to load data. Exiting. Error: {e}")
        return

    if df[TARGET_COLUMN].isnull().any():
        logging.warning(f"Target column '{TARGET_COLUMN}' contains missing values. Dropping these rows.")
        df.dropna(subset=[TARGET_COLUMN], inplace=True)

    # Ensure all defined features are present
    # Handle case variations for 'Sex' and 'Embarked' which are common in datasets
    if 'sex' in df.columns and 'Sex' not in df.columns: df.rename(columns={'sex':'Sex'}, inplace=True)
    if 'embarked' in df.columns and 'Embarked' not in df.columns: df.rename(columns={'embarked':'Embarked'}, inplace=True)

    missing_model_features = [f for f in FEATURES if f not in df.columns]
    if missing_model_features:
        logging.error(f"Required features for modeling missing from DataFrame: {missing_model_features}")
        raise ValueError(f"DataFrame is missing required features: {missing_model_features}")

    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    if len(y.unique()) < 2:
        logging.error(f"Target variable '{TARGET_COLUMN}' has less than 2 unique values. Cannot train classifier.")
        return

    stratify_on = y
    if y.value_counts().min() < 2: # Stratification needs at least 2 samples per class for most splitters
        logging.warning("A class in the target variable has < 2 samples. Disabling stratification.")
        stratify_on = None
        
    logging.info(f"Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_on
        )
    except ValueError as e: # Fallback if stratification fails despite check
        logging.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
    logging.info(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    preprocessor = create_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    try:
        trained_pipeline = train_model(X_train, y_train, preprocessor)
    except Exception as e:
        logging.error(f"Failed during model training: {e}")
        return

    evaluate_model(trained_pipeline, X_train, y_train, data_name="Training")
    evaluate_model(trained_pipeline, X_test, y_test, data_name="Testing")

    try:
        save_pipeline(trained_pipeline, MODEL_OUTPUT_PATH)
    except Exception:
        logging.error(f"Failed to save the pipeline using path: {MODEL_OUTPUT_PATH}")

    logging.info("--- Model Training Pipeline Finished ---")

if __name__ == "__main__":
    main()