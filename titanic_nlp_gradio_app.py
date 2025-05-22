# titanic-nlp-ai/titanic_nlp_gradio_app.py

import gradio as gr
import pickle
import pandas as pd
import numpy as np # numpy is often a dependency of pandas/sklearn
import spacy
import logging
import os

# --- Configuration ---
# These features are the raw inputs expected from the text before pipeline processing.
# They should match the FEATURES list from model_training.py (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
EXPECTED_INPUT_FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
MODEL_PIPELINE_PATH = "titanic_rf_pipeline.pkl" # Path to the saved pipeline

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables / Models (Load once) ---
NLP_MODEL = None
TITANIC_PIPELINE = None

def load_models():
    global NLP_MODEL, TITANIC_PIPELINE
    if NLP_MODEL is None:
        try:
            NLP_MODEL = spacy.load("en_core_web_sm")
            logging.info("SpaCy NLP model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logging.error("SpaCy model 'en_core_web_sm' not found. Please download it: python -m spacy download en_core_web_sm")
            raise RuntimeError("SpaCy model 'en_core_web_sm' not found. App cannot start.")

    if TITANIC_PIPELINE is None:
        if not os.path.exists(MODEL_PIPELINE_PATH):
            logging.error(f"Trained model pipeline not found at {MODEL_PIPELINE_PATH}")
            raise FileNotFoundError(f"Trained model pipeline not found at {MODEL_PIPELINE_PATH}. Ensure it's in the same directory or the path is correct.")
        try:
            with open(MODEL_PIPELINE_PATH, "rb") as f:
                TITANIC_PIPELINE = pickle.load(f)
            logging.info(f"Titanic survival pipeline loaded successfully from {MODEL_PIPELINE_PATH}.")
        except Exception as e:
            logging.error(f"Error loading Titanic survival pipeline: {e}")
            raise RuntimeError(f"Could not load the model pipeline: {e}")

try:
    load_models()
except Exception as e:
    logging.critical(f"Fatal error during model loading: {e}. The application might not work correctly.")
    # Depending on Gradio's behavior, this might still allow the UI to load but fail on interaction.

def extract_features_from_text(text: str) -> dict:
    """Extracts features from text for the Titanic model pipeline."""
    if NLP_MODEL is None:
        logging.error("NLP model not loaded. Cannot extract features.")
        # Return defaults that the pipeline's imputer can handle
        return {feat: None for feat in EXPECTED_INPUT_FEATURES}


    doc = NLP_MODEL(text.lower().strip())
    
    features = {
        'Pclass': None, 'Sex': None, 'Age': None, 'SibSp': 0, # Default SibSp/Parch to 0 if not found
        'Parch': 0, 'Fare': None, 'Embarked': None
    }
    
    # More robust extraction attempts
    age_found, fare_found, sex_found, pclass_found, embarked_found = False, False, False, False, False

    for token in doc:
        # Age
        if not age_found and token.like_num:
            try:
                window = doc[max(0, token.i - 2) : min(len(doc), token.i + 3)].text
                if "year" in window or "old" in window or "age" in window:
                    age_val = int(float(token.text))
                    if 0 < age_val < 120: features['Age'] = age_val; age_found = True
            except ValueError: pass
        
        # Fare
        if not fare_found and token.like_num:
            try:
                window = doc[max(0, token.i - 1) : min(len(doc), token.i + 3)].text
                if any(cur in window for cur in ["$", "dollar", "pound", "fare", "paid", "ticket cost"]):
                    fare_val = float(token.text)
                    if fare_val >= 0: features['Fare'] = fare_val; fare_found = True
            except ValueError: pass

        # Sex
        if not sex_found:
            if token.lemma_ in ['male', 'man', 'boy', 'gentleman', 'he', 'his']: features['Sex'] = 'male'; sex_found = True
            elif token.lemma_ in ['female', 'woman', 'girl', 'lady', 'she', 'her']: features['Sex'] = 'female'; sex_found = True
        
        # Pclass
        if not pclass_found:
            token_text_lower = token.text.lower()
            context_window = doc[max(0, token.i - 1) : min(len(doc), token.i + 2)].text.lower()
            if "class" in context_window or "ticket" in context_window:
                if '1st' in token_text_lower or 'first' in token_text_lower: features['Pclass'] = 1; pclass_found = True
                elif '2nd' in token_text_lower or 'second' in token_text_lower: features['Pclass'] = 2; pclass_found = True
                elif '3rd' in token_text_lower or 'third' in token_text_lower or 'steerage' in token_text_lower : features['Pclass'] = 3; pclass_found = True
        
        # Embarked
        if not embarked_found:
            # Check for full names or common abbreviations with context
            port_context = doc[max(0, token.i - 2) : min(len(doc), token.i + 3)].text.lower()
            if "embark" in port_context or "boarded" in port_context or "port" in port_context or "from" in port_context :
                if token.lemma_ in ['cherbourg'] or token.text.upper() == 'C': features['Embarked'] = 'C'; embarked_found = True
                elif token.lemma_ in ['queenstown'] or token.text.upper() == 'Q': features['Embarked'] = 'Q'; embarked_found = True
                elif token.lemma_ in ['southampton'] or token.text.upper() == 'S': features['Embarked'] = 'S'; embarked_found = True
    
    # Defaulting if still None after parsing (pipeline imputers will handle these Nones)
    if features['Sex'] is None: features['Sex'] = 'male' # A common default if not found
    if features['Pclass'] is None: features['Pclass'] = 3 # A common default
    if features['Embarked'] is None: features['Embarked'] = 'S' # Most common port

    logging.info(f"Raw extracted features: {features}")
    return features

def predict_survival(text: str):
    """Predicts survival based on input text using the loaded pipeline."""
    if TITANIC_PIPELINE is None or NLP_MODEL is None:
        error_msg = "Models not loaded. Please check server logs."
        logging.error(error_msg)
        return f"Error: {error_msg}\n\n🔍 Extracted Features: N/A\n📦 Model Input Data: N/A"

    try:
        raw_features_dict = extract_features_from_text(text)
        
        input_data = {}
        for feature_name in EXPECTED_INPUT_FEATURES:
            input_data[feature_name] = raw_features_dict.get(feature_name) 
            # Using .get without a default means it will be None if not in raw_features_dict
            # This is fine as pipeline's imputers will handle None.

        input_df = pd.DataFrame([input_data])
        # Ensure column order matches what the ColumnTransformer expects (though it selects by name)
        input_df = input_df[EXPECTED_INPUT_FEATURES]

        logging.info(f"DataFrame sent to pipeline: \n{input_df.to_string()}")

        prediction = TITANIC_PIPELINE.predict(input_df)[0]
        proba = TITANIC_PIPELINE.predict_proba(input_df)[0]

        # Determine index for "Survived" class (usually 1)
        # TITANIC_PIPELINE.classes_ shows the mapping, e.g., array([0, 1])
        # The second element of proba corresponds to the second class in TITANIC_PIPELINE.classes_
        try:
            survived_class_index = list(TITANIC_PIPELINE.classes_).index(1)
            survival_probability = proba[survived_class_index]
        except (ValueError, AttributeError, IndexError): # Fallback if classes_ attribute is not as expected or 1 not in classes
            logging.warning("Could not reliably determine 'Survived' class index. Using proba[1] if available.")
            survival_probability = proba[1] if len(proba) > 1 else proba[0]


        result_text = "✅ Likely Survived" if prediction == 1 else "❌ Likely Did Not Survive"
        
        output_message = (
            f"🧠 Prediction: {result_text}\n"
            f"📊 Probability of Survival: {survival_probability*100:.2f}%\n\n"
            f"🔍 Raw Extracted Features (before pipeline): {raw_features_dict}\n"
            f"📦 DataFrame Input to Pipeline (after defaults/None for missing): \n{input_df.to_string()}"
        )
        return output_message

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return f"An error occurred: {str(e)}\n\n🔍 Extracted Features: Error\n📦 Model Input Data: Error"

iface = gr.Interface(
    fn=predict_survival,
    inputs=gr.Textbox(lines=4, placeholder="Describe the passenger (e.g., 'A 30 year old male, 1st class, embarked at Southampton, paid 50 dollars for his ticket. He had 1 sibling.')..."),
    outputs=gr.Textbox(label="Survival Prediction Details", lines=10),
    title="🚢 Titanic Survival Predictor from Text",
    description="Enter a description of a Titanic passenger. The system will attempt to extract features and predict survival. Feature extraction is rule-based.",
    allow_flagging="never",
    examples=[
        ["25 year old man from Southampton in 3rd class, paid 10 dollars for his fare."],
        ["A 40 year old woman, first class, from Cherbourg. Fare was 100. No siblings or parents on board."],
        ["Child, female, age 8, 2nd class, embarked Queenstown, parch was 2 and sibsp 1."],
        ["He was a 60 year old male passenger in Pclass 1, paid a fare of 70 and embarked at C"],
        ["a lady of 22 years, paid 20 for her ticket"]
    ]
)

if __name__ == "__main__":
    # Model loading is attempted when the script is imported/run.
    # If load_models() failed, TITANIC_PIPELINE or NLP_MODEL might be None.
    # The predict_survival function handles this by returning an error message.
    if TITANIC_PIPELINE is None or NLP_MODEL is None:
         print("WARNING: Models did not load correctly. The Gradio app will start but predictions will fail. Check logs.")
    try:
        iface.launch()
    except Exception as e:
        logging.critical(f"Could not launch Gradio interface: {e}")
        print(f"Could not launch Gradio interface: {e}. Check logs for details.")