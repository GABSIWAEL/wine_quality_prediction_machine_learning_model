from pathlib import Path  # Provides filesystem path utilities for locating the saved model.

import joblib  # Loads the trained Random Forest model from its serialized file.
import numpy as np  # Checks uploaded model inputs for infinite numeric values.
import pandas as pd  # Creates model input tables and reads or exports CSV data.
import streamlit as st  # Builds and runs the interactive web application.


APP_DIR = Path(__file__).resolve().parent  # Finds the folder containing this app, independent of the launch directory.
MODEL_PATH = APP_DIR.parent / "Our Trained Model" / "rf_wine_quality_model.joblib"  # Builds the model path from the project folder structure.

FEATURES = [  # Defines the exact input column names and order expected by the trained model.
    "fixed acidity",  # Wine's fixed acidity measurement.
    "volatile acidity",  # Wine's volatile acidity measurement.
    "citric acid",  # Wine's citric acid measurement.
    "residual sugar",  # Wine's residual sugar measurement.
    "chlorides",  # Wine's chloride measurement.
    "free sulfur dioxide",  # Wine's free sulfur dioxide measurement.
    "total sulfur dioxide",  # Wine's total sulfur dioxide measurement.
    "density",  # Wine's density measurement.
    "pH",  # Wine's pH measurement.
    "sulphates",  # Wine's sulphate measurement.
    "alcohol",  # Wine's alcohol percentage.
]

FEATURE_HELP = {  # Stores measurement units displayed beside the input controls.
    "fixed acidity": "g/L (usually reported as tartaric acid)",  # Unit/help text for fixed acidity.
    "volatile acidity": "g/L (usually reported as acetic acid)",  # Unit/help text for volatile acidity.
    "citric acid": "g/L",  # Unit/help text for citric acid.
    "residual sugar": "g/L",  # Unit/help text for residual sugar.
    "chlorides": "g/L",  # Unit/help text for chlorides.
    "free sulfur dioxide": "mg/L",  # Unit/help text for free sulfur dioxide.
    "total sulfur dioxide": "mg/L",  # Unit/help text for total sulfur dioxide.
    "density": "g/cm³",  # Unit/help text for density.
    "pH": "pH",  # Unit/help text for pH.
    "sulphates": "g/L",  # Unit/help text for sulphates.
    "alcohol": "% by volume",  # Unit/help text for alcohol.
}

SINGLE_WINE_DEFAULTS = {  # Supplies typical starting values that users can replace with lab results.
    "fixed acidity": 8.3,  # Default fixed acidity value.
    "volatile acidity": 0.5,  # Default volatile acidity value.
    "citric acid": 0.3,  # Default citric acid value.
    "residual sugar": 2.5,  # Default residual sugar value.
    "chlorides": 0.08,  # Default chloride value.
    "free sulfur dioxide": 15.0,  # Default free sulfur dioxide value.
    "total sulfur dioxide": 46.0,  # Default total sulfur dioxide value.
    "density": 0.996,  # Default density value.
    "pH": 3.3,  # Default pH value.
    "sulphates": 0.65,  # Default sulphate value.
    "alcohol": 10.2,  # Default alcohol percentage.
}


@st.cache_resource  # Caches the loaded estimator so reruns do not repeatedly read the model file.
def load_model():  # Loads the trained model and verifies its input schema before use.
    """Load the saved estimator and confirm it matches the app's wine feature schema."""
    if not MODEL_PATH.is_file():  # Stops inference if the expected model artifact is missing.
        raise FileNotFoundError(  # Reports the exact missing artifact location to the app.
            f"Trained model not found at {MODEL_PATH}. Keep the "  # Explains the model file location.
            "'Our Trained Model' folder beside 'Model Use Case'."  # Explains the required folder layout.
        )

    model = joblib.load(MODEL_PATH)  # Deserializes the trained estimator from disk.
    model_features = getattr(model, "feature_names_in_", None)  # Reads the stored training feature names when available.
    if model_features is not None and list(model_features) != FEATURES:  # Checks feature names and their exact order.
        raise ValueError(  # Prevents incorrect predictions from mismatched input columns.
            "The model's feature names or order do not match the wine input schema."  # Explains the schema mismatch.
        )

    if getattr(model, "n_features_in_", None) != len(FEATURES):  # Confirms the model expects all 11 chemistry measurements.
        raise ValueError(  # Stops use of an incompatible model artifact.
            f"Expected a model trained with {len(FEATURES)} features; "  # States how many inputs the app provides.
            f"found {getattr(model, 'n_features_in_', 'an unknown number')}."  # States how many inputs the model expects.
        )
    return model  # Makes the validated estimator available to the app.


def predict_wines(model, wine_features):  # Predicts quality scores and optional confidence for one or many wines.
    """Return predicted quality labels and the highest class probability, if supported."""
    qualities = model.predict(wine_features)  # Gets one predicted quality score for each input row.
    confidence = None  # Uses no confidence value when this estimator cannot calculate probabilities.
    if hasattr(model, "predict_proba"):  # Checks whether this estimator provides class probabilities.
        confidence = model.predict_proba(wine_features).max(axis=1)  # Uses the largest class probability as confidence.
    return qualities, confidence  # Returns predicted labels and optional confidence values.


def validate_wine_data(data):  # Checks an uploaded CSV and extracts valid, ordered model input columns.
    """Validate uploaded CSV columns and values, returning features or a user-facing error."""
    if data.columns.duplicated().any():  # Rejects ambiguous CSV schemas with repeated column names.
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()  # Collects duplicated column names for display.
        return None, f"CSV has duplicate column names: {', '.join(duplicate_columns)}."  # Returns a useful validation error.

    missing_features = [feature for feature in FEATURES if feature not in data.columns]  # Finds required model columns absent from the CSV.
    if missing_features:  # Prevents prediction if any required chemistry measurement is missing.
        return None, (  # Returns an error instead of passing an incomplete row to the model.
            "CSV is missing required model inputs: "  # Starts the error message.
            + ", ".join(missing_features)  # Names the required columns that are missing.
            + "."  # Finishes the message.
        )

    extra_columns = [  # Finds CSV columns that are not model inputs or the optional target label.
        column for column in data.columns if column not in FEATURES and column != "quality"  # Ignores known inputs and optional quality labels.
    ]
    if extra_columns:  # Notifies the user when uploaded columns will be ignored.
        st.warning(  # Displays an informational message in the app.
            "Additional columns will not be sent to the model: "  # Explains that extra columns are omitted from inference.
            + ", ".join(map(str, extra_columns))  # Lists the ignored columns.
            + "."  # Finishes the warning message.
        )

    features = data.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")  # Selects model columns in training order and converts them to numbers.
    invalid_values = features.isna() | ~np.isfinite(features)  # Marks missing, nonnumeric, infinite, and not-a-number values.
    if invalid_values.to_numpy().any():  # Rejects the batch if any required measurement is invalid.
        invalid_cells = []  # Creates a list of example invalid CSV locations.
        for column in FEATURES:  # Checks each model input column for invalid values.
            bad_rows = np.flatnonzero(invalid_values[column].to_numpy())  # Finds row positions containing invalid values in this column.
            invalid_cells.extend(  # Adds up to five invalid cells for the current column.
                f"row {row + 2} ({column})" for row in bad_rows[:5]  # Uses CSV row numbering, including the header.
            )
            if len(invalid_cells) >= 10:  # Keeps the displayed error concise by listing at most ten invalid cells.
                break  # Stops scanning after collecting enough examples.
        return None, (  # Returns the invalid data error rather than making predictions.
            "All model inputs must be finite numbers. Check "  # Explains the validation requirement.
            + ", ".join(invalid_cells)  # Lists sample rows and columns that need correction.
            + ". Row numbers include the CSV header."  # Clarifies how the reported CSV rows are numbered.
        )

    return features, None  # Returns cleaned, correctly ordered model input rows when validation succeeds.


st.set_page_config(  # Configures the browser title, icon, and layout for the Streamlit page.
    page_title="Red Wine Quality Predictor",  # Sets the browser tab title.
    page_icon="🍷",  # Sets the browser tab icon.
    layout="wide",  # Uses a wide page layout for the app controls and results.
)

st.title("🍷 Red Wine Quality Predictor")  # Displays the main app heading.
st.write(  # Introduces what the prediction app does.
    "Estimate a red wine's quality score from its laboratory measurements "  # Describes the prediction inputs.
    "using the Random Forest model trained in this project."  # Identifies the trained estimator used.
)
st.caption(  # Adds a brief limitation note below the introduction.
    "This is a model estimate based on the training data, not a substitute "  # Clarifies the result is an estimate.
    "for a professional sensory evaluation."  # Cautions against treating the prediction as professional assessment.
)

try:  # Attempts to load and validate the trained model before showing prediction forms.
    wine_model = load_model()  # Gets the cached, validated model for later predictions.
except (FileNotFoundError, ValueError) as error:  # Handles a missing artifact or an incompatible feature schema.
    st.error(str(error))  # Shows the specific model-loading problem to the user.
    st.stop()  # Stops the app because predictions cannot be made without a valid model.
except Exception as error:  # Catches other model-loading failures such as incompatible serialized dependencies.
    st.error(  # Shows an actionable message for unexpected model deserialization errors.
        f"Could not load the trained model ({type(error).__name__}): {error}. "  # Includes the original error details.
        "Check that the model file is valid and that the installed "  # Suggests verifying the saved artifact and environment.
        "scikit-learn version matches the one used to train it."  # Emphasizes matching the training scikit-learn version.
    )
    st.stop()  # Stops execution instead of displaying a broken prediction form.

single_tab, batch_tab = st.tabs(["Single wine", "Batch CSV"])  # Separates individual entry and multi-row CSV workflows.

with single_tab:  # Renders the individual wine prediction workflow.
    st.subheader("Enter laboratory measurements")  # Labels the single-sample form.
    st.write("Provide all 11 measurements to score one wine sample.")  # Explains which measurements the form needs.

    with st.form("single_wine_form"):  # Groups all single-sample inputs into one submit action.
        input_columns = st.columns(3)  # Arranges the measurement controls in three columns.
        single_values = {}  # Stores the values entered for the current wine sample.
        for index, feature in enumerate(FEATURES):  # Creates one input control for each model feature.
            with input_columns[index % len(input_columns)]:  # Places each control in the next available form column.
                single_values[feature] = st.number_input(  # Adds a numeric field and saves its current value.
                    f"{feature} ({FEATURE_HELP[feature]})",  # Labels the field with its name and measurement unit.
                    value=SINGLE_WINE_DEFAULTS[feature],  # Shows a typical starting value for the field.
                    format="%.4f",  # Displays values with enough precision for chemistry measurements.
                    help=f"Enter the laboratory result in {FEATURE_HELP[feature]}.",  # Provides additional unit guidance.
                )
        submitted = st.form_submit_button(  # Adds the action that submits the entered measurements.
            "Predict wine quality", type="primary", use_container_width=True  # Styles the action as the primary full-width button.
        )

    if submitted:  # Runs inference only after the user submits the sample.
        single_features = pd.DataFrame([single_values], columns=FEATURES)  # Converts entered measurements into the model's expected table.
        qualities, confidence = predict_wines(wine_model, single_features)  # Gets the predicted score and optional confidence.
        result_column, confidence_column = st.columns(2)  # Creates side-by-side result areas.
        result_column.metric("Predicted quality score", str(qualities[0]))  # Shows the predicted quality label.
        if confidence is not None:  # Displays confidence only when model probabilities are available.
            confidence_column.metric("Model confidence", f"{confidence[0]:.1%}")  # Formats the top class probability as a percentage.

        if confidence is not None and hasattr(wine_model, "classes_"):  # Shows per-class probability breakdown when supported.
            probabilities = pd.DataFrame(  # Creates a chart-ready table of probabilities for each score.
                {"Probability": wine_model.predict_proba(single_features)[0]},  # Stores each quality class probability.
                index=[str(label) for label in wine_model.classes_],  # Labels probabilities with their quality score classes.
            )
            st.write("Probability by quality score")  # Describes the chart shown below.
            st.bar_chart(probabilities)  # Visualizes the probability of each possible quality score.

with batch_tab:  # Renders the multi-sample CSV prediction workflow.
    st.subheader("Score multiple wine samples")  # Labels the batch prediction section.
    st.write(  # Explains the batch CSV format to the user.
        "Upload a CSV with the 11 chemistry columns used during training. "  # Lists the required CSV contents.
        "A `quality` column is optional and will not be used for prediction."  # Clarifies the target label is not a model input.
    )
    st.code(", ".join(FEATURES), language=None)  # Displays the exact required column names in the correct order.
    uploaded_csv = st.file_uploader(  # Adds a file picker for uploading the batch CSV.
        "Choose a CSV file", type=["csv"], key="wine_csv_upload"  # Restricts selection to CSV files and gives the widget a stable key.
    )

    if uploaded_csv is not None:  # Processes the file only after the user uploads one.
        try:  # Attempts to parse the uploaded CSV into a table.
            wine_data = pd.read_csv(uploaded_csv)  # Reads the uploaded wine samples and column names.
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:  # Handles common file parsing and reading errors.
            st.error(f"Could not read the uploaded CSV: {error}")  # Tells the user why the file could not be loaded.
        else:  # Continues with validation only when CSV reading succeeded.
            if wine_data.empty:  # Checks whether the CSV contains data rows.
                st.error("The uploaded CSV has a header but no wine samples.")  # Reports empty upload data to the user.
            else:  # Validates and scores a CSV containing one or more rows.
                wine_features, validation_error = validate_wine_data(wine_data)  # Checks input columns and values before inference.
                if validation_error:  # Prevents inference when the uploaded data is invalid.
                    st.error(validation_error)  # Displays the validation details to the user.
                else:  # Predicts only when all required CSV data is valid.
                    qualities, confidence = predict_wines(  # Gets a score and optional confidence for each CSV row.
                        wine_model, wine_features  # Uses the loaded estimator and cleaned input measurements.
                    )
                    results = wine_data.copy()  # Keeps the original uploaded columns in the displayed results.
                    results["predicted quality"] = qualities  # Adds the predicted score for each wine sample.
                    if confidence is not None:  # Adds model confidence when the estimator supports probabilities.
                        results["prediction confidence"] = confidence  # Stores the top class probability for each sample.

                    st.success(f"Predicted quality for {len(results)} wine samples.")  # Confirms how many rows were predicted.
                    st.dataframe(results, use_container_width=True)  # Displays the original columns with the new prediction results.
                    st.download_button(  # Adds an action for downloading the results table.
                        "Download predictions as CSV",  # Labels the download action.
                        data=results.to_csv(index=False).encode("utf-8"),  # Converts the results table to downloadable UTF-8 CSV bytes.
                        file_name="wine_quality_predictions.csv",  # Sets the suggested downloaded file name.
                        mime="text/csv",  # Tells the browser that the downloaded content is a CSV file.
                    )