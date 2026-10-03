# Red Wine Quality Prediction

A machine-learning project that predicts a red wine quality score from its
laboratory measurements. The project includes the dataset, a Jupyter notebook
for model development, a saved Random Forest model, and a Streamlit application
for individual and batch predictions.

## Project screenshots

### Correlation matrix

This image shows the feature correlations explored during data analysis:

![Wine feature correlation matrix](Pictures%20For%20Readme/Correlation%20Matrix%20%281%29.png)

### Streamlit prediction application

This screenshot shows the wine-quality prediction interface:

![Streamlit wine quality prediction application](Pictures%20For%20Readme/result%20on%20steamlit.png)

## Repository contents

```text
DataSet/
  winequality-red.csv
Jupiter Notebook/
  wine_quality_prediction_machine_learning_model.ipynb
Model Use Case/
  app.py
  requirements.txt
Our Trained Model/
  rf_wine_quality_model.joblib
Pictures For Readme/
  Correlation Matrix (1).png
  result on steamlit.png
```

The model is a trained `RandomForestClassifier`. The application loads the
existing model artifact; it does not retrain the model on startup.

## Run the Streamlit application

Use Python 3.10 or later. From the repository root, create and activate a
virtual environment, install the app dependencies, and start Streamlit:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and run the application:

```bash
pip install -r "Model Use Case/requirements.txt"
streamlit run "Model Use Case/app.py"
```

The app expects the model at
`Our Trained Model/rf_wine_quality_model.joblib`, relative to the project
folders. Keep that folder beside `Model Use Case`.

## Application features

- Enter laboratory measurements for one wine and get a predicted quality score.
- Upload a CSV to predict scores for multiple wine samples.
- See the highest predicted class probability and, for a single sample, the
  probabilities by quality score.
- Download batch predictions as a CSV file.

Batch CSV files must include these columns (column order does not matter):

```text
fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol
```

An optional `quality` column may be present, but it is not used for prediction.
Additional columns are retained in the output and ignored by the model.

## Model and prediction notes

The model uses 11 physicochemical measurements and predicts the quality label
learned during training. The displayed probability is the model's largest
predicted class probability; it should not be interpreted as a guarantee of
prediction accuracy. Results are estimates and are not a substitute for
professional sensory evaluation.

The saved model was serialized with scikit-learn 1.6.1. The app's
`requirements.txt` pins that version for model compatibility.
