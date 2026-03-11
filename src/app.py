from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os

app = FastAPI(title="AI-Driven Ocean Data Platform 🌊")

# Paths
DATA_PATH = "data/annual.csv"
MODEL_PATH = "models/ocean_model.pkl"
PLOTS_DIR = "plots"

# Load dataset & model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Templates
templates = Jinja2Templates(directory="templates")

# Serve plots folder as static files
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# Home route → render dashboard
@app.get("/")
def dashboard(request: Request):
    # Prepare dataset preview
    data_preview = df.head(5).to_dict(orient="records")
    columns = df.columns.tolist()
    
    # List plots
    plot_files = os.listdir(PLOTS_DIR)
    
    # Predictions
    future_years = pd.DataFrame({'Year': [2025, 2030, 2040, 2050]})
    predictions = {year: round(temp, 2) for year, temp in zip(future_years['Year'], model.predict(future_years))}
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "data": data_preview,
        "columns": columns,
        "plots": plot_files,
        "predictions": predictions
    })

# Optional: keep previous routes if needed
@app.get("/data")
def get_data():
    return {"columns": df.columns.tolist(), "rows": df.head(5).to_dict(orient="records")}

@app.get("/predict")
def predict_future():
    future_years = pd.DataFrame({'Year': [2025, 2030, 2040, 2050]})
    predictions = {year: round(temp, 2) for year, temp in zip(future_years['Year'], model.predict(future_years))}
    return {"future_predictions": predictions}