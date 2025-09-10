# ML-Powered Disease Prediction App

A lightweight, offline-capable web application built with Python and Streamlit to predict risks for diabetes, heart disease, and kidney disease using machine learning models. The app includes real-time visualizations, a personalized AI chatbot (when available), and exportable prediction history, designed for clinicians and patients, especially in low-resource settings.

## Features
- Predicts risks for diabetes, heart disease, and kidney disease using optimized Random Forest models.
- Interactive UI with Plotly visualizations for key input values.
- AI chatbot (powered by transformers like distilgpt2) for personalized health advice (optional).
- Save and export prediction history as CSV files.
- Runs on low-memory devices (<8GB RAM) with CPU optimization.

## Prerequisites
- Python 3.8 or higher
- Git (for version control)
- Internet connection (initial setup for model downloads)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ml-disease-prediction-app.git
   cd ml-disease-prediction-app

2. Create a virtual environment and activate it:   
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the required packages:
    ```bash
    pip install -r requirements.txt