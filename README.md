# LaptopIQ — AI Powered Laptop Advisor

LaptopIQ is a Streamlit-based laptop price predictor and recommendation assistant. It combines a trained regression pipeline with dataset-driven laptop suggestions and an AI-like advisor interface.

## Features

- Predicts laptop price from input specs
- Validates unrealistic configurations (Apple touchscreen, OS mismatches, Nvidia on Apple)
- Shows key metrics: price, availability, confidence
- Provides recommended laptops with image cards and action buttons
- Smart advisor chat flow for budget and usage-based suggestions

## Requirements

- Python 3.10+ (recommended)
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `requests`

The provided `requirements.txt` includes the core packages used by the app.

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install pandas numpy requests
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Files

- `app.py` — main Streamlit application
- `pipe.pkl` — trained regression model pipeline
- `laptop_data.csv` — dataset used for recommendations and persistence
- `requirements.txt` — core package list

## Optional Gemini integration

The app can optionally use Gemini if the following environment variables are set:

- `GEMINI_API_KEY`
- `GEMINI_API_URL`

If these are not provided, the advisor falls back to built-in responses.

## UI improvements

- Dark gradient background with modern card styling
- Two top actions: `💰 Predict Price` and `🚀 Smart Laptop Advisor`
- Colored buttons and product recommendation cards
- Metrics row for price, availability, and confidence

## License

This repository is ready for GitHub upload. If you want to publish it publicly, the MIT License is a good choice for open source projects.

> Add a `LICENSE` file if you want to make the licensing explicit for GitHub.
