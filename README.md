# Real Estate Buyer Segmentation & Investment Profiling

Machine learning based buyer segmentation and investment profiling for **Parcl Co. Limited** real estate market intelligence.

## Overview

This project discovers hidden buyer segments using K-Means and hierarchical clustering on client demographics, financing behavior, and property transaction data. A Streamlit dashboard provides live analytics for marketing and investment decisions.

## Project Structure

```
real-estate-buyer-segmentation/
├── app/streamlit_app.py      # Interactive dashboard
├── scripts/train_model.py    # Full ML training pipeline
├── src/                      # Core modules (cleaning, features, clustering)
├── data/                     # clients.csv & properties.csv
├── models/                   # Trained models & segmented output
├── outputs/figures/          # EDA & evaluation charts
└── docs/research_paper.md    # Research paper with EDA & recommendations
```

## Setup

```bash
cd ~/Projects/real-estate-buyer-segmentation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
python scripts/train_model.py
```

This runs the full pipeline:
1. Data cleaning & property aggregation
2. Feature encoding (One-Hot + Label) & StandardScaler
3. K-Means & hierarchical clustering
4. Elbow & silhouette evaluation
5. Segment interpretation & artifact export

## Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Dashboard Features

- **Buyer Segmentation Overview** – cluster distribution and model metrics
- **Investor Behavior Dashboard** – investment patterns by segment
- **Geographic Buyer Analysis** – regional and country-level segment maps
- **Segment Insights Panel** – descriptive statistics per cluster

### Filters

- Country, Region, Acquisition Purpose, Client Type

## Buyer Segments

| Segment | Profile |
|---------|---------|
| Global Investors | High investment, investment-purpose purchases |
| First-Time Buyers | Younger, loan-dependent, personal use |
| Corporate Buyers | Company clients, multiple units |
| Luxury Investors | High satisfaction, large investments |

## Requirements

- Python 3.10+
- pandas, scikit-learn, streamlit, plotly, matplotlib, seaborn

## Author

Unified Mentor – Parcl Co. Limited Real Estate Market Intelligence Project
