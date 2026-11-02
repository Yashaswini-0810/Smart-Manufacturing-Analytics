# Smart Manufacturing Analytics Dashboard

## Files required in this folder
- app.py
- requirements.txt
- manufacturing_cleaned.csv
- manufacturing_original.csv
- best_model_final.joblib

## Run locally
1. Install Python.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

The dashboard uses the six-feature tuned Random Forest model and intentionally
does not use Optimization_Category as a prediction input because its relationship
with Job_Status raised a possible target-leakage concern.
