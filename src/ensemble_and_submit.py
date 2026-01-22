"""
Ensemble Models & Generate Submission
======================================
- Load all trained models (LightGBM, XGBoost, CatBoost, TabNet)
- Weighted average ensemble based on CV performance
- Generate predictions for test set
- Calculate poverty rates per survey
- Create submission.zip file
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from pathlib import Path
import json
import zipfile
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("c:/Users/user/poverty-prediction/data/processed")
RAW_DATA_DIR = Path("c:/Users/user/poverty-prediction/data/raw")
MODEL_DIRS = {
    'lightgbm': Path("c:/Users/user/poverty-prediction/models/lightgbm"),
    'xgboost': Path("c:/Users/user/poverty-prediction/models/xgboost"),
    'catboost': Path("c:/Users/user/poverty-prediction/models/catboost"),
    'tabnet': Path("c:/Users/user/poverty-prediction/models/tabnet")
}
OUTPUT_DIR = Path("c:/Users/user/poverty-prediction/submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = DATA_DIR / "test_processed.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test_hh_features.csv"

ID_COLS = ["survey_id", "hhid"]
WEIGHT_COL = "weight"

# Test survey IDs
TEST_SURVEY_IDS = [400000, 500000, 600000]

# Poverty thresholds (from problem description)
POVERTY_THRESHOLDS = [
    3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40, 9.13,
    9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37
]


# ============================================================================
# LOAD TEST DATA
# ============================================================================

def load_test_data():
    """Load processed test data"""
    print("Loading test data...")
    df = pd.read_csv(TEST_PATH)
    
    # Get feature columns from metadata
    metadata_path = DATA_DIR / "processing_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_columns']
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Test surveys: {df['survey_id'].unique().tolist()}")
    
    return df, feature_cols


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_all_models():
    """Load all trained models and their CV results"""
    
    print("\n" + "="*70)
    print("LOADING TRAINED MODELS")
    print("="*70)
    
    models = {}
    cv_scores = {}
    
    # LightGBM
    print("\nLoading LightGBM models...")
    lgb_models = []
    for fold_name in ['fold1', 'fold2', 'fold3']:
        model_path = MODEL_DIRS['lightgbm'] / f"lgb_{fold_name}.txt"
        if model_path.exists():
            model = lgb.Booster(model_file=str(model_path))
            lgb_models.append(model)
            print(f"  ✓ Loaded {fold_name}")
    
    # Load CV results
    with open(MODEL_DIRS['lightgbm'] / "training_results.json", 'r') as f:
        lgb_results = json.load(f)
        cv_scores['lightgbm'] = lgb_results['avg_mae_log']
    
    models['lightgbm'] = lgb_models
    print(f"  CV MAE (log): {cv_scores['lightgbm']:.6f}")
    
    # XGBoost
    print("\nLoading XGBoost models...")
    xgb_models = []
    for fold_name in ['fold1', 'fold2', 'fold3']:
        model_path = MODEL_DIRS['xgboost'] / f"xgb_{fold_name}.json"
        if model_path.exists():
            model = xgb.Booster()
            model.load_model(str(model_path))
            xgb_models.append(model)
            print(f"  ✓ Loaded {fold_name}")
    
    with open(MODEL_DIRS['xgboost'] / "training_results.json", 'r') as f:
        xgb_results = json.load(f)
        cv_scores['xgboost'] = xgb_results['avg_mae_log']
    
    models['xgboost'] = xgb_models
    print(f"  CV MAE (log): {cv_scores['xgboost']:.6f}")
    
    # CatBoost
    print("\nLoading CatBoost models...")
    catboost_models = []
    for fold_name in ['fold1', 'fold2', 'fold3']:
        model_path = MODEL_DIRS['catboost'] / f"catboost_{fold_name}.cbm"
        if model_path.exists():
            model = CatBoostRegressor()
            model.load_model(str(model_path))
            catboost_models.append(model)
            print(f"  ✓ Loaded {fold_name}")
    
    with open(MODEL_DIRS['catboost'] / "training_results.json", 'r') as f:
        catboost_results = json.load(f)
        cv_scores['catboost'] = catboost_results['avg_mae_log']
    
    models['catboost'] = catboost_models
    print(f"  CV MAE (log): {cv_scores['catboost']:.6f}")
    
    # TabNet
    print("\nLoading TabNet models...")
    tabnet_models = []
    for fold_name in ['fold1', 'fold2', 'fold3']:
        model_path = MODEL_DIRS['tabnet'] / f"tabnet_{fold_name}.zip"
        if model_path.exists():
            model = TabNetRegressor()
            model.load_model(str(model_path))
            tabnet_models.append(model)
            print(f"  ✓ Loaded {fold_name}")
    
    with open(MODEL_DIRS['tabnet'] / "training_results.json", 'r') as f:
        tabnet_results = json.load(f)
        cv_scores['tabnet'] = tabnet_results['avg_mae_log']
    
    models['tabnet'] = tabnet_models
    print(f"  CV MAE (log): {cv_scores['tabnet']:.6f}")
    
    return models, cv_scores


# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

def predict_with_models(models, X, model_type):
    """Generate predictions from a list of models (one per fold)"""
    
    predictions = []
    
    for model in models:
        if model_type == 'lightgbm':
            pred = model.predict(X)
        elif model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            pred = model.predict(dtest)
        elif model_type == 'catboost':
            pred = model.predict(X)
        elif model_type == 'tabnet':
            pred = model.predict(X.astype(np.float32)).flatten()
        
        predictions.append(pred)
    
    # Average predictions across folds
    return np.mean(predictions, axis=0)


def generate_ensemble_predictions(models, cv_scores, test_df, feature_cols):
    """Generate weighted ensemble predictions"""
    
    print("\n" + "="*70)
    print("GENERATING ENSEMBLE PREDICTIONS")
    print("="*70)
    
    X_test = test_df[feature_cols].values
    
    # Generate predictions for each model type
    all_predictions = {}
    
    for model_type, model_list in models.items():
        print(f"\nPredicting with {model_type.upper()}...")
        preds = predict_with_models(model_list, X_test, model_type)
        all_predictions[model_type] = preds
        print(f"  Predictions: [{preds.min():.4f}, {preds.max():.4f}]")
    
    # Calculate weights based on inverse of CV MAE (lower MAE = higher weight)
    weights = {}
    total_inverse_mae = sum(1.0 / cv_scores[m] for m in cv_scores)
    
    for model_type in cv_scores:
        weights[model_type] = (1.0 / cv_scores[model_type]) / total_inverse_mae
    
    print("\n" + "="*70)
    print("ENSEMBLE WEIGHTS (based on CV performance)")
    print("="*70)
    for model_type, weight in weights.items():
        print(f"  {model_type:15s}: {weight:.4f} (CV MAE: {cv_scores[model_type]:.6f})")
    
    # Weighted ensemble
    ensemble_preds_log = np.zeros(len(X_test))
    for model_type, preds in all_predictions.items():
        ensemble_preds_log += weights[model_type] * preds
    
    # Convert from log to original scale
    ensemble_preds = np.exp(ensemble_preds_log)
    
    print(f"\nEnsemble predictions (original scale):")
    print(f"  Min: {ensemble_preds.min():.2f}")
    print(f"  Max: {ensemble_preds.max():.2f}")
    print(f"  Mean: {ensemble_preds.mean():.2f}")
    print(f"  Median: {np.median(ensemble_preds):.2f}")
    
    return ensemble_preds


# ============================================================================
# CALCULATE POVERTY RATES
# ============================================================================

def calculate_poverty_rates(predictions_df):
    """Calculate poverty rates for each survey using weighted sampling"""
    
    print("\n" + "="*70)
    print("CALCULATING POVERTY RATES")
    print("="*70)
    
    poverty_rates = []
    
    for survey_id in TEST_SURVEY_IDS:
        survey_data = predictions_df[predictions_df['survey_id'] == survey_id].copy()
        
        print(f"\nSurvey {survey_id}:")
        print(f"  Households: {len(survey_data)}")
        
        # Get consumption and weights
        consumption = survey_data['per_capita_household_consumption'].values
        weights = survey_data[WEIGHT_COL].values
        
        # Calculate poverty rate at each threshold
        rates_row = {'survey_id': survey_id}
        
        for threshold in POVERTY_THRESHOLDS:
            # Households below threshold
            below_threshold = consumption < threshold
            
            # Weighted poverty rate
            poverty_rate = np.sum(weights[below_threshold]) / np.sum(weights)
            
            # Store in format required by submission
            col_name = f"pct_hh_below_{threshold:.2f}".replace('.', '_')
            rates_row[col_name] = poverty_rate
        
        poverty_rates.append(rates_row)
        
        # Print summary
        print(f"  Poverty rates: [{rates_row[f'pct_hh_below_3_17']:.4f}, {rates_row[f'pct_hh_below_27_37']:.4f}]")
    
    poverty_df = pd.DataFrame(poverty_rates)
    
    return poverty_df


# ============================================================================
# CREATE SUBMISSION
# ============================================================================

def create_submission(predictions_df, poverty_df):
    """Create submission zip file with required CSVs"""
    
    print("\n" + "="*70)
    print("CREATING SUBMISSION FILES")
    print("="*70)
    
    # 1. Household consumption predictions
    household_csv = predictions_df[['survey_id', 'hhid', 'per_capita_household_consumption']].copy()
    household_csv.columns = ['survey_id', 'hhid', 'cons_ppp17']
    
    household_path = OUTPUT_DIR / "predicted_household_consumption.csv"
    household_csv.to_csv(household_path, index=False)
    print(f"\n✓ Created: {household_path}")
    print(f"  Rows: {len(household_csv)}")
    
    # 2. Poverty distribution
    # Reorder columns to match submission format
    threshold_cols = [f"pct_hh_below_{t:.2f}".replace('.', '_') for t in POVERTY_THRESHOLDS]
    poverty_csv = poverty_df[['survey_id'] + threshold_cols].copy()
    
    # Rename columns to match exact submission format
    col_rename = {'survey_id': 'survey_id'}
    for t in POVERTY_THRESHOLDS:
        old_name = f"pct_hh_below_{t:.2f}".replace('.', '_')
        new_name = f"pct_hh_below_{t:.2f}"
        col_rename[old_name] = new_name
    
    poverty_csv.columns = [col_rename.get(c, c) for c in poverty_csv.columns]
    
    poverty_path = OUTPUT_DIR / "predicted_poverty_distribution.csv"
    poverty_csv.to_csv(poverty_path, index=False)
    print(f"\n✓ Created: {poverty_path}")
    print(f"  Rows: {len(poverty_csv)}")
    
    # 3. Create ZIP file
    submission_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(submission_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(household_path, "predicted_household_consumption.csv")
        zipf.write(poverty_path, "predicted_poverty_distribution.csv")
    
    print(f"\n✓ Created: {submission_path}")
    print(f"  Contains:")
    print(f"    - predicted_household_consumption.csv")
    print(f"    - predicted_poverty_distribution.csv")
    
    return submission_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main submission generation pipeline"""
    
    print("="*70)
    print("ENSEMBLE & SUBMISSION GENERATION")
    print("="*70)
    
    # Load test data
    test_df, feature_cols = load_test_data()
    
    # Load models
    models, cv_scores = load_all_models()
    
    # Generate ensemble predictions
    ensemble_preds = generate_ensemble_predictions(models, cv_scores, test_df, feature_cols)
    
    # Create predictions dataframe
    predictions_df = test_df[ID_COLS + [WEIGHT_COL]].copy()
    predictions_df['per_capita_household_consumption'] = ensemble_preds
    
    # Calculate poverty rates
    poverty_df = calculate_poverty_rates(predictions_df)
    
    # Create submission
    submission_path = create_submission(predictions_df, poverty_df)
    
    print("\n" + "="*70)
    print("✓ SUBMISSION COMPLETE!")
    print("="*70)
    print(f"\nSubmission file: {submission_path}")
    print(f"\nReady to upload to competition!")
    
    return submission_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    submission_path = main()
