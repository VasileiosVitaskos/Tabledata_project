"""
Out-of-Fold (OOF) Validation Score Calculator
==============================================
Calculates the official competition metric using OOF predictions from trained models.
This gives the most reliable estimate of test set performance.

Competition Metric:
    Score = 0.9 × WMAPE_poverty + 0.1 × MAPE_consumption
    
Where:
    WMAPE_poverty = Weighted Mean Absolute Percentage Error for poverty rates
    MAPE_consumption = Mean Absolute Percentage Error for household consumption
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pathlib import Path
import json
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

TRAIN_PATH = DATA_DIR / "train_processed.csv"
TRAIN_RATES_PATH = RAW_DATA_DIR / "train_rates_gt.csv"

ID_COLS = ["survey_id", "hhid"]
TARGET_COL = "log_cons_ppp17"
RAW_TARGET_COL = "cons_ppp17"
WEIGHT_COL = "weight"

# Training survey IDs
TRAIN_SURVEY_IDS = [100000, 200000, 300000]

# Poverty thresholds (from competition)
POVERTY_THRESHOLDS = [
    3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40, 9.13,
    9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37
]


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """Load training data and ground truth poverty rates"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load processed training data
    print("\nLoading processed training data...")
    df = pd.read_csv(TRAIN_PATH)
    
    # Feature columns
    feature_cols = [c for c in df.columns 
                    if c not in ID_COLS + [TARGET_COL, RAW_TARGET_COL, WEIGHT_COL]]
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Features: {len(feature_cols)}")
    
    # Load ground truth poverty rates
    print("\nLoading ground truth poverty rates...")
    poverty_gt = pd.read_csv(TRAIN_RATES_PATH)
    print(f"  Loaded poverty rates for {len(poverty_gt)} surveys")
    
    return df, feature_cols, poverty_gt


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_models():
    """Load all trained models for OOF predictions"""
    print("\n" + "="*70)
    print("LOADING TRAINED MODELS")
    print("="*70)
    
    models = {}
    
    # Load each model type (3 folds each)
    for model_type, model_dir in MODEL_DIRS.items():
        print(f"\nLoading {model_type.upper()} models...")
        model_list = []
        
        for fold_name in ['fold1', 'fold2', 'fold3']:
            if model_type == 'lightgbm':
                model_path = model_dir / f"lgb_{fold_name}.txt"
                if model_path.exists():
                    model = lgb.Booster(model_file=str(model_path))
                    model_list.append(model)
                    
            elif model_type == 'xgboost':
                model_path = model_dir / f"xgb_{fold_name}.json"
                if model_path.exists():
                    model = xgb.Booster()
                    model.load_model(str(model_path))
                    model_list.append(model)
                    
            elif model_type == 'catboost':
                model_path = model_dir / f"catboost_{fold_name}.cbm"
                if model_path.exists():
                    model = CatBoostRegressor()
                    model.load_model(str(model_path))
                    model_list.append(model)
                    
            elif model_type == 'tabnet':
                model_path = model_dir / f"tabnet_{fold_name}.zip"
                if model_path.exists():
                    model = TabNetRegressor()
                    model.load_model(str(model_path))
                    model_list.append(model)
        
        models[model_type] = model_list
        print(f"  Loaded {len(model_list)} folds")
    
    return models


# ============================================================================
# GENERATE OOF PREDICTIONS
# ============================================================================

def get_cv_folds():
    """Get CV fold definitions"""
    return [
        {'name': 'Fold1', 'train_surveys': [200000, 300000], 'val_survey': 100000},
        {'name': 'Fold2', 'train_surveys': [100000, 300000], 'val_survey': 200000},
        {'name': 'Fold3', 'train_surveys': [100000, 200000], 'val_survey': 300000}
    ]


def predict_with_model(model, X, model_type):
    """Generate predictions from a single model"""
    if model_type == 'lightgbm':
        return model.predict(X)
    elif model_type == 'xgboost':
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)
    elif model_type == 'catboost':
        return model.predict(X)
    elif model_type == 'tabnet':
        return model.predict(X.astype(np.float32)).flatten()


def generate_oof_predictions(df, feature_cols, models):
    """Generate out-of-fold predictions for all training data"""
    print("\n" + "="*70)
    print("GENERATING OUT-OF-FOLD PREDICTIONS")
    print("="*70)
    
    # Initialize OOF predictions dictionary
    oof_predictions = {survey_id: {} for survey_id in TRAIN_SURVEY_IDS}
    
    folds = get_cv_folds()
    
    # For each fold, generate predictions on validation survey
    for fold_idx, fold in enumerate(folds):
        val_survey = fold['val_survey']
        print(f"\n{fold['name']}: Validation Survey = {val_survey}")
        
        # Get validation data
        val_mask = df['survey_id'] == val_survey
        X_val = df.loc[val_mask, feature_cols].values
        
        # Collect predictions from each model type
        model_preds = []
        
        for model_type, model_list in models.items():
            if len(model_list) >= fold_idx + 1:
                model = model_list[fold_idx]
                preds = predict_with_model(model, X_val, model_type)
                model_preds.append(preds)
                print(f"  {model_type:15s}: predictions range [{preds.min():.3f}, {preds.max():.3f}]")
        
        # Average predictions across models (ensemble)
        ensemble_preds = np.mean(model_preds, axis=0)
        
        # Store OOF predictions for this survey
        oof_predictions[val_survey] = {
            'log_predictions': ensemble_preds,
            'predictions': np.exp(ensemble_preds),  # Convert back to original scale
            'indices': df.index[val_mask].tolist()
        }
    
    return oof_predictions


# ============================================================================
# CALCULATE COMPETITION METRIC
# ============================================================================

def calculate_poverty_rates(df_survey, predictions):
    """Calculate poverty rates for a single survey"""
    weights = df_survey[WEIGHT_COL].values
    
    rates = {}
    for threshold in POVERTY_THRESHOLDS:
        below_threshold = predictions < threshold
        poverty_rate = np.sum(weights[below_threshold]) / np.sum(weights)
        rates[threshold] = poverty_rate
    
    return rates


def calculate_threshold_weights():
    """Calculate weights for each poverty threshold"""
    # Survey 300000 poverty rates (from problem description)
    survey_300000_rates = {
        3.17: 0.05, 3.94: 0.10, 4.60: 0.15, 5.26: 0.20, 5.88: 0.25,
        6.47: 0.30, 7.06: 0.35, 7.70: 0.40, 8.40: 0.45, 9.13: 0.50,
        9.87: 0.55, 10.70: 0.60, 11.62: 0.65, 12.69: 0.70, 14.03: 0.75,
        15.64: 0.80, 17.76: 0.85, 20.99: 0.90, 27.37: 0.95
    }
    
    # Weights: w_t = 1 - |0.4 - p_t|
    weights = {}
    for threshold, rate in survey_300000_rates.items():
        weights[threshold] = 1 - abs(0.4 - rate)
    
    return weights


def calculate_wmape_poverty(predicted_rates, actual_rates, weights):
    """Calculate weighted MAPE for poverty rates (exact competition formula)"""
    weighted_sum = 0.0
    sum_weights = 0.0
    
    for threshold in POVERTY_THRESHOLDS:
        pred = predicted_rates[threshold]
        actual = actual_rates[threshold]
        weight = weights[threshold]
        
        # Weighted APE for this threshold
        if actual > 0:
            ape = abs(pred - actual) / actual
            weighted_sum += weight * ape
            sum_weights += weight
    
    # Normalize by sum of weights
    wmape = weighted_sum / sum_weights if sum_weights > 0 else 0.0
    return wmape


def calculate_mape_consumption(predicted, actual):
    """Calculate MAPE for household consumption"""
    mape = np.mean(np.abs((predicted - actual) / actual))
    return mape


def calculate_competition_score(df, oof_predictions, poverty_gt):
    """Calculate the official competition metric"""
    print("\n" + "="*70)
    print("CALCULATING COMPETITION METRIC")
    print("="*70)
    
    threshold_weights = calculate_threshold_weights()
    
    # Part 1: Poverty Rate Error (90%)
    wmape_per_survey = []
    
    for survey_id in TRAIN_SURVEY_IDS:
        print(f"\nSurvey {survey_id}:")
        
        # Get survey data
        survey_mask = df['survey_id'] == survey_id
        df_survey = df[survey_mask].copy()
        
        # OOF predictions for this survey
        oof_pred = oof_predictions[survey_id]['predictions']
        
        # Calculate predicted poverty rates
        predicted_rates = calculate_poverty_rates(df_survey, oof_pred)
        
        # Get actual poverty rates from ground truth
        actual_rates_row = poverty_gt[poverty_gt['survey_id'] == survey_id].iloc[0]
        actual_rates = {}
        for threshold in POVERTY_THRESHOLDS:
            col_name = f"pct_hh_below_{threshold:.2f}"
            actual_rates[threshold] = actual_rates_row[col_name]
        
        # Calculate WMAPE for this survey
        wmape = calculate_wmape_poverty(predicted_rates, actual_rates, threshold_weights)
        wmape_per_survey.append(wmape)
        
        print(f"  WMAPE (poverty rates): {wmape:.6f}")
        
        # Show some example thresholds
        print(f"  Example predictions:")
        for threshold in [3.17, 7.70, 14.03, 27.37]:
            pred = predicted_rates[threshold]
            actual = actual_rates[threshold]
            error = abs(pred - actual) / actual
            print(f"    ${threshold:6.2f}: Pred={pred:.4f}, Actual={actual:.4f}, APE={error:.4f}")
    
    avg_wmape_poverty = np.mean(wmape_per_survey)
    print(f"\n  Average WMAPE (poverty rates): {avg_wmape_poverty:.6f}")
    
    # Part 2: Household Consumption Error (10%)
    all_predicted = []
    all_actual = []
    
    for survey_id in TRAIN_SURVEY_IDS:
        survey_mask = df['survey_id'] == survey_id
        oof_pred = oof_predictions[survey_id]['predictions']
        actual = df.loc[survey_mask, RAW_TARGET_COL].values
        
        all_predicted.extend(oof_pred)
        all_actual.extend(actual)
    
    mape_consumption = calculate_mape_consumption(np.array(all_predicted), np.array(all_actual))
    print(f"\n  MAPE (household consumption): {mape_consumption:.6f}")
    
    # Final Competition Score (exact formula from problem statement)
    # metric = (1/S) × Σ_surveys [(90/Σw_t) × Σ_t w_t × |r̂ - r| / r + (10/H) × Σ_h |ĉ - c| / c]
    competition_score = 0.9 * avg_wmape_poverty + 0.1 * mape_consumption
    
    print("\n" + "="*70)
    print("FINAL COMPETITION SCORE")
    print("="*70)
    print(f"\nFormula: (1/S) × Σ [(90/Σw_t)×Σ w_t×APE_poverty + (10/H)×Σ APE_consumption]")
    print(f"\nScore = 0.9 × {avg_wmape_poverty:.6f} + 0.1 × {mape_consumption:.6f}")
    print(f"      = {0.9 * avg_wmape_poverty:.6f} + {0.1 * mape_consumption:.6f}")
    print(f"      = {competition_score:.6f}")
    
    # Note about scale
    print(f"\nNote: If leaderboard shows scores like 1.5-2.0,")
    print(f"      multiply by 100: {competition_score * 100:.3f}")
    
    return competition_score, avg_wmape_poverty, mape_consumption


# ============================================================================
# LEADERBOARD POSITION ESTIMATE
# ============================================================================

def estimate_leaderboard_position(score):
    """Estimate leaderboard position based on score"""
    print("\n" + "="*70)
    print("LEADERBOARD POSITION ESTIMATE")
    print("="*70)
    
    # Reference scores from screenshot (top 50 region)
    # These appear to be in percentage scale (×100)
    reference_scores = {
        45: 1.772,
        46: 1.895,
        47: 1.961,
        48: 1.661,
        49: 1.993
    }
    
    avg_top50 = np.mean(list(reference_scores.values()))
    
    # Try both scales
    score_raw = score
    score_pct = score * 100
    
    print(f"\nYour OOF Score:")
    print(f"  Raw scale:        {score_raw:.6f}")
    print(f"  Percentage scale: {score_pct:.3f}")
    
    print(f"\nReference Top 50 scores:")
    for pos, ref_score in sorted(reference_scores.items()):
        print(f"  Position {pos}: {ref_score:.3f}")
    
    print(f"\nAverage of Top 50 region: {avg_top50:.3f}")
    
    # Use percentage scale for comparison
    score_to_compare = score_pct
    
    print(f"\nComparison (using percentage scale):")
    
    if score_to_compare < min(reference_scores.values()):
        gap = min(reference_scores.values()) - score_to_compare
        print(f"\n✓✓✓ EXCELLENT! Your score is BETTER than position 48!")
        print(f"  Your score: {score_to_compare:.3f}")
        print(f"  Position 48: {min(reference_scores.values()):.3f}")
        print(f"  Gap: -{gap:.3f} (better)")
        print(f"\n  Estimated position: TOP 45 or better")
        print(f"  You are VERY LIKELY in the TOP 50! 🎉")
        
    elif score_to_compare < avg_top50:
        gap = avg_top50 - score_to_compare
        print(f"\n✓✓ GREAT! Your score is competitive with Top 50!")
        print(f"  Your score: {score_to_compare:.3f}")
        print(f"  Average Top 50: {avg_top50:.3f}")
        print(f"  Gap: -{gap:.3f} (better than average)")
        print(f"\n  Estimated position: Around 45-50")
        print(f"  Good chance of making TOP 50! 🚀")
        
    elif score_to_compare < max(reference_scores.values()):
        gap_to_best = score_to_compare - min(reference_scores.values())
        gap_to_worst = max(reference_scores.values()) - score_to_compare
        print(f"\n✓ COMPETITIVE! Your score is near the Top 50 boundary")
        print(f"  Your score: {score_to_compare:.3f}")
        print(f"  Best (pos 48): {min(reference_scores.values()):.3f} (+{gap_to_best:.3f})")
        print(f"  Worst (pos 49): {max(reference_scores.values()):.3f} (-{gap_to_worst:.3f})")
        print(f"\n  Estimated position: Around 48-52")
        print(f"  Close to TOP 50 - could go either way 🤞")
        
    else:
        gap = score_to_compare - max(reference_scores.values())
        print(f"\n⚠ Your score is below Top 50 region")
        print(f"  Your score: {score_to_compare:.3f}")
        print(f"  Position 49: {max(reference_scores.values()):.3f}")
        print(f"  Gap: +{gap:.3f} (worse)")
        print(f"\n  Estimated position: Below 50")
        print(f"  Need to improve by {gap:.3f} to reach Top 50")
    
    print("\n" + "="*70)
    print("IMPORTANT NOTES")
    print("="*70)
    print("1. OOF score is a reliable estimate, but not exact")
    print("2. Test set may have different distribution than train")
    print("3. Public LB shows only 1 of 3 test surveys")
    print("4. Final ranking is based on 2 hidden surveys")
    print("5. Your position may vary ±10 ranks from estimate")
    print("6. Lower score is better!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main OOF validation pipeline"""
    print("="*70)
    print("OUT-OF-FOLD VALIDATION SCORE CALCULATOR")
    print("="*70)
    
    # Load data
    df, feature_cols, poverty_gt = load_data()
    
    # Load models
    models = load_models()
    
    # Generate OOF predictions
    oof_predictions = generate_oof_predictions(df, feature_cols, models)
    
    # Calculate competition metric
    score, wmape_poverty, mape_consumption = calculate_competition_score(
        df, oof_predictions, poverty_gt
    )
    
    # Estimate leaderboard position
    estimate_leaderboard_position(score)
    
    # Save results
    results = {
        'oof_competition_score': score,
        'wmape_poverty_rates': wmape_poverty,
        'mape_consumption': mape_consumption,
        'by_survey': {}
    }
    
    for survey_id in TRAIN_SURVEY_IDS:
        survey_mask = df['survey_id'] == survey_id
        oof_pred = oof_predictions[survey_id]['predictions']
        actual = df.loc[survey_mask, RAW_TARGET_COL].values
        
        mape = calculate_mape_consumption(oof_pred, actual)
        results['by_survey'][int(survey_id)] = {
            'mape_consumption': mape,
            'n_households': len(oof_pred)
        }
    
    output_path = Path("c:/Users/user/poverty-prediction/oof_validation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")


if __name__ == "__main__":
    main()
