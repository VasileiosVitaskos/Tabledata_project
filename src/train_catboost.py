"""
CatBoost Model Training with Optuna Optimization
================================================
- Survey-level Cross-Validation (3 folds)
- GPU acceleration
- Optimize for MAE on log-transformed target
- Save best model per fold
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import optuna
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("c:/Users/user/poverty-prediction/data/processed")
MODEL_DIR = Path("c:/Users/user/poverty-prediction/models/catboost")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_processed.csv"
TARGET_COL = "log_cons_ppp17"
RAW_TARGET_COL = "cons_ppp17"
WEIGHT_COL = "weight"
ID_COLS = ["survey_id", "hhid"]

# Survey IDs for CV
SURVEY_IDS = [100000, 200000, 300000]

# Optuna config
N_TRIALS = 50
RANDOM_STATE = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load processed training data"""
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH)
    
    # Feature columns
    feature_cols = [c for c in df.columns 
                    if c not in ID_COLS + [TARGET_COL, RAW_TARGET_COL, WEIGHT_COL]]
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Surveys: {df['survey_id'].unique().tolist()}")
    
    return df, feature_cols


# ============================================================================
# SURVEY-LEVEL CV SPLITS
# ============================================================================

def get_cv_folds():
    """
    Generate survey-level CV folds
    Each fold uses 2 surveys for training, 1 for validation
    """
    folds = [
        {
            'name': 'Fold1',
            'train_surveys': [200000, 300000],
            'val_survey': 100000
        },
        {
            'name': 'Fold2',
            'train_surveys': [100000, 300000],
            'val_survey': 200000
        },
        {
            'name': 'Fold3',
            'train_surveys': [100000, 200000],
            'val_survey': 300000
        }
    ]
    return folds


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_mae_log(y_true, y_pred):
    """MAE on log scale (what we optimize)"""
    return mean_absolute_error(y_true, y_pred)


def calculate_mae_original(y_true_log, y_pred_log):
    """MAE on original scale (for reporting)"""
    y_true_original = np.exp(y_true_log)
    y_pred_original = np.exp(y_pred_log)
    return mean_absolute_error(y_true_original, y_pred_original)


# ============================================================================
# CATBOOST TRAINING
# ============================================================================

def train_catboost_fold(X_train, y_train, w_train, X_val, y_val, w_val, params):
    """Train CatBoost on one fold"""
    
    # Create Pool objects
    train_pool = Pool(
        X_train,
        label=y_train,
        weight=w_train
    )
    
    val_pool = Pool(
        X_val,
        label=y_val,
        weight=w_val
    )
    
    # Initialize model
    model = CatBoostRegressor(**params)
    
    # Train
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=100,
        early_stopping_rounds=100
    )
    
    # Predict
    y_pred_val = model.predict(X_val)
    
    # Metrics
    mae_log = calculate_mae_log(y_val, y_pred_val)
    mae_orig = calculate_mae_original(y_val, y_pred_val)
    
    return model, mae_log, mae_orig


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial, df, feature_cols):
    """Optuna objective function"""
    
    # Hyperparameters to optimize
    params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'GPU',
        'devices': '0',
        'verbose': False,
        
        # Hyperparameters to tune
        'iterations': 10000,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        
        
        # Fixed
        'random_seed': RANDOM_STATE,
        'thread_count': -1,
    }
    
    # Cross-validation
    folds = get_cv_folds()
    fold_scores = []
    
    for fold in folds:
        # Split data
        train_mask = df['survey_id'].isin(fold['train_surveys'])
        val_mask = df['survey_id'] == fold['val_survey']
        
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, TARGET_COL].values
        w_train = df.loc[train_mask, WEIGHT_COL].values
        
        X_val = df.loc[val_mask, feature_cols].values
        y_val = df.loc[val_mask, TARGET_COL].values
        w_val = df.loc[val_mask, WEIGHT_COL].values
        
        # Train
        try:
            _, mae_log, _ = train_catboost_fold(
                X_train, y_train, w_train,
                X_val, y_val, w_val,
                params
            )
            fold_scores.append(mae_log)
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    # Return average MAE across folds
    avg_mae = np.mean(fold_scores)
    return avg_mae


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_with_optuna():
    """Main training pipeline with Optuna optimization"""
    
    print("="*70)
    print("CATBOOST TRAINING WITH OPTUNA")
    print("="*70)
    
    # Load data
    df, feature_cols = load_data()
    
    # Optuna study
    print(f"\nStarting Optuna optimization ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        lambda trial: objective(trial, df, feature_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    # Best parameters
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best MAE (log): {study.best_value:.6f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final models on each fold with best params
    print("\n" + "="*70)
    print("TRAINING FINAL MODELS ON EACH FOLD")
    print("="*70)
    
    best_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'GPU',
        'devices': '0',
        'iterations': 10000,
        'random_seed': RANDOM_STATE,
        'thread_count': -1,
        **study.best_params
    }
    
    folds = get_cv_folds()
    fold_results = []
    
    for fold in folds:
        print(f"\n{fold['name']}: Train={fold['train_surveys']}, Val={fold['val_survey']}")
        
        # Split data
        train_mask = df['survey_id'].isin(fold['train_surveys'])
        val_mask = df['survey_id'] == fold['val_survey']
        
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, TARGET_COL].values
        w_train = df.loc[train_mask, WEIGHT_COL].values
        
        X_val = df.loc[val_mask, feature_cols].values
        y_val = df.loc[val_mask, TARGET_COL].values
        w_val = df.loc[val_mask, WEIGHT_COL].values
        
        # Train
        model, mae_log, mae_orig = train_catboost_fold(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            best_params
        )
        
        print(f"  MAE (log): {mae_log:.6f}")
        print(f"  MAE (original): {mae_orig:.2f}")
        
        # Save model
        model_path = MODEL_DIR / f"catboost_{fold['name'].lower()}.cbm"
        model.save_model(str(model_path))
        print(f"  Saved: {model_path}")
        
        fold_results.append({
            'fold': fold['name'],
            'val_survey': fold['val_survey'],
            'mae_log': mae_log,
            'mae_original': mae_orig,
            'best_iteration': model.best_iteration_
        })
    
    # Summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    avg_mae_log = np.mean([r['mae_log'] for r in fold_results])
    avg_mae_orig = np.mean([r['mae_original'] for r in fold_results])
    print(f"Average MAE (log): {avg_mae_log:.6f}")
    print(f"Average MAE (original): {avg_mae_orig:.2f}")
    
    # Save results
    results = {
        'model': 'CatBoost',
        'best_params': best_params,
        'cv_folds': fold_results,
        'avg_mae_log': avg_mae_log,
        'avg_mae_original': avg_mae_orig,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols)
    }
    
    results_path = MODEL_DIR / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    print("\n✓ Training complete!")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = train_with_optuna()
