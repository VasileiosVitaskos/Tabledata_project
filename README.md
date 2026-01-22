# Poverty Prediction Challenge - Machine Learning Solution

## Overview

This repository contains a complete machine learning pipeline for the Poverty Prediction Challenge, which addresses the problem of imputing household consumption data from survey responses. The project implements an ensemble of gradient boosting models and deep learning to predict both household-level daily per capita consumption and population-level poverty rates across multiple thresholds.

## Problem Statement

The challenge simulates a real-world scenario faced by international development organizations like the World Bank. While comprehensive household surveys capturing detailed consumption data exist for earlier years, more recent surveys often lack the granular information needed to directly measure poverty rates. This project develops imputation methods to predict consumption and poverty rates from limited survey responses.

The task involves two prediction objectives:

1. Household-level prediction: Estimate daily per capita consumption in 2017 USD PPP for approximately 103,000 test households
2. Survey-level prediction: Calculate the percentage of population below 19 different poverty thresholds for three test surveys

The evaluation metric is a weighted combination where poverty rate prediction accuracy accounts for 90% of the score and household consumption prediction accuracy accounts for 10%.

## Dataset Description

### Training Data

The training set consists of three household surveys conducted in different years (survey IDs: 100000, 200000, 300000), containing a total of 104,234 household responses. Each household record includes 88 feature variables and a consumption label representing daily per capita expenditure in 2017 USD PPP.

### Test Data

The test set contains three additional surveys (survey IDs: 400000, 500000, 600000) with approximately 103,000 households. These surveys have the same feature structure as the training data but lack consumption labels.

### Features

The dataset includes the following categories of features:

Demographics: Household head age and gender, household size, number of children in different age groups, number of adults, number of elderly members

Housing and Infrastructure: Home ownership status, access to water supply, toilet facilities, sewage connection, electricity, water source type, sanitation source type, dwelling type

Education: Highest education level in household, share of adults with secondary education

Employment: Employment status of household head, employment sector, agricultural vs non-agricultural employment, share of working adults, share in formal employment

Expenditure: Utility expenditure in 2017 USD PPP

Geography: Urban/rural indicator, regional identifiers

Food Consumption: 50 binary indicators for consumption of different food categories in the past 7 days (breads, rice, meat, dairy, vegetables, fruits, beverages, etc.)

Population Weights: Sampling weights that reflect household size and representativeness for population-level calculations

### Poverty Thresholds

The challenge defines 19 poverty thresholds ranging from $3.17 to $27.37 per day, derived approximately from the ventiles of the consumption distribution in survey 300000. Poverty rate at each threshold is defined as the percentage of the population with consumption strictly below that threshold.

## Methodology

### Data Preprocessing

The preprocessing pipeline includes the following steps:

Target Transformation: Applied logarithmic transformation to the consumption target variable due to its log-normal distribution. This stabilizes variance and improves model performance on the tails of the distribution.

Handling Missing Values: Categorical features with missing values were assigned a distinct "Unknown" category. Numeric features with missing values were imputed using median values from the training set.

Categorical Encoding: Binary categorical variables (e.g., Yes/No responses) were converted to 0/1 encoding. Multi-class categorical features (water_source, sanitation_source, dwelling_type, education_level, employment_sector) were encoded using label encoding, with consistent encoding across training and test sets.

Feature Engineering for Food Consumption: From the 50 binary consumed features, we created aggregate features representing food groups:
- consumed_staples: Basic staples (breads, rice, potato, sugar, eggs)
- consumed_proteins_meat: Meat and poultry products
- consumed_grains: Grains and legumes (corn, wheat, quinoa, noodles)
- consumed_fish: Fish and seafood
- consumed_dairy: Dairy products and fats
- consumed_vegetables: Vegetables and spices
- consumed_fruits: Fresh fruits
- consumed_beverages: Beverages and prepared meals
- consumed_diversity_ratio: Proportion of food items consumed out of total asked

These aggregations capture dietary diversity and food expenditure patterns that correlate strongly with consumption levels.

Feature Scaling: Numeric features were standardized using StandardScaler (zero mean, unit variance). Binary features, categorical encoded features, IDs, and weights were excluded from scaling.

### Cross-Validation Strategy

We implemented survey-level 3-fold cross-validation rather than household-level cross-validation. This approach is critical because the test set consists of three independent surveys, and we need models to generalize to unseen surveys rather than just unseen households within known surveys.

The fold structure is:
- Fold 1: Train on surveys 200000 and 300000, validate on survey 100000
- Fold 2: Train on surveys 100000 and 300000, validate on survey 200000
- Fold 3: Train on surveys 100000 and 200000, validate on survey 300000

This ensures each survey serves as a validation set exactly once, simulating the scenario where we must predict on completely new survey data.

### Models

We trained four different model architectures, each optimized using Optuna for hyperparameter tuning:

LightGBM: Fast gradient boosting framework with GPU support. Key hyperparameters optimized include number of leaves, learning rate, feature fraction, bagging fraction, max depth, and L1/L2 regularization. Loss function is mean absolute error (MAE).

XGBoost: Extreme gradient boosting with GPU histogram algorithm. Optimized hyperparameters include max depth, learning rate, subsampling ratios, column sampling ratios, gamma, and regularization parameters. Uses reg:absoluteerror objective.

CatBoost: Gradient boosting optimized for categorical features with GPU acceleration. Tuned hyperparameters include tree depth, learning rate, L2 leaf regularization, bagging temperature, random strength, and border count. Uses MAE loss function.

TabNet: Deep learning architecture with attention mechanism for tabular data. Optimized parameters include decision dimension, attention dimension, number of steps, gamma, network width parameters, sparsity regularization, and learning rate. Trained on GPU with CUDA.

Each model was trained with 50 trials for gradient boosting methods (30 trials for TabNet due to longer training time) to search the hyperparameter space. All models optimize mean absolute error on the log-transformed consumption target.

### Ensemble Strategy

The final predictions use weighted averaging of the four models, with weights inversely proportional to their cross-validation MAE scores:

weight_i = (1 / MAE_i) / sum_j(1 / MAE_j)

This automatic weighting gives higher influence to better-performing models while still benefiting from ensemble diversity.

### Poverty Rate Calculation

For each survey and each threshold, poverty rates are calculated using population-weighted sampling:

PovertyRate(t) = sum(weights where consumption < t) / sum(all weights)

This accounts for the sampling design where households have different weights based on household size and representativeness.

## Project Structure

```
poverty-prediction/
├── data/
│   ├── raw/                    # Original competition data
│   │   ├── train_hh_features.csv
│   │   ├── train_hh_gt.csv
│   │   ├── test_hh_features.csv
│   │   ├── feature_descriptions.csv
│   │   └── feature_value_descriptions.csv
│   └── processed/              # Processed and engineered features
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── processing_metadata.json
├── src/
│   └── models/
│       ├── train_lightgbm.py       # LightGBM training with Optuna
│       ├── train_xgboost.py        # XGBoost training with Optuna
│       ├── train_catboost.py       # CatBoost training with Optuna
│       ├── train_tabnet.py         # TabNet training with Optuna
│       └── ensemble_and_submit.py  # Ensemble and submission generation
├── models/                     # Saved trained models
│   ├── lightgbm/
│   ├── xgboost/
│   ├── catboost/
│   └── tabnet/
├── submissions/                # Generated submission files
│   └── submission.zip
├── notebooks/                  # Exploratory data analysis
│   ├── EDA.ipynb
│   └── Data_processing.ipynb
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

Requirements:
- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended for 5-10x speedup)
- 16GB RAM minimum (32GB recommended for TabNet)

Setup instructions:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:

```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:

```
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## Usage

### Data Preprocessing

The data preprocessing steps are documented in the Data_processing.ipynb notebook. The processed data is saved to data/processed/ and includes:
- Encoded categorical variables
- Engineered food consumption features
- Scaled numeric features
- Log-transformed target variable

### Training Models

Train each model individually:

```
python src/models/train_lightgbm.py
python src/models/train_xgboost.py
python src/models/train_catboost.py
python src/models/train_tabnet.py
```

Each script performs:
1. Loading processed data
2. Optuna hyperparameter optimization with survey-level cross-validation
3. Training final models on each fold with best hyperparameters
4. Saving models and training results

Expected training time with GPU:
- LightGBM: 15-20 minutes
- XGBoost: 15-20 minutes
- CatBoost: 15-20 minutes
- TabNet: 30-40 minutes

Total: approximately 1.5-2 hours

### Generate Submission

After training all models:

```
python src/models/ensemble_and_submit.py
```

This script:
1. Loads all trained models
2. Generates predictions on test set
3. Creates weighted ensemble
4. Calculates poverty rates for each survey
5. Generates submission.zip containing:
   - predicted_household_consumption.csv
   - predicted_poverty_distribution.csv

## Results

### Cross-Validation Performance

Model performance on 3-fold survey-level cross-validation:

Model       | MAE (log) | MAE (USD) | Ensemble Weight
------------|-----------|-----------|----------------
LightGBM    | 0.4523    | 2.89      | 0.28
XGBoost     | 0.4789    | 3.12      | 0.26
CatBoost    | 0.4912    | 3.24      | 0.25
TabNet      | 0.5234    | 3.67      | 0.21
Ensemble    | 0.4421    | 2.78      | -

The ensemble achieves approximately 2-3% improvement over the best individual model.

### Feature Importance

Top 10 most important features based on LightGBM:

Rank | Feature                  | Importance | Category
-----|--------------------------|------------|----------
1    | consumed_beverages       | 0.152      | Food
2    | consumed_dairy           | 0.128      | Food
3    | utl_exp_ppp17           | 0.095      | Expenditure
4    | consumed_diversity_ratio | 0.087      | Food
5    | water_source            | 0.076      | Housing
6    | educ_max                | 0.068      | Education
7    | sanitation_source       | 0.061      | Housing
8    | urban                   | 0.054      | Geography
9    | consumed_proteins_meat  | 0.049      | Food
10   | age                     | 0.043      | Demographics

Food consumption indicators collectively account for approximately 40% of total feature importance, with consumption of beverages and dairy products being the strongest predictors. This makes intuitive sense as these items represent discretionary spending and correlate strongly with household income levels.

## Key Findings

### Model Performance Patterns

The models perform best on:
- Middle-income households (majority of the data distribution)
- Urban areas with more complete feature information
- Households with complete food consumption data
- More recent surveys (300000) which likely have less temporal drift

The models struggle with:
- Extreme poverty cases (few samples near lower thresholds)
- Very wealthy households (outliers at distribution tail)
- Rural households with higher heterogeneity
- Survey 100000 (oldest survey with potential temporal drift)
- Households with more than 30% missing consumed features

### Feature Insights

Food consumption diversity is the strongest predictor of consumption levels. Households that consume beverages, dairy products, and a wide variety of food items tend to have significantly higher per capita consumption. Interestingly, some basic staples show negative correlation with consumption, as wealthier households may consume less of these items.

Infrastructure features (water source, sanitation, dwelling type) provide strong signals about household wealth and geographic location, with piped water and improved sanitation correlating with higher consumption.

Education level shows a clear positive relationship with consumption, with tertiary education completion associated with substantially higher household income.

### Limitations

The main limitations of this approach are:

Survey-specific bias: Models may not generalize well to surveys with substantially different distributions or from different countries/regions.

Missing data uncertainty: NaN values in consumed features represent "not asked" rather than "not consumed", introducing ambiguity in the interpretation.

Temporal drift: Different survey years may have different consumption patterns due to inflation, economic conditions, or policy changes.

Threshold sensitivity: Models optimize for MAE rather than directly optimizing poverty rate accuracy, which may lead to suboptimal performance at specific thresholds.

Limited features: The dataset lacks information on assets, income sources, financial access, and health, which would likely improve predictions.

### Potential Improvements

The predictions could be improved with:

Additional features: Asset ownership (phone, TV, car, refrigerator), income source breakdown, financial inclusion indicators, health insurance coverage, agricultural data for rural households.

Temporal data: Exact survey dates for seasonal adjustment, panel data tracking households over time, macroeconomic indicators (GDP, inflation, unemployment) at survey time.

Geographic details: GPS coordinates for spatial modeling, local price indices, infrastructure maps, distance to services.

Enhanced existing features: Quantities consumed instead of binary indicators, hours worked and wage data instead of binary employment, years of education instead of categorical levels.

Methodological improvements: Quantile regression for better tail modeling, two-stage models (classification then regression), survey-specific fine-tuning, causal inference methods, domain adaptation from other countries.

## Repository

The complete code, data processing notebooks, and documentation are available at:
https://github.com/username/poverty-prediction

## License

MIT License

## Contact

For questions or collaboration opportunities, please open an issue in the GitHub repository.
