# Bulldozer Sale Price Prediction

This project predicts the sale price of used bulldozers and other heavy equipment sold at auction. The data and problem was provided as part of a machine learning task to help build a Blue Book pricing model for the company Fast Iron.

##  Dataset Overview

The training dataset consists of 400,000+ rows and over 50 features related to:
- Machine and model specs
- Sales history and auction details
- Time of sale (with full date)
- Usage metrics like engine hours
- Categorical features such as product group, state, enclosure, etc.

The test set included 12,000+ rows for which we need to predict the `SalePrice`.

##  Steps Followed

### 1. Data Preprocessing

- Parsed the `saledate` column and extracted `saleYear` and `saleMonth`.
- Dropped columns with more than 80% missing values.
- For the remaining missing values:
  - Categorical columns were filled with `"None"`
  - Numerical columns were filled using the column median

### 2. Feature Encoding

Initially, I tried one-hot encoding but it caused memory crash on Colab due to high dimensionality. So instead, I used **Label Encoding** for all object columns to keep things light and efficient.

### 3. Model Building

- I used **RandomForestRegressor** from scikit-learn.
- The data was split into training and validation sets (80/20).
- After training, the model was evaluated using **RMSLE** (Root Mean Squared Log Error), since that was the metric required in the problem.

It gave a decent RMSLE score on the validation set.

### 4. Final Predictions

The final trained model was used to predict prices on the test dataset. Predictions were saved in a CSV file called `test_predictions.csv` in the required format:
