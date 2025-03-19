import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np

# Load the dataset
# original_data = pd.read_csv('./csvs/london_merged.csv')
# data = original_data.sample(n=4000)
#
# # Feature Extraction {timestamp} into {year, month, day, hour}
# feature ='timestamp'
#
# data['datetime_column'] = pd.to_datetime(data['timestamp'])
# data['year'] = data['datetime_column'].dt.year
# data['month'] = data['datetime_column'].dt.month
# data['day'] = data['datetime_column'].dt.day
# data['hour'] = data['datetime_column'].dt.hour
# data.drop(columns=['datetime_column'], inplace=True)
# # Separate target and prediction features
# unused_col = 'timestamp'
target_col = 'cnt'
# data = data.drop(unused_col, axis=1)

# data.to_csv('./csvs/Clean_london_merged.csv')

# 600 samples from original
print("600 samples from original")

data = pd.read_csv('../csvs/Clean_london_merged_smaller.csv')

X = data.drop(target_col, axis=1)
y = data[target_col]

# Apply train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Define model
rf = RandomForestRegressor(random_state=42)

# # Define hyperparameter search space
# param_dist = {
#     "n_estimators": [100, 200, 300, 400, 500],
#     "max_depth": [10, 20, 30, 40, 50, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["auto", "sqrt", "log2"],
#     "bootstrap": [True, False]
# }
#
# # Random search with 5-fold cross-validation
# random_search = RandomizedSearchCV(estimator=rf,param_distributions=param_dist,
#     n_iter=20,cv=5,verbose=2,n_jobs=-1,random_state=42)

# Fit the model
# random_search.fit(X_train, y_train)

# Best hyperparameters
# print("Best Parameters:", random_search.best_params_)

rf = RandomForestRegressor(
    n_estimators= 300,
    min_samples_split= 2,
    min_samples_leaf= 1,
    max_features= 'sqrt',
    max_depth= None,
    bootstrap= True
)
rf.fit(X_train, y_train.values.ravel())

y_pred = rf.predict(X_test)

# Calculate the score/error
print("R2 score:", r2_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))

print("----")

## -----------------------

target_col = 'cnt'
# 1200 samples from original
print("1200 samples from original")

data = pd.read_csv('../csvs/Clean_london_merged_1200.csv')
X = data.drop(target_col, axis=1)
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForestRegressor(random_state=42)

param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, 40, 50, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=20, cv=5, verbose=2, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Parameters:", best_params)

best_rf = RandomForestRegressor(**best_params)
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

print("R2 score:", r2_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))
print("---------")
## -----------------------

target_col = 'cnt'
# 600 samples from original + 600 synthetic
print("600 samples from original + 600 synthetic")

data = pd.read_csv('../csvs/Clean_london_merged_smaller_vae_concate_smaller.csv')
X = data.drop(target_col, axis=1)
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForestRegressor(random_state=42)

param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, 40, 50, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=20, cv=5, verbose=2, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Parameters:", best_params)

best_rf = RandomForestRegressor(**best_params)
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

print("R2 score:", r2_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))

