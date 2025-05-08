import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

all_train_data = pd.read_csv('./train.csv')
submission_data = pd.read_csv('./test.csv')
pd.set_option('display.max_columns', None)
print(all_train_data.columns.values)
print(all_train_data[["Id", "SalePrice"]].head())

all_train_dummies = pd.get_dummies(all_train_data).fillna(0)
all_features = list(all_train_dummies.columns.values)

corr = all_train_dummies.corr()

sale_price_pos = all_features.index('SalePrice')
features_with_high_corr = [all_features[i] for i in range(len(all_features)) if abs(corr.iloc[i, sale_price_pos]) > 0.5 and i != sale_price_pos]
print(features_with_high_corr)

y = all_train_dummies.SalePrice
X = all_train_dummies[features_with_high_corr]
submission_data_preprocessed = pd.get_dummies(submission_data.fillna(0))[features_with_high_corr]
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

lr_model = LinearRegression()
lr_model.fit(train_X, train_y)

train_predictions = lr_model.predict(train_X)
val_predictions = lr_model.predict(val_X)

# metrics
train_mae = mean_absolute_error(train_predictions, train_y)
train_r2_score = r2_score(train_predictions, train_y)
val_mae = mean_absolute_error(val_predictions, val_y)
val_r2_score = r2_score(val_predictions, val_y)

print(f"Train R2: {train_r2_score:.3f}")
print(f"Train MAE: {train_mae:.3f}")
print(f"Valid R2: {val_r2_score:.3f}")
print(f"Valid MAE: {val_mae:.3f}")

best_model = lr_model

if best_model != None:
    submission_prediction = best_model.predict(submission_data_preprocessed)
    
    output = pd.DataFrame({'Id': submission_data.Id, 'SalePrice': submission_prediction})
    output.to_csv('./submission.csv', index=False)
    output.head()
    print("submission.csv generated!!!")