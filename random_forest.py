import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

all_train_data = pd.read_csv('./train.csv')
submission_data = pd.read_csv('./test.csv')
pd.set_option('display.max_columns', None)
print(all_train_data.columns.values)
print(all_train_data[["Id", "SalePrice"]].head())

all_train_dummies = pd.get_dummies(all_train_data).fillna(0)
all_features = list(all_train_dummies.columns.values)

f, ax = plt.subplots(figsize=(25, 22))
corr = all_train_dummies.corr()
sns.heatmap(corr,
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True, ax=ax)
f.savefig('./random_forest/corr.png')

sale_price_pos = all_features.index('SalePrice')
features_with_high_corr = [all_features[i] for i in range(len(all_features)) if abs(corr.iloc[i, sale_price_pos]) > 0.5 and i != sale_price_pos]
print(features_with_high_corr)

y = all_train_dummies.SalePrice
X = all_train_dummies[features_with_high_corr]
submission_data_preprocessed = pd.get_dummies(submission_data.fillna(0))[features_with_high_corr]
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

from itertools import product
n_estimators_options = list(range(150, 351, 50))
max_depth_options = list(range(4, 13, 2))
min_samples_split_options = list(range(2, 12, 2))
min_samples_leaf_options = list(range(1, 6, 1))
criterion_options = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

permutations = list(product(
    n_estimators_options, 
    max_depth_options, 
    min_samples_split_options,
    min_samples_leaf_options,
    criterion_options
))

total_permutations = len(permutations)
print(f"total permutations: {total_permutations}")

best_options = dict(n_estimators=300, max_depth=6)
best_distance = 1e99
for i, (
    n_estimators, 
    max_depth, 
    min_samples_split,
    min_samples_leaf,
    criterion
    ) in enumerate(permutations):

    # não precisamos mais fazer a busca exaustiva, achamos uma configuração boa
    break
    options = dict(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )

    model = RandomForestRegressor(**options, random_state=1)
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    val_predictions = model.predict(val_X)
    

    train_r2 = r2_score(train_predictions, train_y)
    train_mae = mean_absolute_error(train_predictions, train_y)
    val_r2 = r2_score(val_predictions, val_y)
    val_mae = mean_absolute_error(val_predictions, val_y)

    # a = (|train_r2 - val_r2|, val_r2)
    # b = (0, 1)
    # here 'b' represents our best scenario, no overfit, train and validation has the same
    # r2 score, and validation has the best validation possible wich is 1
    # so we will take de options that better aproximate the model to 'b'

    # distance from a to b
    distance = ((train_r2 - val_r2) ** 2 + (val_r2 - 1) ** 2) ** (1/2)
    if distance < best_distance:
        best_distance = distance
        best_options = options

    print(f"|{i + 1: >5d}º of {total_permutations}|: {options} {train_r2:.3f} {val_r2:.3f} {train_mae:.1f} {val_mae:.1f}")

print(best_options)

best_model = RandomForestRegressor(**best_options, random_state=1)
best_model.fit(train_X, train_y)

train_predictions = best_model.predict(train_X)
val_predictions = best_model.predict(val_X)

# metrics
train_mae = mean_absolute_error(train_y, train_predictions)
train_r2 = r2_score(train_y, train_predictions)
val_mae = mean_absolute_error(val_y, val_predictions)
val_r2 = r2_score(val_y, val_predictions)

print(f"Train R2: {train_r2:.3f}")
print(f"Train MAE: {train_mae:.3f}")
print(f"Valid R2: {val_r2:.3f}")
print(f"Valid MAE: {val_mae:.3f}")

submission_prediction = best_model.predict(submission_data_preprocessed)

output = pd.DataFrame({'Id': submission_data.Id, 'SalePrice': submission_prediction})
output.to_csv('./submission.csv', index=False)
output.head()
print("submission.csv generated!!!")