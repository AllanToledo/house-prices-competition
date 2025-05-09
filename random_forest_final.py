import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

all_train_data = pd.read_csv('./train.csv')
submission_data = pd.read_csv('./test.csv')
all_train_dummies = pd.get_dummies(all_train_data).fillna(0)
all_features = list(all_train_dummies.columns.values)

corr = all_train_dummies.corr()

sale_price_pos = all_features.index('SalePrice')
features_with_high_corr = [
    all_features[i] for i in range(len(all_features)) 
    if abs(corr.iloc[i, sale_price_pos]) > 0.5 and i != sale_price_pos
]

y = all_train_dummies.SalePrice
X = all_train_dummies[features_with_high_corr]
preprocessed_submission_data = pd.get_dummies(submission_data.fillna(0))[features_with_high_corr]
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

best_hyperparams = None
best_distance = 1e99
combinations = list(product(
    list(range(150, 351, 50)),  #n_estimators_options
    list(range(4, 13, 2)),      #max_depth_options
    list(range(2, 12, 2)),      #min_samples_split_options
    list(range(1, 6, 1)),       #min_samples_leaf_options
    ["squared_error", "absolute_error", "friedman_mse", "poisson"] #criterion_options
))

for combination in combinations:
    hyperparams = dict(n_estimators=combination[0], 
        max_depth=combination[1],
        min_samples_split=combination[2],
        min_samples_leaf=combination[3],
        criterion=combination[4])

    model = RandomForestRegressor(**hyperparams, random_state=1, n_jobs=-1)
    model.fit(train_X, train_y)
    train_r2 = r2_score(model.predict(train_X), train_y)
    val_r2 = r2_score(model.predict(val_X), val_y)

    distance = ((train_r2 - val_r2) ** 2 + (val_r2 - 1) ** 2) ** (1/2)
    if distance < best_distance:
        best_distance = distance
        best_hyperparams = hyperparams

best_model = RandomForestRegressor(**best_hyperparams, random_state=1)
best_model.fit(train_X, train_y)

train_predictions = best_model.predict(train_X)
val_predictions = best_model.predict(val_X)

print(f"Train R2: {r2_score(train_y, train_predictions):.3f}")
print(f"Train MAE: {mean_absolute_error(train_y, train_predictions):.3f}")
print(f"Valid R2: {r2_score(val_y, val_predictions):.3f}")
print(f"Valid MAE: {mean_absolute_error(val_y, val_predictions):.3f}")

submission_prediction = best_model.predict(preprocessed_submission_data)
output = pd.DataFrame({'Id': submission_data.Id, 'SalePrice': submission_prediction})
output.to_csv('./submission.csv', index=False)
output.head()
print("submission.csv generated!!!")