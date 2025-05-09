import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

model = LinearRegression()
model.fit(train_X, train_y)

train_predictions = model.predict(train_X)
val_predictions = model.predict(val_X)

print(f"Train R2: {r2_score(train_y, train_predictions):.3f}")
print(f"Train MAE: {mean_absolute_error(train_y, train_predictions):.3f}")
print(f"Valid R2: {r2_score(val_y, val_predictions):.3f}")
print(f"Valid MAE: {mean_absolute_error(val_y, val_predictions):.3f}")

submission_prediction = model.predict(preprocessed_submission_data)

output = pd.DataFrame({'Id': submission_data.Id, 'SalePrice': submission_prediction})
output.to_csv('./submission.csv', index=False)
output.head()
print("submission.csv generated!!!")