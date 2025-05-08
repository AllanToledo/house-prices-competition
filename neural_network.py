import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class HousePriceNet(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class HousePriceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

all_train_data = pd.read_csv('./train.csv')
submission_data = pd.read_csv('./test.csv')
pd.set_option('display.max_columns', None)
print(all_train_data.columns.values)
print(all_train_data[["Id", "SalePrice"]].head())

all_train_dummies = pd.get_dummies(all_train_data).fillna(0)
all_features = list(all_train_dummies.columns.values)

f, ax = plt.subplots(figsize=(25, 22))
corr = all_train_dummies.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), vmin=-1.0, vmax=1.0, square=True, ax=ax)
f.savefig('./neural_network/corr.png')

sale_price_pos = all_features.index('SalePrice')
features_with_high_corr = [all_features[i] for i in range(len(all_features)) if abs(corr.iloc[i, sale_price_pos]) > 0.3 and i != sale_price_pos]
print("Features selecionadas:", features_with_high_corr)
print("Número de features:", len(features_with_high_corr))

y = all_train_dummies.SalePrice
X = all_train_dummies[features_with_high_corr]
submission_data_preprocessed = pd.get_dummies(submission_data.fillna(0))[features_with_high_corr]

print("\nValores ausentes em X:", X.isnull().sum().sum())
print("Valores ausentes em y:", y.isnull().sum())

Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
X = X[outlier_mask]
y = y[outlier_mask]

print("\nTamanho do dataset após remoção de outliers:", len(y))

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42, test_size=0.2)

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)
submission_data_scaled = scaler.transform(submission_data_preprocessed)

train_y_log = np.log1p(train_y)
val_y_log = np.log1p(val_y)

y_scaler = StandardScaler()
train_y_scaled = y_scaler.fit_transform(train_y_log.values.reshape(-1, 1)).ravel()
val_y_scaled = y_scaler.transform(val_y_log.values.reshape(-1, 1)).ravel()

train_dataset = HousePriceDataset(train_X_scaled, train_y_scaled)
val_dataset = HousePriceDataset(val_X_scaled, val_y_scaled)
submission_dataset = HousePriceDataset(submission_data_scaled)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
submission_loader = DataLoader(submission_dataset, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

model = HousePriceNet(input_dim=train_X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
n_epochs = 150

best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_loss += criterion(y_pred.squeeze(), y_batch).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f'Early stopping na época {epoch+1}')
        break
    
    if (epoch + 1) % 10 == 0:
        print(f'Época [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

model.eval()
train_predictions = []
val_predictions = []
train_true = []
val_true = []

with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)
        pred_denorm = y_scaler.inverse_transform(pred.cpu().numpy())
        pred_final = np.expm1(pred_denorm)
        train_predictions.extend(pred_final.flatten())
        train_true.extend(y_batch.cpu().numpy().flatten())
    
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)
        pred_denorm = y_scaler.inverse_transform(pred.cpu().numpy())
        pred_final = np.expm1(pred_denorm)
        val_predictions.extend(pred_final.flatten())
        val_true.extend(y_batch.cpu().numpy().flatten())

train_predictions = np.array(train_predictions)
val_predictions = np.array(val_predictions)
train_true = np.array(train_true)
val_true = np.array(val_true)

train_true = np.expm1(y_scaler.inverse_transform(train_true.reshape(-1, 1))).flatten()
val_true = np.expm1(y_scaler.inverse_transform(val_true.reshape(-1, 1))).flatten()

train_mae = mean_absolute_error(train_true, train_predictions)
train_r2 = r2_score(train_true, train_predictions)
val_mae = mean_absolute_error(val_true, val_predictions)
val_r2 = r2_score(val_true, val_predictions)

print(f"\nMétricas finais:")
print(f"Train MAE: ${train_mae:.2f} | Train R2: {train_r2:.3f}")
print(f"Val MAE: ${val_mae:.2f} | Val R2: {val_r2:.3f}")

model.eval()
submission_predictions = []

with torch.no_grad():
    for X_batch in submission_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch)
        pred_denorm = y_scaler.inverse_transform(pred.cpu().numpy())
        pred_final = np.expm1(pred_denorm)
        submission_predictions.extend(pred_final)

submission_prediction = np.array(submission_predictions).flatten()
output = pd.DataFrame({'Id': submission_data.Id, 'SalePrice': submission_prediction})
output.to_csv('./submission_nn.csv', index=False)
print("submission_nn.csv generated!!!") 
