import pandas as pd
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             root_mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
path = Path("./data")

target = "Price"
X = pd.read_parquet("encoded.parquet")
y = pd.read_csv(path / "Ad_table (extra).csv")[target].loc[X.index]


class ResBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.d = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        h = self.d(F.silu(self.fc1(x)))
        h = self.fc2(h)
        return F.silu(x + h)

class TabRes(nn.Module):
    def __init__(self, num_emb, emb_dim, in_dim, hidden=192, depth=3):
        super().__init__()
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.blocks = nn.ModuleList([ResBlock(emb_dim+in_dim, hidden) for _ in range(depth)])
        self.out = nn.Linear(emb_dim + in_dim, 1)
    
    def forward(self, idx, xnum):
        # idx: [B] (genmodelid idx)
        # xnum: [B, in_dim] (num + ohe)
        e = self.emb(idx)
        x = torch.cat([e, xnum], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out(x).squeeze(-1) # reg out
    
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

le = LabelEncoder()
X["Genmodel_ID_enc"] = le.fit_transform(X["Genmodel_ID"])

X_train["Genmodel_ID_enc"] = le.transform(X_train["Genmodel_ID"])
X_val["Genmodel_ID_enc"] = le.transform(X_val["Genmodel_ID"])
X_test["Genmodel_ID_enc"] = le.transform(X_test["Genmodel_ID"])

num_cols = X.select_dtypes("number").columns.tolist()
num_cols = [c for c in num_cols if c != "Genmodel_ID_enc"]  

ss = StandardScaler()

X_train[num_cols] = ss.fit_transform(X_train[num_cols])
X_val[num_cols] = ss.transform(X_val[num_cols])
X_test[num_cols]  = ss.transform(X_test[num_cols])

ss_y = StandardScaler()
y_train = ss_y.fit_transform(np.log1p(y_train).values.reshape(-1,1))
y_val   = ss_y.transform(np.log1p(y_val).values.reshape(-1,1))
y_test  = ss_y.transform(np.log1p(y_test).values.reshape(-1,1))

X_train_id = torch.LongTensor(X_train["Genmodel_ID_enc"].values)
X_val_id   = torch.LongTensor(X_val["Genmodel_ID_enc"].values)
X_test_id  = torch.LongTensor(X_test["Genmodel_ID_enc"].values)

X_train_num = torch.FloatTensor(X_train[num_cols].values)
X_val_num   = torch.FloatTensor(X_val[num_cols].values)
X_test_num  = torch.FloatTensor(X_test[num_cols].values)

y_train_t = torch.FloatTensor(y_train)
y_val_t   = torch.FloatTensor(y_val)
y_test_t  = torch.FloatTensor(y_test)

train_ds = TensorDataset(X_train_id, X_train_num, y_train_t)
val_ds   = TensorDataset(X_val_id, X_val_num, y_val_t)
test_ds  = TensorDataset(X_test_id, X_test_num, y_test_t)

tr_l  = DataLoader(train_ds, batch_size=512, shuffle=True)
va_l  = DataLoader(val_ds, batch_size=512)
te_l  = DataLoader(test_ds, batch_size=512)

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_score = float("inf")
        self.counter = 0
    def __call__(self, loss):
        self.loss = loss
        if self.loss > self.best_score - self.delta:
            self.counter += 1
            if self.patience == self.counter:
                return True
        else:
            self.best_score = self.loss
            self.counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            return False
        return False

def train_model(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss, total = 0.0, 0
    
    for Xid, Xnum, labels in tqdm(dataloader):
        Xid, Xnum, labels = Xid.to(device), Xnum.to(device), labels.to(device)
        optimizer.zero_grad()
        with amp.autocast(device_type=device):
            outputs = model(Xid, Xnum)
            loss = criterion(outputs, labels.squeeze())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    epoch_loss = running_loss / total
    return epoch_loss
        
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, total = 0.0, 0
    with torch.no_grad():
        for Xid, Xnum, labels in tqdm(dataloader):
            Xid, Xnum, labels = Xid.to(device), Xnum.to(device), labels.to(device)
            outputs = model(Xid, Xnum)
            loss = criterion(outputs, labels.squeeze())
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    epoch_loss = running_loss / total
    return epoch_loss

def test_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for Xid, Xnum, labels in tqdm(dataloader):
            Xid, Xnum, labels = Xid.to(device), Xnum.to(device), labels.to(device)
            outputs = model(Xid, Xnum)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

model = TabRes(num_emb=X["Genmodel_ID"].nunique(), 
               emb_dim=48, in_dim=X_train_num.shape[1],
               depth=4, hidden=128).to(device)
epochs = 90
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = (optim.lr_scheduler
             .ReduceLROnPlateau(optimizer, patience=3, factor=0.5))

train_losses, val_losses = [], []
scaler = amp.GradScaler(device)
early_stopping = EarlyStopping(patience=8, delta=1e-5)

for epoch in range(epochs):
    torch.cuda.empty_cache()
    train_loss = train_model(model, tr_l, criterion, optimizer, scaler)
    val_loss = evaluate(model, va_l, criterion)
    scheduler.step(val_loss)
    if early_stopping(val_loss):
        print("Early stopping triggered!")
        break

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
model.load_state_dict(torch.load("best_model.pt"))
all_preds, all_labels = test_model(model, te_l)
y_pred, y_true = np.array(all_preds), np.array(all_labels)

y_true = ss_y.inverse_transform(np.array(all_labels).reshape(-1,1)).ravel()

y_pred = ss_y.inverse_transform(np.array(all_preds).reshape(-1,1)).ravel()
y_true = np.expm1(y_true)
y_pred = np.expm1(y_pred)
mse = mean_squared_error(y_true, y_pred)

rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")
print(f"MAPE: {mape*100:.2f}%")


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    )
score = smape(y_true, y_pred)
print(f"SMAPE: {score:.2f}%")

