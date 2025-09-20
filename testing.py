import torch
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, idx, xnum):
        out = self.base(idx, xnum)
        return out.unsqueeze(1)   # [B] → [B,1]

device = "cuda" if torch.cuda.is_available() else "cpu"

X = pd.read_parquet("encoded.parquet")
y = pd.read_csv("./data/Ad_table (extra).csv")["Price"].loc[X.index]

le = LabelEncoder()
X["Genmodel_ID_enc"] = le.fit_transform(X["Genmodel_ID"])

num_cols = X.select_dtypes("number").columns.tolist()
num_cols = [c for c in num_cols if c != "Genmodel_ID_enc"]

ss = StandardScaler()
X[num_cols] = ss.fit_transform(X[num_cols])

X_id = torch.LongTensor(X["Genmodel_ID_enc"].values)
X_num = torch.FloatTensor(X[num_cols].values)

model = TabRes(
    num_emb=X["Genmodel_ID"].nunique(),
    emb_dim=48,
    in_dim=X_num.shape[1],
    depth=4,
    hidden=128
).to(device)

model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

torch.cuda.empty_cache()
def model_predict(data):
    idx = torch.LongTensor(data[:, 0]).to(device)
    xnum = torch.FloatTensor(data[:, 1:]).to(device)
    with torch.no_grad():
        out = model(idx, xnum)
    return out.cpu().numpy()

background = np.hstack([
    X_id[:20].cpu().numpy().reshape(-1, 1),
    X_num[:20].cpu().numpy()
])

X_sample = np.hstack([
    X_id[:50].cpu().numpy().reshape(-1, 1),
    X_num[:50].cpu().numpy()
])

explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(X_sample)

feature_names = ["Genmodel_ID_enc"] + num_cols
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
