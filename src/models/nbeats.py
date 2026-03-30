"""
N-BEATS模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

from .base import BaseModel


class NBeatsBlock(nn.Module):
    """N-BEATS基本块"""

    def __init__(self, input_dim, output_dim, theta_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.theta_dim = theta_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, theta_dim)
        self.bn3 = nn.BatchNorm1d(theta_dim)
        self.fc4 = nn.Linear(theta_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        theta = self.fc3(x)
        theta = self.bn3(theta)

        y = self.fc4(theta)
        return y


class NBeatsNet(nn.Module):
    """N-BEATS网络"""

    def __init__(self, window_size, output_steps, hidden_dim=128, num_blocks=4, dropout=0.1):
        super().__init__()
        self.window_size = window_size

        self.blocks = nn.ModuleList([
            NBeatsBlock(window_size, output_steps, theta_dim=8,
                        hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: (batch, window_size)
        residuals = x
        forecasts = 0

        for block in self.blocks:
            forecast = block(residuals)
            forecasts = forecasts + forecast
            residuals = residuals - forecast

        return forecasts


class NBeatsModel(BaseModel):
    """N-BEATS模型"""

    def __init__(self, config: dict):
        super().__init__('N-BEATS', config)

        self.window_size = config.get('window_size', 6)
        self.output_steps = config.get('output_steps', 1)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_blocks = config.get('num_blocks', 4)
        self.dropout = config.get('dropout', 0.1)
        self.epochs = config.get('epochs', 200)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.verbose = config.get('verbose', 1)
        self.random_seed = config.get('random_seed', 42)

        torch.manual_seed(self.random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        print(f"[N-BEATS] window_size={self.window_size}, hidden_dim={self.hidden_dim}")

    def _build_model(self):
        self.model = NBeatsNet(
            window_size=self.window_size,
            output_steps=self.output_steps,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout
        ).to(self.device)

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray):
        # 更新 window_size 以匹配输入
        if X_train.shape[1] != self.window_size:
            print(f"[N-BEATS] 更新 window_size: {self.window_size} -> {X_train.shape[1]}")
            self.window_size = X_train.shape[1]

        # 划分验证集
        val_size = int(len(X_train) * 0.1)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]

        # 构建模型
        self._build_model()

        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).reshape(-1, 1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).reshape(-1, 1)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item() * len(X_batch)

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 50 == 0 and self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 加载最佳权重
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        return y_pred.cpu().numpy().flatten()

    def save(self, path: str):
        import os
        if self.model is None:
            raise ValueError("模型未训练")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not path.endswith('.pt'):
            path = path.replace('.pkl', '.pt')
        torch.save(self.model.state_dict(), path)
        print(f"[{self.name}] 模型已保存: {path}")

    def load(self, path: str):
        self._build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_fitted = True
        print(f"[{self.name}] 模型已加载: {path}")