"""
pipeline.models.paper_model
============================
Research Paper Architecture: MobileNetV3-Small
Paper: Howard et al. (2019) "Searching for MobileNetV3", ICCV 2019.
"""
from __future__ import annotations
import numpy as np, sys, os
import torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pipeline.optimizers import EarlyStopping


class _FallbackCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4,256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.net(x)


class PaperModel:
    """
    MobileNetV3-Small (Howard et al., ICCV 2019) via PyTorch.
    Falls back to a 3-layer CNN if torchvision is unavailable.
    Dataset loading/preprocessing always done via minicv before entering here.
    """
    def __init__(self, n_classes=6, lr=1e-3):
        self.n_classes = n_classes
        self.device    = torch.device("cpu")
        try:
            import torchvision.models as tv
            self.model = tv.mobilenet_v3_small(weights=None)
            in_f = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_f, n_classes)
            self.arch_name = "MobileNetV3-Small"
        except Exception:
            self.model = _FallbackCNN(n_classes)
            self.arch_name = "FallbackCNN"
        self.model     = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5)
        self.criterion = nn.CrossEntropyLoss()
        self.lr        = lr

    def _to_tensor(self, X):
        X_norm = X / 255.0 if X.max() > 1.1 else X.copy()
        return torch.from_numpy(X_norm.transpose(0,3,1,2)).float().to(self.device)

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=30, batch_size=32, patience=10, verbose=True):
        early_stop = EarlyStopping(patience=patience)
        history    = {k:[] for k in ("train_loss","val_loss","train_acc","val_acc","lr")}
        N   = len(y_train)
        rng = np.random.default_rng(42)

        for epoch in range(1, epochs+1):
            self.model.train()
            idx = rng.permutation(N)
            el, ec = 0.0, 0
            for s in range(0, N, batch_size):
                bi  = idx[s:s+batch_size]
                Xb  = self._to_tensor(X_train[bi])
                yb  = torch.from_numpy(y_train[bi]).long().to(self.device)
                self.optimizer.zero_grad()
                out  = self.model(Xb)
                loss = self.criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                el += loss.item()*len(bi); ec += (out.argmax(1)==yb).sum().item()
            tl, ta = el/N, ec/N

            self.model.eval()
            with torch.no_grad():
                vo  = self.model(self._to_tensor(X_val))
                yt  = torch.from_numpy(y_val).long().to(self.device)
                vl  = self.criterion(vo, yt).item()
                va  = (vo.argmax(1)==yt).float().mean().item()
            self.scheduler.step(vl)

            cur_lr = self.optimizer.param_groups[0]['lr']
            history["train_loss"].append(tl); history["val_loss"].append(vl)
            history["train_acc"].append(ta);  history["val_acc"].append(va)
            history["lr"].append(cur_lr)
            if verbose:
                print(f"    Epoch {epoch:>3d}/{epochs} | loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} val_acc={va:.4f}")
            if early_stop.update(vl, epoch):
                if verbose: print(f"    Early stop (best={early_stop.best_epoch})")
                break
        return history

    def predict(self, X, batch_size=64):
        self.model.eval(); preds=[]
        with torch.no_grad():
            for i in range(0,len(X),batch_size):
                preds.append(self.model(self._to_tensor(X[i:i+batch_size])).argmax(1).cpu().numpy())
        return np.concatenate(preds).astype(np.int64)

    def predict_proba(self, X, batch_size=64):
        self.model.eval(); probs=[]; sm=nn.Softmax(dim=1)
        with torch.no_grad():
            for i in range(0,len(X),batch_size):
                probs.append(sm(self.model(self._to_tensor(X[i:i+batch_size]))).cpu().numpy())
        return np.concatenate(probs)

    def save(self, path):
        torch.save({"model_state":self.model.state_dict(),
                    "optimizer_state":self.optimizer.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
