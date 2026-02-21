from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .single_maxsim import single_maxsim
from .model import MLP


class Lemur:

    def __init__(self, index: Optional[Union[str, Path]] = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.index_path = Path(index) if index is not None else None
        self.final_hidden_dim: Optional[int] = None

    def fit(
        self,
        train: np.ndarray,
        train_counts: np.ndarray,
        learn: Optional[np.ndarray] = None,
        learn_counts: Optional[np.ndarray] = None,
        test: Optional[np.ndarray] = None,
        test_counts: Optional[np.ndarray] = None,
        train_subset_size: int = 8192,
        learn_subset_size: int = 100000,
        ols_sample_size: int = 16384,
        num_layers: int = 1,
        hidden_dim: int = 512,
        final_hidden_dim: Optional[int] = 2048,
        activation: str = "gelu",
        epochs: int = 100,
        lr: float = 3e-3,
        batch_size: int = 512,
        grad_clip: Optional[float] = 0.5,
        force_retrain: bool = False,
        verbose: bool = True,
    ) -> "Lemur":

        def validate_pair(
            name: str, matrix: Optional[np.ndarray], counts: Optional[np.ndarray]
        ) -> None:
            if (matrix is None) != (counts is None):
                raise ValueError(f"{name} and {name}_counts must be provided together.")
            if matrix is None:
                return
            if matrix.ndim != 2:
                raise ValueError(f"{name} must be a 2D numpy array.")
            if matrix.dtype != np.float32:
                raise ValueError(f"{name} must have dtype float32.")
            if counts is None or counts.ndim != 1:
                raise ValueError(f"{name}_counts must be a 1D numpy array.")
            if counts.dtype != np.int32:
                raise ValueError(f"{name}_counts must have dtype int32.")
            if np.any(counts < 0):
                raise ValueError(f"{name}_counts must be non-negative.")
            if counts.sum(dtype=np.int64) != matrix.shape[0]:
                raise ValueError(f"sum({name}_counts) must equal the number of rows in {name}.")

        validate_pair("train", train, train_counts)
        validate_pair("test", test, test_counts)
        validate_pair("learn", learn, learn_counts)
        if learn is None:
            learn = train
            learn_counts = train_counts

        self.train = train
        self.train_counts = train_counts
        self.learn = learn
        self.learn_counts = learn_counts
        self.test = test
        self.test_counts = test_counts

        cached_mlp = None
        if not force_retrain and self.index_path is not None:
            cached_path = self.index_path / "mlp.pt"
            if cached_path.exists():
                cached_mlp = self.load_mlp(cached_path)

        if cached_mlp is None:
            self.train_mlp(
                lr=lr,
                epochs=epochs,
                hidden_dim=hidden_dim,
                final_hidden_dim=final_hidden_dim,
                num_layers=num_layers,
                activation=activation,
                train_subset_size=train_subset_size,
                learn_subset_size=learn_subset_size,
                batch_size=batch_size,
                grad_clip=grad_clip,
                verbose=verbose,
            )

        cached_w = None
        if not force_retrain and self.index_path is not None and cached_mlp is not None:
            cached_path = self.index_path / "w.pt"
            if cached_path.exists():
                cached_w = self.load_w(cached_path)

        if cached_w is None:
            self.fit_corpus(
                sample_size=ols_sample_size,
                verbose=verbose,
            )

        return self

    def create_training_data(
        self,
        train_subset_size: int = 8192,
        learn_subset_size: int = 100000,
        block_bytes: int | None = 256 * 1024 * 1024,
    ):
        train_offsets = np.concatenate([[0], np.cumsum(self.train_counts)])

        train_subset_ix = np.random.choice(
            len(self.train_counts), size=train_subset_size, replace=False
        )
        if self.test is not None:
            test_subset_ix = np.random.choice(
                len(self.test), size=min(len(self.test), 32000), replace=False
            )

        def pick(i):
            return self.train[train_offsets[i] : train_offsets[i + 1]]

        device = self.device

        tmp_train = torch.tensor(np.vstack([pick(i) for i in train_subset_ix])).to(device)
        tmp_train_counts = torch.tensor(self.train_counts[train_subset_ix]).to(device)

        learn_subset_ix = np.random.choice(
            len(self.learn), size=min(len(self.learn), learn_subset_size), replace=False
        )
        learn_vectors = self.learn[learn_subset_ix]

        X_train = torch.tensor(learn_vectors).to(device)
        X_val = (
            torch.tensor(self.test[test_subset_ix]).to(device) if self.test is not None else None
        )

        y_train = single_maxsim(tmp_train, tmp_train_counts, X_train, block_bytes=block_bytes)
        y_val = (
            single_maxsim(tmp_train, tmp_train_counts, X_val, block_bytes=block_bytes)
            if X_val is not None
            else None
        )

        mean = y_train.mean()
        std = y_train.std(unbiased=False)
        y_train = (y_train - mean) / std
        if y_val is not None:
            y_val = (y_val - mean) / std
        self.mean = mean
        self.std = std

        return X_train, y_train, X_val, y_val

    def train_mlp(
        self,
        epochs: int = 120,
        batch_size: int = 1024,
        lr: float = 5e-3,
        hidden_dim: int = 1024,
        final_hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "gelu",
        train_subset_size: int = 8192,
        learn_subset_size: int = 100000,
        grad_clip: Optional[float] = None,
        verbose: bool = True,
    ) -> MLP:
        X_train, y_train, X_val, y_val = self.create_training_data(
            train_subset_size=train_subset_size,
            learn_subset_size=learn_subset_size,
        )

        device = self.device
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        if X_val is not None:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

        model = MLP(
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            hidden_dim=hidden_dim,
            final_hidden_dim=final_hidden_dim,
            num_layers=num_layers,
            activation=activation,
        ).to(device)
        self.final_hidden_dim = int(model.config["final_hidden_dim"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(X_train.shape[0], device=device)
            running_loss = 0.0
            num_batches = 0

            for start in range(0, X_train.shape[0], batch_size):
                idx = perm[start : start + batch_size]
                batch_x = X_train[idx]
                batch_y = y_train[idx]

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        preds = model(batch_x)
                        loss = loss_fn(preds, batch_y)
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(batch_x)
                    loss = loss_fn(preds, batch_y)
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                running_loss += loss.detach()
                num_batches += 1

            train_loss = (running_loss / max(1, num_batches)).item()

            if X_val is not None:
                model.eval()
                with torch.inference_mode():
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            val_preds = model(X_val)
                            val_loss = loss_fn(val_preds, y_val).item()
                    else:
                        val_preds = model(X_val)
                        val_loss = loss_fn(val_preds, y_val).item()
                metric_loss = val_loss
                if verbose:
                    print(
                        f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.6f} "
                        f"val_loss={val_loss:.6f}"
                    )
            else:
                metric_loss = train_loss
                if verbose:
                    print(f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.6f}")

            if metric_loss < best_loss:
                best_loss = metric_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        self.mlp = model
        if self.index_path is not None:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.save_mlp()
        return model

    def save_mlp(self, path: Optional[Union[str, Path]] = None) -> Path:
        if path is None:
            if self.index_path is None:
                raise ValueError("path is required when index_path is not set")
            self.index_path.mkdir(parents=True, exist_ok=True)
            path = self.index_path / "mlp.pt"
        else:
            path = Path(path)

        model = self.mlp
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("mlp config is missing; re-create the model with a config-aware MLP")

        payload = {"state_dict": model.state_dict(), "config": config}
        if hasattr(self, "mean") and hasattr(self, "std"):
            payload["output_mean"] = float(self.mean)
            payload["output_std"] = float(self.std)
        torch.save(payload, path)
        return path

    def load_mlp(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> MLP:
        if path is None:
            if self.index_path is None:
                raise ValueError("path is required when index_path is not set")
            path = self.index_path / "mlp.pt"
        else:
            path = Path(path)

        payload = torch.load(path, map_location=self.device)
        if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
            raise ValueError("mlp checkpoint must contain state_dict and config")

        config = dict(payload["config"])
        config.pop("normalize", None)
        config.pop("eps", None)
        config.pop("dropout", None)
        model = MLP(**config).to(self.device)
        self.final_hidden_dim = int(config["final_hidden_dim"])
        state = payload["state_dict"]
        model.load_state_dict(state)
        self.mlp = model
        if "output_mean" in payload and "output_std" in payload:
            self.mean = torch.tensor(payload["output_mean"], device=self.device)
            self.std = torch.tensor(payload["output_std"], device=self.device)
        else:
            self.mean = torch.tensor(0.0, device=self.device)
            self.std = torch.tensor(1.0, device=self.device)
        return model

    def save_w(self, path: Optional[Union[str, Path]] = None) -> Path:
        if path is None:
            if self.index_path is None:
                raise ValueError("path is required when index_path is not set")
            self.index_path.mkdir(parents=True, exist_ok=True)
            path = self.index_path / "w.pt"
        else:
            path = Path(path)

        if not hasattr(self, "W"):
            raise ValueError("W is not set; fit the corpus or load W first")

        torch.save({"W": self.W}, path)
        return path

    def load_w(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        if path is None:
            if self.index_path is None:
                raise ValueError("path is required when index_path is not set")
            path = self.index_path / "w.pt"
        else:
            path = Path(path)

        payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and "W" in payload:
            W = payload["W"]
        else:
            W = payload

        if not torch.is_tensor(W):
            raise ValueError("W checkpoint must contain a tensor or {'W': tensor}")

        self.W = W.to(self.device)
        return self.W

    def _compute_features_batched(
        self,
        data: torch.Tensor,
        batch_size: int = 2048,
    ) -> torch.Tensor:
        device = self.device
        self.mlp.to(device)
        data = data.to(device)
        feature_extractor = self.mlp.feature_extractor

        self.mlp.eval()
        with torch.inference_mode():
            num_rows = data.shape[0]
            if self.final_hidden_dim is None:
                raise ValueError("final_hidden_dim is not set; train or load the MLP first")
            outputs = torch.empty(
                (num_rows, self.final_hidden_dim),
                device=data.device,
                dtype=data.dtype,
            )
            for start in range(0, num_rows, batch_size):
                end = min(start + batch_size, num_rows)
                outputs[start:end] = feature_extractor(data[start:end])
            return outputs

    def fit_corpus(
        self,
        sample_size: int = 16384,
        verbose: bool = True,
    ):
        sample_ix = np.random.choice(len(self.learn), size=sample_size, replace=False)
        sampled = torch.from_numpy(self.learn[sample_ix])
        Z = self._compute_features_batched(sampled)

        device = Z.device
        sampled = sampled.to(device)

        num_segments = len(self.train_counts)
        W = torch.empty((num_segments, Z.shape[1]), device=device, dtype=torch.float32)

        train_tensor = torch.from_numpy(self.train)
        counts_tensor = torch.from_numpy(self.train_counts)
        train_offsets = np.concatenate([[0], np.cumsum(self.train_counts)])
        bytes_per_score = sampled.element_size()
        target_bytes = 16 * 1024 * 1024
        max_segments = target_bytes // (sampled.shape[0] * bytes_per_score)
        if max_segments < 1:
            max_segments = 1

        use_non_blocking = device.type == "cuda"
        with torch.inference_mode():
            U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
            tol = torch.finfo(S.dtype).eps * max(Z.shape) * S.max()
            scales = torch.where(S > tol, S.reciprocal(), torch.zeros_like(S))
            U_t = U.T
            Vh_t = Vh.T

            progress = (
                tqdm(total=num_segments, desc="Indexing documents", unit="doc")
                if verbose
                else nullcontext(None)
            )
            with progress as pbar:
                for seg_start in range(0, num_segments, max_segments):
                    seg_end = min(seg_start + max_segments, num_segments)
                    row_start = train_offsets[seg_start]
                    row_end = train_offsets[seg_end]

                    train_slice = train_tensor[row_start:row_end]
                    if device.type != "cpu":
                        train_slice = train_slice.to(device, non_blocking=use_non_blocking)
                    counts_slice = counts_tensor[seg_start:seg_end]
                    Y_batch = (
                        single_maxsim(train_slice, counts_slice, sampled) - self.mean
                    ) / self.std

                    UtY = U_t @ Y_batch
                    scaled = scales[:, None] * UtY
                    W[seg_start:seg_end] = (Vh_t @ scaled).T.to(W.dtype)
                    if pbar is not None:
                        pbar.update(seg_end - seg_start)

        self.W = W
        if self.index_path is not None:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.save_w()
        return self.W

    def compute_features(self, X):
        self.mlp.eval()
        with torch.inference_mode():
            if isinstance(X, tuple):
                if len(X) != 2:
                    raise ValueError("X must be a tuple of (queries, queries_counts)")
                queries, queries_counts = X
                x_flat = torch.tensor(queries, dtype=torch.float32)
                feats = self.mlp.feature_extractor(x_flat)
                Q = torch.segment_reduce(
                    feats,
                    "sum",
                    lengths=torch.from_numpy(queries_counts),
                    axis=0,
                )
                return Q / 32

            x = torch.from_numpy(X)
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            B, N, D = x.shape
            feats = self.mlp.feature_extractor(x.flatten(0, 1))
            Q = feats.view(B, N, -1).mean(dim=1)
        return Q
