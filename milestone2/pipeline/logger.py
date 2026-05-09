"""
pipeline.logger
===============
Training logger: writes epoch metrics to CSV, saves best checkpoints,
and stores the full run configuration. Training is fully resumable.

Files produced per run
-----------------------
runs/<run_name>/logs.csv          — epoch-level metrics (appended per epoch)
runs/<run_name>/best_checkpoint.npz — best model weights + optimizer state
runs/<run_name>/config.json        — full run hyperparameter configuration
"""

from __future__ import annotations
import csv, json, os, time
import numpy as np


class TrainingLogger:
    """
    Per-run logger with CSV metrics, best-checkpoint saving, and
    JSON config persistence.

    Parameters
    ----------
    run_dir  : directory for this run's output files
    config   : dict of hyperparameters to save (learning rate, batch_size, …)
    resume   : if True, append to existing logs.csv instead of overwriting
    """

    COLUMNS = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"]

    def __init__(self, run_dir: str, config: dict, resume: bool = False):
        self.run_dir   = run_dir
        self.config    = config
        self.resume    = resume
        os.makedirs(run_dir, exist_ok=True)

        self._log_path  = os.path.join(run_dir, "logs.csv")
        self._ckpt_path = os.path.join(run_dir, "best_checkpoint.npz")
        self._cfg_path  = os.path.join(run_dir, "config.json")

        self._best_val_loss = np.inf
        self._best_epoch    = 0

        # Save config
        config_full = dict(config)
        config_full["run_dir"]    = run_dir
        config_full["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._cfg_path, "w") as f:
            json.dump(config_full, f, indent=2, default=str)

        # Prepare CSV
        write_header = (not resume) or (not os.path.exists(self._log_path))
        self._csv_fh = open(self._log_path, "a" if resume else "w", newline="")
        self._writer = csv.DictWriter(self._csv_fh, fieldnames=self.COLUMNS)
        if write_header:
            self._writer.writeheader()
            self._csv_fh.flush()

    # ── Per-epoch logging ─────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Write one row to logs.csv and flush immediately."""
        self._writer.writerow({
            "epoch":         epoch,
            "train_loss":    round(train_loss, 6),
            "val_loss":      round(val_loss,   6),
            "train_acc":     round(train_acc,  6),
            "val_acc":       round(val_acc,    6),
            "learning_rate": round(lr,         8),
        })
        self._csv_fh.flush()

    # ── Checkpoint ───────────────────────────────────────────────────────

    def maybe_save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        model_params: dict,
        optimizer_state: dict | None = None,
    ) -> bool:
        """
        Save a checkpoint if val_loss is the best seen so far.

        Parameters
        ----------
        model_params     : dict of {name: np.ndarray} weight arrays
        optimizer_state  : dict of optimizer internal state (moments, etc.)

        Returns
        -------
        True if a new best checkpoint was saved.
        """
        if val_loss < self._best_val_loss - 1e-4:
            self._best_val_loss = val_loss
            self._best_epoch    = epoch

            payload = {}
            for k, v in model_params.items():
                payload[f"param_{k}"] = np.asarray(v)
            if optimizer_state:
                payload["optimizer_state_json"] = np.array(
                    json.dumps(optimizer_state, default=lambda x: x.tolist()
                    if hasattr(x, "tolist") else str(x))
                )
            payload["meta_epoch"]    = np.array(epoch)
            payload["meta_val_loss"] = np.array(val_loss)

            np.savez(self._ckpt_path, **payload)
            return True
        return False

    def load_checkpoint(self) -> dict:
        """
        Load best checkpoint.

        Returns
        -------
        dict with keys: params (dict name→array), epoch, val_loss
        """
        if not os.path.exists(self._ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {self._ckpt_path}")

        data   = np.load(self._ckpt_path, allow_pickle=True)
        params = {k[6:]: data[k] for k in data.files if k.startswith("param_")}
        return {
            "params":   params,
            "epoch":    int(data["meta_epoch"]),
            "val_loss": float(data["meta_val_loss"]),
        }

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary dict of this run."""
        return {
            "run_dir":        self.run_dir,
            "best_epoch":     self._best_epoch,
            "best_val_loss":  self._best_val_loss,
            "log_path":       self._log_path,
            "ckpt_path":      self._ckpt_path,
            "config_path":    self._cfg_path,
        }

    def close(self) -> None:
        self._csv_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
