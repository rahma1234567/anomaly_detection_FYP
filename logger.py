import logging
import os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

def _make_logger(name: str, prefix: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    today = datetime.now().strftime("%Y-%m-%d")
    handler = logging.FileHandler(LOG_DIR / f"{prefix}_{today}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    return logger

_system    = _make_logger("system",    "system")
_anomalies = _make_logger("anomalies", "anomalies")

def log_info(msg: str)    -> None: _system.info(msg)
def log_warning(msg: str) -> None: _system.warning(msg)
def log_error(msg: str)   -> None: _system.error(msg)

def log_anomaly(transaction_id: str, account_id: str, amount: float,
                location: str, risk_pct: float, model: str, triggered_by: str) -> None:
    """One structured line per flagged transaction."""
    _anomalies.info(
        f"FLAGGED tx={transaction_id} acc={account_id} amount={amount:.2f} "
        f"loc={location} risk={risk_pct:.1f}% model={model} user={triggered_by}")