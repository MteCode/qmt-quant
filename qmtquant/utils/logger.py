"""日志初始化。

普通日志与交易日志分离：交易日志（下单/成交/风控拒绝）单独归档，
便于事后审计与对账。
"""
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

TRADE_LOGGER_NAME = "qmtquant.trade"


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    """配置根日志与交易日志，按日切分保留 90 天"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FMT)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        log_dir / "qmtquant.log", when="midnight", backupCount=90, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    trade_logger = logging.getLogger(TRADE_LOGGER_NAME)
    trade_logger.setLevel(logging.INFO)
    trade_handler = TimedRotatingFileHandler(
        log_dir / "trade.log", when="midnight", backupCount=365, encoding="utf-8"
    )
    trade_handler.setFormatter(formatter)
    trade_logger.addHandler(trade_handler)
    # 交易日志同时进主日志，方便实时观察
    trade_logger.propagate = True


def get_trade_logger() -> logging.Logger:
    return logging.getLogger(TRADE_LOGGER_NAME)
