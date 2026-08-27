"""本地状态持久化（SQLite）。

## 什么该存，什么不该存

**不该存：持仓与资金。** 券商是唯一真相来源。重启后必须从券商查询并对账，
而不是信任本地记录 —— 本地与券商不一致时，信本地会导致重复下单或
以为持有实际已被卖出的仓位。`LiveEngine.reconcile()` 做的就是这件事。

**该存：券商不知道的东西。**

- 策略内部状态：指标窗口、调仓计数、上次信号方向。
  券商只知道你有多少股，不知道你的 MA20 算到哪了。
  不持久化的话，盘中重启会丢失全部指标，策略要重新预热。
- 回撤控制状态：峰值净值、当前档位。
  这是**跨天累积**的，重启若从头开始，等于抹掉回撤记忆，
  一个已经亏了 15% 的账户会被当成刚开始交易。
- 每日净值：用于事后复盘与绩效归因，券商只给当前快照。
- 成交与委托流水：用于与券商对账单核对。

## 为什么用 SQLite

单机、单进程、数据量小（每天几十条记录），文件即数据库，零运维。
真正的行情数据走 Parquet（列式压缩，见 datafeed），两者分工不同：
Parquet 存海量只读的时间序列，SQLite 存少量需要事务与更新的状态。
"""
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_state (
    strategy   TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (strategy, key)
);

CREATE TABLE IF NOT EXISTS daily_equity (
    trade_date TEXT PRIMARY KEY,
    balance    REAL NOT NULL,
    available  REAL NOT NULL,
    market_value REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_log (
    vt_tradeid TEXT PRIMARY KEY,
    trade_date TEXT NOT NULL,
    datetime   TEXT NOT NULL,
    vt_symbol  TEXT NOT NULL,
    direction  TEXT NOT NULL,
    price      REAL NOT NULL,
    volume     REAL NOT NULL,
    commission REAL NOT NULL,
    strategy   TEXT
);
CREATE INDEX IF NOT EXISTS idx_trade_date ON trade_log(trade_date);

CREATE TABLE IF NOT EXISTS order_log (
    vt_orderid TEXT PRIMARY KEY,
    trade_date TEXT NOT NULL,
    datetime   TEXT NOT NULL,
    vt_symbol  TEXT NOT NULL,
    direction  TEXT NOT NULL,
    price      REAL NOT NULL,
    volume     REAL NOT NULL,
    traded     REAL NOT NULL,
    status     TEXT NOT NULL,
    message    TEXT,
    strategy   TEXT
);
CREATE INDEX IF NOT EXISTS idx_order_date ON order_log(trade_date);
"""


def _json_default(obj: Any):
    """只处理已知可安全还原的类型。

    不能用 `default=str` 兜底：那会把任意对象变成 repr 字符串存进去，
    恢复时策略读到 "<object object at 0x...>" 这种垃圾值却不报错 ——
    比直接跳过该字段危险得多。未知类型主动抛错，由调用方跳过并告警。
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"不支持序列化的类型: {type(obj).__name__}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return date.today().isoformat()


class StateStore:
    """SQLite 状态存储。

    线程安全：事件引擎与主线程都会写，因此每次操作独立开连接并加锁。
    数据量很小（每天几十条），这点开销无关紧要，换来的是不必担心
    SQLite 连接的线程亲和性问题。
    """

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------ 策略状态

    def save_state(self, strategy: str, data: dict[str, Any]) -> None:
        """保存策略状态。整体覆盖，不做增量合并 ——
        增量合并会让删掉的字段残留，重启后读到早已失效的旧值。
        """
        ts = _now()
        rows = []
        for key, value in data.items():
            try:
                rows.append((strategy, key,
                             json.dumps(value, ensure_ascii=False,
                                        default=_json_default), ts))
            except (TypeError, ValueError):
                logger.warning("策略 %s 的 %s 无法序列化，跳过", strategy, key)

        with self._conn() as conn:
            conn.execute("DELETE FROM strategy_state WHERE strategy = ?",
                         (strategy,))
            conn.executemany(
                "INSERT INTO strategy_state (strategy, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)", rows)
        logger.debug("已保存策略状态 %s，%d 个字段", strategy, len(rows))

    def load_state(self, strategy: str) -> dict[str, Any]:
        """读取策略状态。无记录返回空 dict。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM strategy_state WHERE strategy = ?",
                (strategy,)).fetchall()

        result = {}
        for r in rows:
            try:
                result[r["key"]] = json.loads(r["value"])
            except json.JSONDecodeError:
                logger.warning("策略 %s 的 %s 反序列化失败，忽略", strategy, r["key"])
        return result

    def state_updated_at(self, strategy: str) -> str | None:
        """状态的最后更新时间。用于判断是否过于陈旧而不宜恢复。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(updated_at) AS ts FROM strategy_state "
                "WHERE strategy = ?", (strategy,)).fetchone()
        return row["ts"] if row and row["ts"] else None

    def clear_state(self, strategy: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM strategy_state WHERE strategy = ?",
                         (strategy,))

    def list_strategies(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT strategy FROM strategy_state "
                "ORDER BY strategy").fetchall()
        return [r["strategy"] for r in rows]

    # ------------------------------------------------------------ 每日净值

    def save_equity(self, balance: float, available: float,
                    market_value: float, trade_date: str | None = None) -> None:
        """记录当日净值。同日重复写入按覆盖处理（盘中会多次调用）。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO daily_equity "
                "(trade_date, balance, available, market_value, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(trade_date) DO UPDATE SET "
                "balance=excluded.balance, available=excluded.available, "
                "market_value=excluded.market_value, "
                "updated_at=excluded.updated_at",
                (trade_date or _today(), balance, available, market_value, _now()))

    def load_equity(self, start: str | None = None,
                    end: str | None = None) -> list[dict]:
        sql = "SELECT * FROM daily_equity"
        params: list = []
        clauses = []
        if start:
            clauses.append("trade_date >= ?"); params.append(start)
        if end:
            clauses.append("trade_date <= ?"); params.append(end)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY trade_date"

        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------ 成交与委托

    def save_trade(self, trade) -> None:
        """记录成交。按 vt_tradeid 幂等 —— 断线重连后会重复推送同一笔成交，
        不去重会导致对账时金额翻倍。
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trade_log "
                "(vt_tradeid, trade_date, datetime, vt_symbol, direction, "
                " price, volume, commission, strategy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trade.vt_tradeid,
                 (trade.datetime or datetime.now()).date().isoformat(),
                 (trade.datetime or datetime.now()).isoformat(timespec="seconds"),
                 trade.vt_symbol, trade.direction.value, trade.price,
                 trade.volume, getattr(trade, "commission", 0.0),
                 trade.reference))

    def save_order(self, order) -> None:
        """记录委托。同一委托状态会多次更新，按 vt_orderid 覆盖。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO order_log "
                "(vt_orderid, trade_date, datetime, vt_symbol, direction, "
                " price, volume, traded, status, message, strategy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(vt_orderid) DO UPDATE SET "
                "traded=excluded.traded, status=excluded.status, "
                "message=excluded.message, datetime=excluded.datetime",
                (order.vt_orderid,
                 (order.datetime or datetime.now()).date().isoformat(),
                 (order.datetime or datetime.now()).isoformat(timespec="seconds"),
                 order.vt_symbol, order.direction.value, order.price,
                 order.volume, order.traded, order.status.value,
                 order.message, order.reference))

    def load_trades(self, trade_date: str | None = None) -> list[dict]:
        sql = "SELECT * FROM trade_log"
        params: list = []
        if trade_date:
            sql += " WHERE trade_date = ?"
            params.append(trade_date)
        sql += " ORDER BY datetime"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def load_orders(self, trade_date: str | None = None) -> list[dict]:
        sql = "SELECT * FROM order_log"
        params: list = []
        if trade_date:
            sql += " WHERE trade_date = ?"
            params.append(trade_date)
        sql += " ORDER BY datetime"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------ 观测

    def summary(self) -> dict:
        with self._conn() as conn:
            def count(table: str) -> int:
                return conn.execute(f"SELECT COUNT(*) AS n FROM {table}"
                                    ).fetchone()["n"]
            return {
                "path": str(self.path.resolve()),
                "size_kb": round(self.path.stat().st_size / 1024, 1)
                if self.path.exists() else 0,
                "strategies": count("strategy_state"),
                "equity_days": count("daily_equity"),
                "trades": count("trade_log"),
                "orders": count("order_log"),
            }
