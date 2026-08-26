"""财务数据存取。

**本模块存在的核心意义是强制 point-in-time 语义。**

财务数据的报告期（`m_timetag`）与公告日（`m_anntime`）之间有巨大滞后：
实测贵州茅台年报平均滞后 100 天以上，最大 478 天。
按报告期取数 = 提前三个多月知道年报内容，这是比幸存者偏差更严重的前视偏差，
而且**不会报错、不会异常，只会让回测收益凭空变好**。

因此本模块的查询接口一律按**公告日**过滤，不提供按报告期查询的快捷方式。
"""
import logging
from pathlib import Path

import pandas as pd

from ..utils.symbol import normalize, split_vt_symbol, to_xt_symbol

logger = logging.getLogger(__name__)

#: QMT 财务报表
TABLE_BALANCE = "Balance"           # 资产负债表
TABLE_INCOME = "Income"             # 利润表
TABLE_CASHFLOW = "CashFlow"         # 现金流量表
TABLE_CAPITAL = "Capital"           # 股本结构
TABLE_PERSHARE = "PershareIndex"    # 每股指标与财务比率
TABLE_HOLDERNUM = "HolderNum"       # 股东户数
TABLE_TOP10 = "Top10Holder"         # 十大股东
TABLE_TOP10_FLOW = "Top10FlowHolder"  # 十大流通股东

#: 默认下载的报表。三大报表 + 每股指标 + 股本，覆盖绝大多数选股因子
DEFAULT_TABLES = [
    TABLE_BALANCE, TABLE_INCOME, TABLE_CASHFLOW,
    TABLE_CAPITAL, TABLE_PERSHARE,
]

#: 报告期字段与公告日字段。不同报表命名不一致，需分别处理
REPORT_DATE_COLS = ("m_timetag", "endDate")
ANNOUNCE_DATE_COLS = ("m_anntime", "declareDate")


def _pick(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_datetime(series: pd.Series) -> pd.Series:
    """QMT 的日期是 'YYYYMMDD' 数字或字符串，0 表示缺失"""
    s = series.astype(str).str.replace(r"\.0$", "", regex=True)
    s = s.where(~s.isin(["0", "", "nan", "None"]))
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


class FinancialStore:
    """财务数据的本地存取与 point-in-time 查询"""

    def __init__(self, store_dir: str) -> None:
        self.root = Path(store_dir) / "financial"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, vt_symbol: str, table: str) -> Path:
        symbol, exchange = split_vt_symbol(vt_symbol)
        d = self.root / table / exchange.value
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}.parquet"

    def has_data(self, vt_symbol: str, table: str) -> bool:
        return self._path(normalize(vt_symbol), table).exists()

    # ------------------------------------------------------------ 下载

    def download(self, vt_symbols: list[str], tables: list[str] | None = None,
                 skip_existing: bool = False, progress=None,
                 batch_size: int = 20, timeout: float = 60.0,
                 download_progress=None) -> dict:
        """从 QMT 下载财务数据并落地 Parquet。

        分两步：

        1. `download_financial_data2` 一次性拉取全部标的（异步批量，带进度回调）
        2. 分块 `get_financial_data` 从本地缓存读出并落地

        为什么不用 `download_financial_data`：实测它极不可靠 ——
        传多只会阻塞 240 秒以上无返回；即使逐只调用，连续十几次后连接
        也会失去响应且**不报错、不超时、直接无限阻塞**，实测 9 分钟只完成 13 只。
        而 `download_financial_data2` 下载 40 只 × 5 表仅需 6.3 秒。

        :param batch_size: 读取阶段每批标的数（下载阶段是一次性的）
        :param timeout: 单次调用超时。xtdata 会无限阻塞，必须强制超时。
        """
        import concurrent.futures as cf

        from xtquant import xtdata

        tables = tables or DEFAULT_TABLES
        result = {"ok": [], "failed": [], "skipped": []}

        targets = []
        for raw in vt_symbols:
            vt_symbol = normalize(raw)
            if skip_existing and all(self.has_data(vt_symbol, t) for t in tables):
                result["skipped"].append(vt_symbol)
            else:
                targets.append(vt_symbol)

        total = len(targets)
        done = 0

        if not targets:
            return result

        # ---- 步骤 1：一次性批量下载到 QMT 本地缓存
        all_xt = [to_xt_symbol(s) for s in targets]
        logger.info("批量下载 %d 只 × %d 张表 ...", len(all_xt), len(tables))

        def _on_progress(info) -> None:
            if download_progress:
                download_progress(info.get("finished", 0), info.get("total", 0))

        try:
            xtdata.download_financial_data2(all_xt, table_list=tables,
                                            callback=_on_progress)
        except Exception:
            logger.exception("批量下载失败，仍尝试从本地缓存读取已有数据")

        # ---- 步骤 2：分块读取并落地
        for start in range(0, total, batch_size):
            batch = targets[start:start + batch_size]
            xt_symbols = [to_xt_symbol(s) for s in batch]

            try:
                # xtdata 偶发无限阻塞且不报错，必须强制超时
                with cf.ThreadPoolExecutor(1) as ex:
                    data = ex.submit(xtdata.get_financial_data,
                                     xt_symbols, table_list=tables).result(timeout=timeout)
            except cf.TimeoutError:
                logger.warning("读取超时(%.0fs)，跳过：%s", timeout, ", ".join(batch))
                result["failed"].extend(batch)
                done += len(batch)
                if progress:
                    progress(done, total, batch[-1])
                continue
            except Exception:
                logger.exception("读取失败（%d 只）", len(batch))
                result["failed"].extend(batch)
                done += len(batch)
                if progress:
                    progress(done, total, batch[-1])
                continue

            for vt_symbol, xt_symbol in zip(batch, xt_symbols):
                done += 1
                if progress:
                    progress(done, total, vt_symbol)

                tabs = (data or {}).get(xt_symbol) or {}
                saved = 0
                for table in tables:
                    df = tabs.get(table)
                    if df is None or len(df) == 0:
                        continue
                    try:
                        self._normalize_and_save(df, vt_symbol, table)
                        saved += 1
                    except Exception:
                        logger.exception("保存失败 %s/%s", vt_symbol, table)

                (result["ok"] if saved else result["failed"]).append(vt_symbol)

        logger.info("财务数据下载完成：成功 %d 跳过 %d 失败 %d",
                    len(result["ok"]), len(result["skipped"]), len(result["failed"]))
        return result

    def _normalize_and_save(self, df: pd.DataFrame, vt_symbol: str, table: str) -> None:
        """统一日期列为 report_date / announce_date 后落地"""
        df = df.copy()

        rep_col = _pick(df, REPORT_DATE_COLS)
        ann_col = _pick(df, ANNOUNCE_DATE_COLS)

        df["report_date"] = _to_datetime(df[rep_col]) if rep_col else pd.NaT
        if ann_col:
            df["announce_date"] = _to_datetime(df[ann_col])
        else:
            df["announce_date"] = pd.NaT

        # 公告日缺失时无法判断何时可用。保守处理：按报告期 + 一个季度顺延，
        # 宁可晚用也不能早用。
        missing = df["announce_date"].isna()
        if missing.any():
            df.loc[missing, "announce_date"] = (
                df.loc[missing, "report_date"] + pd.Timedelta(days=90)
            )
            logger.debug("%s/%s 有 %d 行缺公告日，按报告期+90天保守顺延",
                         vt_symbol, table, int(missing.sum()))

        df = df.dropna(subset=["announce_date"]).sort_values("announce_date")
        df.to_parquet(self._path(vt_symbol, table))

    # ------------------------------------------------------------ 读取

    def load(self, vt_symbol: str, table: str) -> pd.DataFrame:
        """读取原始财务数据（含 report_date / announce_date 列）"""
        path = self._path(normalize(vt_symbol), table)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def get_asof(self, vt_symbol: str, table: str, date,
                 fields: list[str] | None = None) -> pd.Series | None:
        """取截至 `date` **已公告**的最新一期财务数据。

        这是本模块的核心接口：只返回公告日 <= date 的记录，
        因此回测中不可能用到当时尚未披露的数据。

        :return: 最新一期的 Series；无可用数据返回 None
        """
        df = self.load(vt_symbol, table)
        if df.empty:
            return None

        d = pd.Timestamp(date)
        available = df[df["announce_date"] <= d]
        if available.empty:
            return None

        row = available.iloc[-1]
        return row[fields] if fields else row

    def get_panel(self, vt_symbols: list[str], table: str, date,
                  fields: list[str]) -> pd.DataFrame:
        """取一个横截面：多个标的在 `date` 时点已公告的最新财务数据。

        选股策略每个调仓日调用一次，拿到当日真实可用的因子矩阵。
        """
        rows = {}
        for raw in vt_symbols:
            vt_symbol = normalize(raw)
            s = self.get_asof(vt_symbol, table, date, fields)
            if s is not None:
                rows[vt_symbol] = s

        if not rows:
            return pd.DataFrame(columns=fields)
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------ 观测

    def summary(self) -> pd.DataFrame:
        """统计本地已下载的财务数据"""
        rows = []
        for table_dir in sorted(self.root.iterdir()):
            if not table_dir.is_dir():
                continue
            files = list(table_dir.rglob("*.parquet"))
            if not files:
                continue
            rows.append({
                "报表": table_dir.name,
                "标的数": len(files),
                "占用(MB)": round(
                    sum(f.stat().st_size for f in files) / 1024 / 1024, 2),
            })
        return pd.DataFrame(rows)
