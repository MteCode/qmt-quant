"""因子面板的本地读取与 point-in-time 查询。

## 为什么策略不能直接读 parquet

策略在回测里逐 Bar 推进，每个调仓日都要取一次横截面。
每次现读 parquet 会让回测慢几十倍；更危险的是容易写成
「读整段再切片」，一不小心就用上了未来数据。

本模块一次性载入内存并按日期索引，``get_cross_section(d)``
只返回**该日及之前**已知的值，用不到未来。

## 数据来源

``data/factor/daily_basic/{年}.parquet``，由
``scripts/download_daily_basic.py`` 从 Tushare 下载。
这是逐日快照，天然 point-in-time —— 2024-01-15 那天的换手率
就是当天盘后算出来的，不像财报有公告日滞后。
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class FactorStore:
    """按 (日期, 标的) 查询因子值"""

    def __init__(self, store_dir: str, columns: list[str],
                 start: str | None = None, end: str | None = None) -> None:
        """
        :param columns: 要载入的因子列，如 ``["turnover_rate_f"]``。
            只载需要的列 —— 全量 1100 万行 × 18 列载进内存要几个 GB
        """
        self.root = Path(store_dir) / "factor" / "daily_basic"
        self.columns = list(columns)
        #: col -> DataFrame(index=日期, columns=vt_symbol)
        self._panels: dict[str, pd.DataFrame] = {}
        self._load(start, end)

    def _load(self, start: str | None, end: str | None) -> None:
        if not self.root.exists():
            raise FileNotFoundError(
                f"缺少因子数据 {self.root}，"
                "请先运行 scripts/download_daily_basic.py")

        years = None
        if start and end:
            years = set(range(pd.Timestamp(start).year,
                              pd.Timestamp(end).year + 1))

        frames = []
        for path in sorted(self.root.glob("*.parquet")):
            if years is not None and int(path.stem) not in years:
                continue
            frames.append(pd.read_parquet(
                path, columns=["trade_date", "symbol"] + self.columns))
        if not frames:
            raise FileNotFoundError(f"{self.root} 内没有可用的年度分片")

        df = pd.concat(frames, ignore_index=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if start:
            df = df[df["trade_date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["trade_date"] <= pd.Timestamp(end)]

        for col in self.columns:
            self._panels[col] = df.pivot_table(
                index="trade_date", columns="symbol", values=col,
                aggfunc="last").sort_index()

        first = self._panels[self.columns[0]]
        logger.info("因子载入完成：%d 个交易日 × %d 只标的，列 %s",
                    len(first), first.shape[1], self.columns)

    # ------------------------------------------------------------ 查询

    def get_cross_section(self, dt, col: str) -> dict[str, float]:
        """取某日的横截面。

        用 asof 语义取「**不晚于** dt 的最近一期」——
        停牌或数据缺失时退回上一个已知值，绝不用未来的值。
        """
        panel = self._panels.get(col)
        if panel is None:
            raise KeyError(f"未载入因子列 {col}，可用: {self.columns}")

        d = pd.Timestamp(dt).normalize()
        idx = panel.index.searchsorted(d, side="right") - 1
        if idx < 0:
            return {}
        row = panel.iloc[idx]
        return {s: float(v) for s, v in row.items() if pd.notna(v)}

    def rolling_mean(self, dt, col: str, window: int) -> dict[str, float]:
        """取截至 dt 的过去 window 个交易日均值。

        单日因子值噪声大 —— 某天一条利好就能让换手率翻几倍，
        而那不代表这只股票长期的换手水平。取均值是为了
        测的是「结构性特征」而不是「当天有没有事件」。
        """
        panel = self._panels.get(col)
        if panel is None:
            raise KeyError(f"未载入因子列 {col}，可用: {self.columns}")

        d = pd.Timestamp(dt).normalize()
        end = panel.index.searchsorted(d, side="right")
        if end <= 0:
            return {}
        start = max(0, end - window)
        chunk = panel.iloc[start:end]
        if len(chunk) < window:
            # 数据不足时返回空而非用短窗口凑 —— 短窗口的均值噪声大得多，
            # 混在一起会让回测早期的选股质量与后期不可比
            return {}
        mean = chunk.mean(skipna=True)
        return {s: float(v) for s, v in mean.items() if pd.notna(v)}

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._panels[self.columns[0]].index
