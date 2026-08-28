"""下载全收益指数（Tushare index_daily）。

## 为什么必须用全收益指数做基准

价格指数不含股息再投资，而策略持有成分股是**实际收到股息的**。
用价格指数做基准，等于把成分股的股息率算成策略的超额收益 ——
一个纯指数复制组合会凭空显示出约 2%/年的 alpha。

沪深300 的股息率约 2%，这个口径错误足以让一个毫无 alpha 的策略
看起来「年化超额 2%」。

产出 data/index/{代码}_tr.parquet。

用法::

    python scripts/download_total_return_index.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.datafeed.tushare_feed import TushareError, TushareFeed  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402

#: 价格指数 -> 全收益指数。中证的全收益代码惯例是 H 开头
TOTAL_RETURN = {
    "000300.SH": ("H00300.CSI", "沪深300全收益"),
    "000905.SH": ("H00905.CSI", "中证500全收益"),
    "000852.SH": ("H00852.CSI", "中证1000全收益"),
    "000016.SH": ("H00016.CSI", "上证50全收益"),
}


def main() -> int:
    p = argparse.ArgumentParser(description="下载全收益指数")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    import pandas as pd
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        feed = TushareFeed(cfg.tushare)
    except TushareError as e:
        print(f"[!] {e}")
        return 1

    out_dir = Path(cfg.data.store_dir) / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    for price_code, (tr_code, name) in TOTAL_RETURN.items():
        try:
            df = feed.index_daily(tr_code, args.start, end)
        except TushareError as e:
            print(f"  [!] {tr_code} 失败：{e}")
            continue
        if df.empty:
            print(f"  [!] {tr_code}（{name}）无数据，跳过")
            continue

        path = out_dir / f"{price_code}_tr.parquet"
        df.to_parquet(path, index=False)
        first, last = df["close"].iloc[0], df["close"].iloc[-1]
        yrs = (df["trade_date"].iloc[-1] - df["trade_date"].iloc[0]).days / 365.25
        print(f"{tr_code:<14} {name:<14} {len(df):>5} 行  "
              f"{df['trade_date'].iloc[0].date()} ~ {df['trade_date'].iloc[-1].date()}"
              f"  年化 {(last / first) ** (1 / yrs) - 1:+.2%}")

    print(f"\n已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
