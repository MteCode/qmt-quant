"""下载行业分类（Tushare stock_basic）。

行业是因子中性化的必需输入。不做中性化的话，实测「低换手因子」
选出的 20 只里 15 只是银行 —— 所谓因子显著很可能只是行业暴露。

产出 data/universe/industry.parquet。

用法::

    python scripts/download_industry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.datafeed.tushare_feed import TushareError, TushareFeed  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402


def main() -> int:
    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    try:
        feed = TushareFeed(cfg.tushare)
        print("拉取全市场基础信息（含在市/退市/暂停上市）...")
        df = feed.stock_basic()
    except TushareError as e:
        print(f"[!] {e}")
        return 1

    if df.empty:
        print("未取到数据")
        return 1

    out = Path(cfg.data.store_dir) / "universe" / "industry.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    n_ind = df["industry"].nunique()
    missing = int(df["industry"].isna().sum())
    print(f"\n{len(df):,} 只标的，{n_ind} 个行业，缺行业标签 {missing} 只")
    print("\n行业分布（前 15）:")
    print(df["industry"].value_counts().head(15).to_string())
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
