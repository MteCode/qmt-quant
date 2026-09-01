"""下载北向资金、龙虎榜、融资融券数据。

三类「大资金」信号：
1. 北向资金（沪深股通）—— 个股层面每日净买入，「聪明钱」代理
2. 龙虎榜 —— 机构席位净买入，事件型信号
3. 融资融券 —— 杠杆资金方向，日频

数据源：Tushare Pro（需 2000 积分）

用法::

    python scripts/download_money_flow.py --all
    python scripts/download_money_flow.py --hsgt          # 只下北向
    python scripts/download_money_flow.py --dragon        # 只下龙虎榜
    python scripts/download_money_flow.py --margin        # 只下融资融券
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from qmtquant.config import LOG_DIR, get_config
from qmtquant.datafeed.tushare_feed import TushareClient
from qmtquant.utils.logger import setup_logging
from qmtquant.utils.symbol import from_xt_symbol


def download_hsgt(client: TushareClient, out_dir: Path,
                  start: str, end: str) -> None:
    """北向资金个股层面每日净买入。

    接口 hsgt_top10：沪深股通每日十大活跃股
    接口 moneyflow_hsgt：北向资金每日汇总（大盘层面）

    个股层面用 ggt_daily 或 hsgt_top10。
    但 hsgt_top10 只有十大活跃股，覆盖面窄。
    用 moneyflow_hsgt 做大盘层面的择时信号更实际。
    """
    print("\n--- 北向资金（moneyflow_hsgt）---")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 大盘层面汇总
    print("  下载每日北向资金汇总...")
    df = client.query("moneyflow_hsgt",
                      start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                      end_date=pd.Timestamp(end).strftime("%Y%m%d"))
    if df.empty:
        print("  返回空，可能积分不足或日期范围无数据")
        return
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    path = out_dir / "northbound_flow.parquet"
    df.to_parquet(path)
    print(f"  {len(df)} 条  {df.trade_date.min().date()} ~ "
          f"{df.trade_date.max().date()}")
    print(f"  -> {path}")

    # 十大活跃股（个股层面）
    print("  下载沪股通十大活跃股...")
    frames = []
    months = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    for i, m in enumerate(months, 1):
        m_end = (m + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")
        m_start = m.strftime("%Y%m%d")
        for market in ("1", "3"):  # 1=沪股通, 3=深股通
            d = client.query("hsgt_top10", start_date=m_start,
                             end_date=m_end, market_type=market)
            if not d.empty:
                d["market_type"] = market
                frames.append(d)
        if i % 12 == 0:
            sys.stdout.write(f"\r  已拉取 {i}/{len(months)} 个月")
            sys.stdout.flush()

    if frames:
        sys.stdout.write("\n")
        top10 = pd.concat(frames, ignore_index=True)
        top10["trade_date"] = pd.to_datetime(top10["trade_date"],
                                              format="%Y%m%d")
        if "ts_code" in top10.columns:
            top10["symbol"] = top10["ts_code"].map(from_xt_symbol)
        top10 = top10.sort_values("trade_date").reset_index(drop=True)
        path = out_dir / "northbound_top10.parquet"
        top10.to_parquet(path)
        print(f"  十大活跃股 {len(top10)} 条")
        print(f"  -> {path}")
    else:
        print("  十大活跃股无数据")


def download_dragon(client: TushareClient, out_dir: Path,
                    start: str, end: str) -> None:
    """龙虎榜数据。

    接口 top_list：龙虎榜每日上榜个股
    接口 top_inst：龙虎榜机构明细（买卖方向+金额）
    """
    print("\n--- 龙虎榜 ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    # top_list / top_inst 要求 trade_date（单日），逐日拉取
    print("  获取交易日历...")
    from qmtquant.datafeed.tushare_feed import TushareFeed
    feed = TushareFeed(client=client)
    trade_days = feed.trade_dates(start, end)
    print(f"  {len(trade_days)} 个交易日")

    print("  下载龙虎榜上榜个股（逐日）...")
    frames_list = []
    frames_inst = []

    for i, td in enumerate(trade_days, 1):
        td_str = td.strftime("%Y%m%d")

        d = client.query("top_list", trade_date=td_str)
        if not d.empty:
            frames_list.append(d)

        d2 = client.query("top_inst", trade_date=td_str)
        if not d2.empty:
            frames_inst.append(d2)

        if i % 100 == 0:
            sys.stdout.write(f"\r  已拉取 {i}/{len(trade_days)} 日")
            sys.stdout.flush()

    sys.stdout.write(f"\r  已拉取 {len(trade_days)}/{len(trade_days)} 日\n")

    if frames_list:
        top = pd.concat(frames_list, ignore_index=True)
        top["trade_date"] = pd.to_datetime(top["trade_date"], format="%Y%m%d")
        if "ts_code" in top.columns:
            top["symbol"] = top["ts_code"].map(from_xt_symbol)
        top = top.sort_values("trade_date").reset_index(drop=True)
        path = out_dir / "dragon_tiger_list.parquet"
        top.to_parquet(path)
        print(f"  上榜个股 {len(top)} 条")
        print(f"  -> {path}")

    if frames_inst:
        inst = pd.concat(frames_inst, ignore_index=True)
        inst["trade_date"] = pd.to_datetime(inst["trade_date"],
                                             format="%Y%m%d")
        if "ts_code" in inst.columns:
            inst["symbol"] = inst["ts_code"].map(from_xt_symbol)
        inst = inst.sort_values("trade_date").reset_index(drop=True)
        path = out_dir / "dragon_tiger_inst.parquet"
        inst.to_parquet(path)
        print(f"  机构明细 {len(inst)} 条")
        print(f"  -> {path}")


def download_margin(client: TushareClient, out_dir: Path,
                    start: str, end: str) -> None:
    """融资融券余额。

    接口 margin_detail：个股融资融券每日明细
    """
    print("\n--- 融资融券 ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  下载个股融资融券明细...")
    frames = []
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")

    for i, m in enumerate(dates, 1):
        m_end = (m + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")
        m_start = m.strftime("%Y%m%d")
        d = client.query("margin_detail", start_date=m_start,
                         end_date=m_end)
        if not d.empty:
            frames.append(d)
        if i % 12 == 0:
            sys.stdout.write(f"\r  已拉取 {i}/{len(dates)} 个月")
            sys.stdout.flush()

    sys.stdout.write("\n")

    if frames:
        mg = pd.concat(frames, ignore_index=True)
        mg["trade_date"] = pd.to_datetime(mg["trade_date"], format="%Y%m%d")
        if "ts_code" in mg.columns:
            mg["symbol"] = mg["ts_code"].map(from_xt_symbol)
        mg = mg.sort_values("trade_date").reset_index(drop=True)
        path = out_dir / "margin_detail.parquet"
        mg.to_parquet(path)
        print(f"  融资融券明细 {len(mg)} 条")
        print(f"  -> {path}")
    else:
        print("  无数据")


def main() -> int:
    p = argparse.ArgumentParser(description="下载大资金数据")
    p.add_argument("--hsgt", action="store_true", help="北向资金")
    p.add_argument("--dragon", action="store_true", help="龙虎榜")
    p.add_argument("--margin", action="store_true", help="融资融券")
    p.add_argument("--all", action="store_true", help="全部下载")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    if not any([args.hsgt, args.dragon, args.margin, args.all]):
        print("请指定 --hsgt / --dragon / --margin / --all")
        return 1

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out = Path(cfg.data.store_dir) / "money_flow"

    client = TushareClient(cfg.tushare)
    print(f"数据目录: {out.resolve()}")
    print(f"区间: {args.start} ~ {end}")

    t0 = time.time()

    if args.hsgt or args.all:
        download_hsgt(client, out, args.start, end)
    if args.dragon or args.all:
        download_dragon(client, out, args.start, end)
    if args.margin or args.all:
        download_margin(client, out, args.start, end)

    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
