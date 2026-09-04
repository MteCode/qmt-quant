"""下载 A 股公告并归类 —— 公告类型是可回测的另类数据。

## 为什么用公告而不是新闻

新闻数据有两个问题：东财个股新闻接口只返回最近约 10 条，做不了历史回测；
而**用 LLM 读新闻判断利好利空，有无法修正的前视偏差** ——
模型权重里已经包含 2022~2024 年发生过什么，让它读 2022 年的稿子
判断风险，实为用已知结果反推。这不是滞后能解决的。

公告不同：

- **时间戳可靠** —— 交易所披露时间，不是采集时间
- **类型是结构化字段** —— 减持、诉讼、业绩预增各有明确标签，
  用规则映射即可，不需要 LLM，因此可以正常回测
- **免费且可按日回溯** —— akshare 的 stock_notice_report

## 归类逻辑

原始类型有 110 多种，多数是流程性的（董事会决议、法律意见书、
保荐意见），与股价无关。只保留有明确金融含义的几类，
其余归入 neutral 不参与因子计算。

方向判定基于常识而非数据挖掘 —— 减持是负面、增持是正面、
诉讼处罚是负面。**先定规则再看结果**，避免用结果反推规则。

用法::

    python scripts/download_announcements.py --start 20220101 --end 20260831
    python scripts/download_announcements.py --recent 30
"""
import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "announcement"

#: 公告类型 -> (大类, 方向)。方向 +1 利好 / -1 利空 / 0 中性
#:
#: 只列有明确金融含义的。原始 110 多种类型里大部分是流程性公告
#: （董事会决议、法律意见书、保荐意见等），与股价无关，不做映射。
#:
#: 方向按常识判定，不是从数据里挖出来的 —— 先定规则再看结果，
#: 否则就是用结果反推规则，回测必然虚高。
CATEGORY = {
    # --- 股东行为：最直接的信号
    "股东/实际控制人股份减持": ("reduce", -1),
    "高管人员持股变动": ("reduce", -1),
    "股东/实际控制人股份增持": ("increase", +1),
    "回购进展情况": ("buyback", +1),
    "回购实施公告": ("buyback", +1),
    "回购报告书": ("buyback", +1),
    "回购方案修订": ("buyback", 0),

    # --- 风险事件
    "诉讼仲裁": ("risk", -1),
    "处罚": ("risk", -1),
    "违法违规": ("risk", -1),
    "风险提示性公告": ("risk", -1),
    "其它风险提示公告": ("risk", -1),
    "终止上市风险提示": ("risk", -1),
    "终止上市提示公告": ("risk", -1),
    "股份质押、冻结": ("risk", -1),
    "上交所股票监管工作函": ("risk", -1),
    "监管工作函回复公告": ("risk", -1),
    "回复问询函公告": ("risk", -1),
    "会计师事务所问询函回复公告": ("risk", -1),
    "股票交易异常波动": ("volatile", 0),

    # --- 经营与业绩
    "重大合同": ("business", +1),
    "签订协议": ("business", +1),
    "获得认证": ("business", +1),
    "获得补贴（资助）": ("business", +1),
    "月度经营情况": ("business", 0),
    "产品价格调整": ("business", 0),
    "对外项目投资": ("business", 0),
    "投资设立公司": ("business", 0),
    "收购出售资产/股权": ("business", 0),
    "重组进展公告": ("business", 0),

    # --- 分配与融资
    "分配方案实施": ("dividend", +1),
    "分配方案决议公告": ("dividend", +1),
    "增发预案": ("dilution", -1),
    "增发提示性公告": ("dilution", -1),
    "增发获准公告": ("dilution", -1),
    "增发发行结果公告": ("dilution", -1),
    "其他增发事项公告": ("dilution", -1),
    "增发终止": ("dilution", +1),
    "限售股份上市流通": ("unlock", -1),

    # --- 激励与关注度
    "股权激励计划": ("incentive", +1),
    "股权激励进展公告": ("incentive", +1),
    "员工持股计划": ("incentive", +1),
    "调研活动": ("attention", 0),

    # --- 担保
    "提供/对外担保公告": ("guarantee", -1),
    "其他担保公告": ("guarantee", -1),
}


def to_vt_symbol(code: str) -> str | None:
    """6 位代码 -> vt_symbol。北交所不在标的池内，跳过。"""
    c = str(code).zfill(6)
    if c.startswith(("60", "68", "51", "58")):
        return f"{c}.SSE"
    if c.startswith(("00", "30", "15", "16")):
        return f"{c}.SZSE"
    return None


def fetch_day(day: str):
    """拉单日全市场公告。"""
    import akshare as ak
    import pandas as pd

    try:
        df = ak.stock_notice_report(symbol="全部", date=day)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    rows = []
    for r in df.itertuples():
        vt = to_vt_symbol(getattr(r, "代码", ""))
        if not vt:
            continue
        raw = str(getattr(r, "公告类型", ""))
        cat, direction = CATEGORY.get(raw, ("neutral", 0))
        rows.append({
            "date": day,
            "vt_symbol": vt,
            "name": getattr(r, "名称", ""),
            "raw_type": raw,
            "category": cat,
            "direction": direction,
            "title": str(getattr(r, "公告标题", ""))[:120],
        })
    return pd.DataFrame(rows) if rows else None


def main() -> int:
    p = argparse.ArgumentParser(description="下载 A 股公告")
    p.add_argument("--start", default=None, help="起始日 YYYYMMDD")
    p.add_argument("--end", default=None, help="结束日 YYYYMMDD")
    p.add_argument("--recent", type=int, default=0, help="最近 N 个自然日")
    p.add_argument("--sleep", type=float, default=0.4,
                    help="每次请求间隔秒数，避免被限流")
    args = p.parse_args()

    import pandas as pd

    if args.recent:
        end = date.today()
        start = end - timedelta(days=args.recent)
    else:
        if not args.start:
            print("需要 --start 或 --recent")
            return 1
        start = datetime.strptime(args.start, "%Y%m%d").date()
        end = (datetime.strptime(args.end, "%Y%m%d").date()
               if args.end else date.today())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 58)
    print(f"下载公告  {start} ~ {end}")
    print(f"  输出 {OUT_DIR}")
    print("=" * 58)

    days = [(start + timedelta(days=i))
            for i in range((end - start).days + 1)]
    # 只跑工作日 —— 周末无公告披露，省掉一半请求
    days = [d for d in days if d.weekday() < 5]
    print(f"  待抓 {len(days)} 个工作日\n")

    ok = skip = fail = 0
    total_rows = 0
    t0 = time.time()
    for i, d in enumerate(days, 1):
        ds = d.strftime("%Y%m%d")
        dst = OUT_DIR / f"{ds}.parquet"
        if dst.exists():
            skip += 1
            continue

        df = fetch_day(ds)
        if df is None or df.empty:
            fail += 1
        else:
            df.to_parquet(dst, index=False)
            ok += 1
            total_rows += len(df)
        time.sleep(args.sleep)

        if i % 20 == 0:
            el = time.time() - t0
            eta = el / i * (len(days) - i) / 60
            print(f"  {i}/{len(days)}  成功 {ok} 跳过 {skip} 无数据 {fail}"
                  f"  {total_rows:,} 条  剩约 {eta:.0f} 分钟")

    print(f"\n完成：成功 {ok}，跳过 {skip}，无数据 {fail}")
    print(f"新增 {total_rows:,} 条，耗时 {(time.time() - t0) / 60:.1f} 分钟")

    files = sorted(OUT_DIR.glob("*.parquet"))
    if files:
        all_df = pd.concat([pd.read_parquet(f) for f in files[-30:]],
                           ignore_index=True)
        print(f"\n最近 30 个交易日的类别分布:")
        g = all_df[all_df["category"] != "neutral"].groupby(
            ["category", "direction"]).size().sort_values(ascending=False)
        for (cat, d), n in g.items():
            sign = "利好" if d > 0 else ("利空" if d < 0 else "中性")
            print(f"  {cat:<12s} {sign:<4s} {n:>6,} 条")
        print(f"\n  已归类 {(all_df['category'] != 'neutral').sum():,} / "
              f"{len(all_df):,} 条"
              f"（其余为流程性公告，与股价无关）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
