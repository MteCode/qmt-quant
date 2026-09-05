"""换机器后的第一道自检 —— 不依赖 miniQMT，clone 完就能跑。

## 为什么单独做一个

换环境最容易漏的不是代码（代码在 GitHub 上），是那些**故意没入库**的东西：

    config/config.yaml    Tushare token + 资金账号，gitignored
    data/                 9.6 GB 行情与财报，gitignored
    strategies/**/state/  回撤峰值与实盘净值，本机记录

漏掉哪一个，报错都发生在很后面 —— 比如漏了 config.yaml，
要跑到某个取数脚本才报「未配置 token」；漏了 state/，
回撤控制器会从零重新记峰值，等于抹掉实盘回撤记忆。

本脚本把这些一次性查清楚，并给出补救路径。

## 与其他检查脚本的分工

    check_env.py       依赖包与 Python 版本
    check_tushare.py   Tushare 连通性与积分
    check_qlib.py      Qlib 数据能否加载
    check_data.py      数据内容的完整性
    check_migration.py **搬家后缺什么**，且不需要联网或启动客户端

用法::

    python scripts/check_migration.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "[ OK ]", "[缺失]", "[注意]"

#: 必须存在的数据目录 -> 缺失时的补救办法
DATA_DIRS = {
    "1d": "python scripts/download_full_market.py",
    "clean": "python scripts/clean_data.py",
    "qlib_data": "python scripts/export_qlib.py",
    "financial": "python scripts/download_financial.py --sector 沪深A股",
    "factor/daily_basic": "python scripts/download_daily_basic.py",
}


def check_config() -> int:
    """配置文件与其中的关键项。"""
    bad = 0
    print("-" * 62)
    print("配置")
    print("-" * 62)

    cfg_file = ROOT / "config" / "config.yaml"
    if not cfg_file.exists():
        print(f"{BAD} config/config.yaml 不存在")
        print("     它含 Tushare token 与资金账号，**故意不入库**，")
        print("     GitHub 上没有 —— 必须从旧机器手工拷过来")
        return 1
    print(f"{OK} config/config.yaml")

    try:
        from qmtquant.config import get_config
        c = get_config()
    except Exception as e:
        print(f"{BAD} 配置加载失败: {e}")
        return bad + 1

    if getattr(c.tushare, "token", ""):
        print(f"{OK} Tushare token 已配置")
    else:
        print(f"{BAD} Tushare token 未配置 —— 指数成分股、估值因子都取不到")
        bad += 1

    qmt = Path(getattr(c.gateway, "qmt_path", "") or "")
    if not str(qmt):
        print(f"{BAD} 未配置 qmt_path")
        bad += 1
    elif qmt.exists():
        print(f"{OK} QMT 数据目录 {qmt}")
    else:
        print(f"{BAD} QMT 数据目录不存在: {qmt}")
        print("     新机器的 miniQMT 装在别处时要改 config.yaml 的 qmt_path，")
        print("     注意填 userdata_mini 而不是安装根目录")
        bad += 1

    return bad


def check_data() -> int:
    """数据目录与最新日期。"""
    from qmtquant.config import get_config

    bad = 0
    print("\n" + "-" * 62)
    print("数据")
    print("-" * 62)

    d = Path(get_config().data.store_dir)
    if not d.exists():
        print(f"{BAD} 数据根目录不存在: {d}")
        print("     data/ 有 9.6 GB 且不入库，必须从旧机器整个拷过来")
        return 1

    for name, fix in DATA_DIRS.items():
        p = d / name
        if p.exists() and any(p.iterdir()):
            n = sum(1 for _ in p.rglob("*.parquet"))
            extra = f"  {n:,} 个 parquet" if n else ""
            print(f"{OK} {name:<20s}{extra}")
        else:
            print(f"{BAD} {name:<20s}  补救: {fix}")
            bad += 1

    db = d / "market.db"
    if db.exists():
        print(f"{OK} market.db  {db.stat().st_size / 1024**3:.1f} GB")
    else:
        print(f"{BAD} market.db  补救: python scripts/build_database.py")
        bad += 1

    return bad


def check_freshness() -> int:
    """行情最新日期。不连 miniQMT，直接读本地落地文件。"""
    import pandas as pd

    from qmtquant.config import get_config

    print("\n" + "-" * 62)
    print("数据新鲜度")
    print("-" * 62)

    d = Path(get_config().data.store_dir)
    cal = d / "qlib_data" / "calendars" / "day.txt"
    if cal.exists():
        days = cal.read_text(encoding="utf-8").split()
        if days:
            print(f"{OK} 交易日历  {len(days):,} 天，最新 {days[-1]}")

    # 抽查若干只，确认日期没有参差 —— 批量下载中断会留下日期不齐的坑
    bars = d / "1d"
    if not bars.exists():
        return 0
    files = list(bars.rglob("*.parquet"))[:200]
    if not files:
        return 0

    last = {}
    for f in files:
        try:
            idx = pd.read_parquet(f, columns=[]).index
        except (OSError, ValueError):
            continue
        if len(idx):
            last[f.stem] = str(idx[-1])[:10].replace("-", "")
    if not last:
        return 0

    s = pd.Series(last)
    top = s.value_counts().head(3)
    print(f"{OK} 抽查 {len(s)} 只，最新日期分布：")
    for date, n in top.items():
        print(f"       {date}  {n} 只")
    if len(top) > 1 and top.iloc[0] < len(s) * 0.9:
        print(f"{WARN} 日期不齐 —— 批量下载可能中断过，")
        print("       补救: python scripts/update_market_data.py --mode post --all")
    return 0


def check_state() -> int:
    """实盘运行时状态。缺了不阻塞，但要明确告知。

    这些文件**没有任何办法重建**：回撤峰值是跨重启累积的实盘记录，
    净值曲线更是漏一天补不回来 —— 券商查不到历史序列。
    删掉等于抹掉实盘记忆，风控会从零重新记峰值。
    """
    import json

    from qmtquant.config import get_config

    print("\n" + "-" * 62)
    print("实盘状态（不入库，漏拷补不回来）")
    print("-" * 62)

    d = Path(get_config().data.store_dir)
    found = False

    rs = d / "risk_state.json"
    if rs.exists():
        found = True
        try:
            s = json.loads(rs.read_text(encoding="utf-8"))
            print(f"{OK} data/risk_state.json  峰值 {s.get('peak', 0):,.0f}"
                  f"  回撤 {s.get('drawdown', 0):.2%}"
                  f"  末次观测 {s.get('last_obs_date', '?')}")
        except (OSError, ValueError):
            print(f"{OK} data/risk_state.json（解析失败，但文件在）")

    sdb = d / "state.db"
    if sdb.exists():
        found = True
        print(f"{OK} data/state.db  {sdb.stat().st_size / 1024:.0f} KB")

    for p in sorted(ROOT.glob("strategies/*/state")):
        items = [x.name for x in p.iterdir()] if p.is_dir() else []
        if items:
            found = True
            print(f"{OK} {p.relative_to(ROOT)}  {', '.join(items[:5])}")

    if not found:
        print(f"{WARN} 没找到任何运行时状态")
        print("       没跑过实盘就正常；跑过就必须从旧机器拷 data/risk_state.json、")
        print("       data/state.db 和 strategies/*/state/")
    return 0


def main() -> int:
    print("=" * 62)
    print("换机器自检")
    print(f"项目路径: {ROOT}")
    print("=" * 62)

    bad = check_config()
    if not bad or (ROOT / "config" / "config.yaml").exists():
        bad += check_data()
        try:
            check_freshness()
        except Exception as e:
            print(f"{WARN} 新鲜度检查跳过: {e}")
    check_state()

    print("\n" + "=" * 62)
    if bad:
        print(f"发现 {bad} 处缺失 —— 按上面的补救办法逐条处理")
        print("详见 MIGRATION.md")
        return 1
    print("全部就绪。下一步建议先用模拟盘验证下单链路：")
    print("  python strategies/lgb_agents_ppo/generate_signal.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
