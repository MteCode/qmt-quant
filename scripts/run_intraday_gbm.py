"""日内 GBM 实盘 —— 动量模式，20 万本金。

用法：
    # 预览模式（不实际下单）
    python scripts/run_intraday_gbm.py --dry-run

    # 实盘执行
    python scripts/run_intraday_gbm.py

    # 自定义参数
    python scripts/run_intraday_gbm.py --capital 200000 --max-positions 10
    python scripts/run_intraday_gbm.py --trade-mode mean_reversion

miniQMT 客户端必须已启动并登录。按 Ctrl+C 优雅退出。
"""
import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def preheat_lightgbm() -> bool:
    """在任何 xtquant 导入之前，把 LightGBM 的原生运行时初始化完。

    ## 不这么做会发生什么

    xtquant 和 LightGBM 各自带一套 OpenMP 运行时。同进程下若 xtquant 先
    加载，LightGBM 后续第一次 predict 会直接
    `access violation reading 0x0` 崩在原生层 —— Python 侧只看到 OSError，
    堆栈指向 lightgbm/basic.py，完全指不到真正的原因。

    实测：先 predict 一次再 import xtquant 正常，反过来必崩；
    `num_threads=1` 之类的参数不管用，因为崩在运行时初始化而不是并行调度。

    表现形式极具迷惑性：引擎正常、行情正常、心跳正常，只是每根 bar 都
    在 _score_all 里抛异常被吞掉，**一整天不下一笔单也不报错到界面上**。

    这个函数必须在 load_universe / 网关连接之前调用 —— 那两处都会 import
    xtquant。
    """
    try:
        import numpy as np

        from strategies.intraday_gbm.strategy import MODEL_PATH
    except ImportError as e:
        print(f"  [WARN] LightGBM 预热跳过（导入失败）: {e}")
        return False

    if not MODEL_PATH.exists():
        print(f"  [WARN] LightGBM 预热跳过：模型不存在 {MODEL_PATH}")
        return False

    try:
        import joblib

        data = joblib.load(MODEL_PATH)
        model = data["model"]
        n = len(data.get("features") or [])
        model.predict_proba(np.zeros((1, n), dtype=np.float64))
    except Exception as e:                           # noqa: BLE001
        print(f"  [WARN] LightGBM 预热失败: {e}")
        return False
    return True


def load_full_market() -> list[str]:
    """全市场标的池 —— 取自 1m 清洗数据目录，与模型训练集完全一致。

    BSE 目录直接跳过（北交所硬性排除）。
    """
    syms: list[str] = []
    base = ROOT / "data" / "clean" / "1m"
    for ex in ("SSE", "SZSE"):
        d = base / ex
        if not d.is_dir():
            continue
        syms.extend(f"{f.stem}.{ex}" for f in d.glob("*.parquet"))
    return syms


def load_csi1000() -> list[str]:
    """CSI1000 最新一期成分快照。

    instruments 文件按区间记录成分，`end` 是该期的截止日而非「至今有效」。
    筛 `end >= today` 在成分表落后于当天时会得到空集，所以取最大 end 那一期。
    """
    import pandas as pd

    p = ROOT / "data" / "qlib_data" / "instruments" / "csi1000.txt"
    if not p.exists():
        return []
    df = pd.read_csv(p, sep="\t", header=None,
                     names=["instrument", "start", "end"])
    if df.empty:
        return []
    latest = df["end"].max()
    df = df[df["end"] == latest]
    print(f"  CSI1000 成分快照: {latest}（{len(df)} 只）")
    return [f"{i[2:]}.{'SSE' if i[:2].lower() == 'sh' else 'SZSE'}"
            for i in df["instrument"]]


def load_st_set() -> set[str]:
    """当前仍挂 ST 的标的。"""
    import pandas as pd

    p = ROOT / "data" / "universe" / "st_history.parquet"
    if not p.exists():
        return set()
    try:
        df = pd.read_parquet(p)
    except (OSError, ValueError) as e:
        print(f"  [WARN] ST 名单读取失败: {e}")
        return set()
    if "end_date" in df.columns:
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = df[df["end_date"].isna() | (df["end_date"].astype(str) >= today)]
    if "vt_symbol" in df.columns:
        return set(df["vt_symbol"].astype(str))
    if "ts_code" in df.columns:
        out = set()
        for tc in df["ts_code"].astype(str):
            code, _, suf = tc.partition(".")
            out.add(f"{code}.{'SSE' if suf.upper() == 'SH' else 'SZSE'}")
        return out
    return set()


def filter_tradable(symbols: list[str], st_set: set[str]) -> list[str]:
    """硬性过滤：科创板 688 / 北交所 43·83·87·92 / ST。"""
    kept, n_star, n_bse, n_st = [], 0, 0, 0
    for vt in set(symbols):
        code = vt.split(".")[0]
        if code.startswith("688"):
            n_star += 1
        elif code[:2] in ("43", "83", "87", "92"):
            n_bse += 1
        elif vt in st_set:
            n_st += 1
        else:
            kept.append(vt)
    print(f"  排除 科创板 {n_star} / 北交所 {n_bse} / ST {n_st}")
    return sorted(kept)


def rank_by_snapshot(symbols: list[str],
                     max_price: float) -> list[tuple[str, float]]:
    """用实时快照过滤高价股，并按成交额从大到小排序。

    价格用 miniQMT 快照而非本地日线：本地日线是复权价，复权后的价位
    和实际下单价没有可比性（茅台复权价 8000+，实际 1259）。

    排序是必须的 —— 全市场四千多只要截断到可订阅规模，按字母序截
    等于只买深市 000 开头那一批，样本严重偏斜。
    """
    from qmtquant.utils.symbol import to_xt_symbol
    from xtquant import xtdata

    xt_map: dict[str, str] = {}
    for vt in symbols:
        try:
            xt_map[to_xt_symbol(vt)] = vt
        except (KeyError, ValueError):
            continue

    ranked: list[tuple[str, float]] = []
    n_high = n_quiet = 0
    codes = list(xt_map)
    for i in range(0, len(codes), 500):
        chunk = codes[i:i + 500]
        try:
            ticks = xtdata.get_full_tick(chunk)
        except Exception as e:
            print(f"  [WARN] 行情快照失败，跳过价格过滤与排序: {e}")
            return [(v, 0.0) for v in symbols]
        for xt in chunk:
            t = ticks.get(xt) or {}
            px = t.get("lastPrice") or t.get("lastClose") or 0.0
            if not px:
                n_quiet += 1
                continue
            if px > max_price:
                n_high += 1
                continue
            ranked.append((xt_map[xt], float(t.get("amount") or 0.0)))

    print(f"  排除 高价(>{max_price:.0f}元) {n_high} / 无报价 {n_quiet}")
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked


def fetch_limit_up_prices(symbols: list[str]) -> dict[str, float]:
    """取每只标的的涨停价，供策略过滤涨停票。

    用券商给的 UpStopPrice 而非自己按涨跌幅推算 —— 主板 10%、创业板/科创板
    20%、ST 5%、北交所 30%，还要考虑新股上市首日无涨跌幅，自己算迟早算错
    一类。券商这个值是精确的。
    """
    from qmtquant.utils.symbol import to_xt_symbol
    from xtquant import xtdata

    out: dict[str, float] = {}
    for vt in symbols:
        try:
            d = xtdata.get_instrument_detail(to_xt_symbol(vt)) or {}
        except Exception:                            # noqa: BLE001
            continue
        up = d.get("UpStopPrice") or 0
        if up:
            out[vt] = float(up)
    return out


def warmup_buffers(strategy, symbols: list[str]) -> None:
    """下载当日 1m bar 灌进策略缓冲区，省掉 30 分钟冷启动。"""
    import datetime as _dt

    from qmtquant.utils.symbol import to_xt_symbol
    from xtquant import xtdata

    xt_map: dict[str, str] = {}
    for vt in symbols:
        try:
            xt_map[to_xt_symbol(vt)] = vt
        except (KeyError, ValueError):
            continue

    codes = list(xt_map)
    today = _dt.date.today().strftime("%Y%m%d")
    print(f"预热 1m 缓冲区（{len(codes)} 只）...")
    t0 = time.time()
    try:
        xtdata.download_history_data2(codes, period="1m",
                                      start_time=today, end_time="")
        data = xtdata.get_local_data(field_list=[], stock_list=codes,
                                     period="1m", count=120)
    except Exception as e:
        print(f"  [WARN] 预热失败，策略需自行累积 30 根 bar 才出信号: {e}")
        return

    hist: dict[str, list[dict]] = {}
    for xt, df in (data or {}).items():
        vt = xt_map.get(xt)
        if vt is None or df is None or len(df) == 0:
            continue
        hist[vt] = [{
            "open": float(r.open), "high": float(r.high),
            "low": float(r.low), "close": float(r.close),
            "volume": float(r.volume), "amount": float(r.amount),
        } for r in df.itertuples()]

    n = strategy.warmup(hist)
    print(f"  完成 {n} 只，耗时 {time.time() - t0:.1f}s")


def load_universe(scope: str, max_price: float) -> list[tuple[str, float]]:
    raw = load_full_market() if scope == "full" else load_csi1000()
    if not raw:
        print(f"  [ERROR] 标的池为空（scope={scope}）")
        return []
    print(f"  原始: {len(raw)} 只")
    return rank_by_snapshot(filter_tradable(raw, load_st_set()), max_price)


def main() -> int:
    parser = argparse.ArgumentParser(description="日内 GBM 实盘 —— 动量模式")
    parser.add_argument("--dry-run", action="store_true",
                        help="启动后开启急停，只跑行情不下单")
    parser.add_argument("--trade-mode", default="momentum",
                        choices=["t_plus_0", "mean_reversion", "momentum"],
                        help="交易模式（默认 momentum）")
    parser.add_argument("--capital", type=float, default=200000,
                        help="总资金（默认 200000）")
    parser.add_argument("--max-positions", type=int, default=10,
                        help="最大持仓数（默认 10）")
    parser.add_argument("--max-symbols", type=int, default=500,
                        help="最大订阅标的数（默认 500，过多会拖慢行情推送）")
    parser.add_argument("--universe", default="full",
                        choices=["full", "csi1000"],
                        help="标的范围：full=全市场（默认）/ csi1000")
    parser.add_argument("--max-price", type=float, default=500.0,
                        help="股价上限，超过则排除（默认 500 元）")
    parser.add_argument("--vol-z", type=float, default=0.5,
                        help="量能放大阈值（默认 0.5）。原先 1.5 要求放量到"
                             "1.5 个标准差，筛出来的几乎全是已涨停的票")
    parser.add_argument("--allow-limit-up", action="store_true",
                        help="允许买涨停票。默认不买 —— 涨停板没有卖单，"
                             "限价单只能排队，实测排在封单百万手后面成交不了")
    args = parser.parse_args()

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level, cfg)

    if not cfg.gateway.account_id:
        print("config.yaml 中未配置 gateway.account_id")
        return 1

    position_size = args.capital / args.max_positions
    print("=" * 50)
    print("日内 GBM 实盘")
    print("=" * 50)
    print(f"  模式:     {args.trade_mode}")
    print(f"  范围:     {'全市场' if args.universe == 'full' else 'CSI1000'}")
    print(f"  资金:     {args.capital:,.0f} 元")
    print(f"  最大持仓: {args.max_positions} 只")
    print(f"  每只金额: {position_size:,.0f} 元")
    print(f"  排序选股: 是（use_rank=True）")
    if args.dry_run:
        print("  ** DRY RUN — 只观察不下单 **")
    print()

    # 必须在 load_universe / 网关连接之前 —— 它们都会 import xtquant，
    # 之后 LightGBM 的第一次 predict 就会崩在原生层
    print("预热 LightGBM 运行时...")
    print(f"  {'完成' if preheat_lightgbm() else '未完成（打分可能失败）'}")

    # 加载标的池
    print("加载标的池...")
    ranked = load_universe(args.universe, args.max_price)
    if not ranked:
        return 1

    if len(ranked) > args.max_symbols:
        print(f"  可交易 {len(ranked)} 只，按成交额取前 {args.max_symbols} 只")
        ranked = ranked[:args.max_symbols]
    symbols = sorted(vt for vt, _ in ranked)
    print(f"  标的池: {len(symbols)} 只")

    # 装配引擎
    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway
    from qmtquant.engine.live_engine import LiveEngine
    from qmtquant.risk.risk_manager import RiskManager

    event_engine = EventEngine()
    event_engine.start()

    gateway = MiniQmtGateway(event_engine)
    risk_manager = RiskManager(cfg.risk, event_engine)

    # 必须在 connect 之前构造 —— connect 内部会 query_account 并发出
    # EVENT_ACCOUNT，而注册该事件处理器的正是 LiveEngine。顺序反了的话
    # 账户快照发出时没人接，risk_manager.account 永远是 None，
    # 于是每一笔买单都被判「可用资金不足」——哪怕账上有一千万。
    engine = LiveEngine(event_engine, gateway, risk_manager)

    print(f"\n连接 miniQMT...")
    print(f"  路径: {cfg.gateway.qmt_path}")
    print(f"  账号: {cfg.gateway.account_id}")

    connected = gateway.connect({
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
        "account_type": cfg.gateway.account_type,
        "reconnect_max_retry": cfg.gateway.reconnect_max_retry,
        "reconnect_base_delay": cfg.gateway.reconnect_base_delay,
        "request_timeout": cfg.gateway.request_timeout,
    })
    if not connected:
        print("连接失败！请确认 miniQMT 已启动并登录")
        event_engine.stop()
        return 1
    print("  连接成功")

    if args.dry_run:
        risk_manager.activate_kill_switch("dry-run 模式，只观察不下单")

    # 加载策略
    from strategies.intraday_gbm.strategy import IntradayGBMStrategy

    print("取涨停价...")
    limit_up = fetch_limit_up_prices(symbols)
    print(f"  {len(limit_up)}/{len(symbols)} 只拿到涨停价")

    setting = {
        "trade_mode": args.trade_mode,
        "max_positions": args.max_positions,
        "position_size": position_size,
        "use_rank": True,
        "max_intraday_loss": 0.02,
        "entry_time": "09:35",
        "exit_time": "14:50",
        "vol_z_threshold": args.vol_z,
        "avoid_limit_up": not args.allow_limit_up,
        "limit_up_prices": limit_up,
    }

    engine.add_strategy(
        IntradayGBMStrategy,
        "INTRADAY_GBM",
        symbols,
        setting,
    )

    engine.init_all()
    engine.start_all()

    # 必须在 start_all 之后 —— on_start 会清空缓冲区
    strategy = engine.strategies.get("INTRADAY_GBM")
    if strategy is not None:
        warmup_buffers(strategy, symbols)

    print(f"\n引擎已启动（{'dry-run' if args.dry_run else '实盘'}），Ctrl+C 退出")
    print(f"  入场时间: 09:35 ~ 14:50")
    print(f"  单票止损: 2%")

    # 主循环
    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    for _sig in ("SIGBREAK", "SIGTERM"):
        h = getattr(signal, _sig, None)
        if h is None:
            continue
        try:
            signal.signal(h, _stop)
        except (OSError, ValueError):
            pass

    from webui import services
    try:
        tick = 0
        while running:
            time.sleep(1)
            tick += 1
            if tick % 30 == 0 and event_engine.is_active():
                services.beat("live")
    finally:
        print("\n正在退出：撤单 → 停策略 → 断开网关 ...")
        engine.close()
        event_engine.stop()
        print("已安全退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
