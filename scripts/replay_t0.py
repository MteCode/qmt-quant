"""做 T 策略的单票分钟级回放。

## 为什么要单独写这个而不用回测引擎

做 T 的损益必须和**底仓自身的涨跌**分开算。底仓涨 5% 不是策略的功劳，
跌 5% 也不是策略的锅 —— 做 T 赚的只是「卖得比买回来贵」的那部分差价。

把两者混在一起算，得到的数字毫无意义：实测某次混算得出「4 万底仓做出
+265%」，那全是底仓涨幅，和做 T 无关。

## 损益口径

按**配对平仓**算：每一笔买回，和最早未配对的卖出配成一对，
差价 = (卖价 - 买价) × 股数。日终未配对的卖出按收盘价折回（相当于
尾盘必须买回），未配对的买入同理。

这样算出来的就是纯粹的回转收益，与底仓方向无关。

## T+1 账本

卖出减少可卖量，买入**不增加**可卖量 —— 当日买回的部分次日才解冻。
这是 A 股做 T 的核心约束，回放里必须如实模拟，否则会算出根本做不到的
操作次数。

用法::

    python scripts/replay_t0.py --top 10
    python scripts/replay_t0.py --symbols 600396.SSE,600094.SSE --grid 0.01
"""
import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RANKED = ROOT / "data" / "universe" / "t0_minute_ranked.csv"
MIN_DIR = ROOT / "data" / "clean" / "1m"

#: 双边成本：佣金万0.854×2 + 印花税千1（卖出单边）+ 滑点
COST_RATE = 0.0008


class ReplayEngine:
    """带 T+1 账本的最小引擎。"""

    def __init__(self, base_volume: float, cash: float = 1e8):
        self.base = base_volume
        self.pos = base_volume
        self.avail = base_volume          # 昨仓可卖
        self.cash = cash
        self.fills: list[tuple[str, float, float]] = []

    def get_pos(self, vt): return self.pos
    def get_available(self, vt): return self.avail
    def get_cost_price(self, vt): return 0.0
    def get_cash(self): return self.cash

    def send_order(self, strategy_name, vt_symbol, direction, price,
                   volume, order_type=None):
        # 签名与 StrategyBase._send_order 的调用保持一致，
        # 多一个参数就会整体错位且不报错
        is_sell = "SHORT" in str(direction).upper()
        if is_sell:
            v = min(volume, self.avail)
            if v <= 0:
                return "o"
            self.avail -= v
            self.pos -= v
            self.cash += v * price
            self.fills.append(("sell", price, v))
        else:
            v = volume
            if v <= 0 or v * price > self.cash:
                return "o"
            self.pos += v
            self.cash -= v * price
            # T+1：当日买入不解冻，avail 不增加
            self.fills.append(("buy", price, v))
        return "o"


def pair_pnl(fills, last_price: float) -> tuple[float, int]:
    """配对算回转差价，返回 (毛差价, 配对数)。

    先进先出配对：每笔买回和最早未配对的卖出配成一对。收盘时未配对的
    卖出按收盘价强制买回、未配对的买入按收盘价卖出 —— 与「尾盘回补底仓」
    的策略设定一致。
    """
    open_sells: deque = deque()
    open_buys: deque = deque()
    pnl = 0.0
    pairs = 0
    for side, px, vol in fills:
        if side == "sell":
            while vol > 0 and open_buys:
                bpx, bvol = open_buys[0]
                m = min(vol, bvol)
                pnl += (px - bpx) * m
                pairs += 1
                vol -= m
                if m >= bvol:
                    open_buys.popleft()
                else:
                    open_buys[0] = (bpx, bvol - m)
            if vol > 0:
                open_sells.append((px, vol))
        else:
            while vol > 0 and open_sells:
                spx, svol = open_sells[0]
                m = min(vol, svol)
                pnl += (spx - px) * m
                pairs += 1
                vol -= m
                if m >= svol:
                    open_sells.popleft()
                else:
                    open_sells[0] = (spx, svol - m)
            if vol > 0:
                open_buys.append((px, vol))

    # 未配对的按收盘价轧平
    for spx, svol in open_sells:
        pnl += (spx - last_price) * svol
    for bpx, bvol in open_buys:
        pnl += (last_price - bpx) * bvol
    return pnl, pairs


def replay_symbol(vt: str, args) -> dict | None:
    import pandas as pd

    from qmtquant.core.objects import BarData
    from strategies.intraday_t0.strategy import IntradayT0Strategy

    code, _, ex = vt.rpartition(".")
    f = MIN_DIR / ex / f"{code}.parquet"
    if not f.exists():
        return None
    m = pd.read_parquet(f, columns=["close", "volume"])
    days = sorted(set(m.index.strftime("%Y%m%d")))[-args.days:]
    if len(days) < 10:
        return None

    gross = 0.0
    n_fills = 0
    n_pairs = 0
    turnover = 0.0
    day_pnls = []

    for d in days:
        sub = m[m.index.strftime("%Y%m%d") == d]
        if len(sub) < 100:
            continue
        p0 = float(sub["close"].iloc[0])
        base_vol = int(args.base_value / p0 // 100 * 100)
        if base_vol <= 0:
            continue

        eng = ReplayEngine(base_vol)
        s = IntradayT0Strategy(eng, "t0", [vt], {
            "grid": args.grid,
            "max_layers": args.layers,
            "trade_value": args.trade_value,
            "exit_time": args.exit_time,
        })
        s.trading = True
        s.on_start()

        for ts, r in sub.iterrows():
            b = BarData(symbol=code, exchange=None,
                        datetime=ts.to_pydatetime(), gateway_name="rp")
            px = float(r["close"])
            b.close_price = b.open_price = b.high_price = b.low_price = px
            b.volume = float(r["volume"])
            b.turnover = 0.0
            s.on_bars({vt: b})

        last = float(sub["close"].iloc[-1])
        pnl, pairs = pair_pnl(eng.fills, last)
        gross += pnl
        day_pnls.append(pnl)
        n_pairs += pairs
        n_fills += len(eng.fills)
        turnover += sum(px * v for _, px, v in eng.fills)

    if not day_pnls:
        return None
    cost = turnover * COST_RATE
    net = gross - cost
    wins = sum(1 for p in day_pnls if p > 0)
    return {
        "vt_symbol": vt,
        "days": len(day_pnls),
        "fills": n_fills,
        "fills_per_day": n_fills / len(day_pnls),
        "gross": gross,
        "cost": cost,
        "net": net,
        "net_pct": net / args.base_value,
        "win_days": wins / len(day_pnls),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="做 T 策略分钟级回放")
    p.add_argument("--symbols", default="", help="逗号分隔，留空则用排名前 N")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--base-value", type=float, default=40000.0,
                   help="底仓金额。20 万 / 5 只 = 4 万，符合单票 ≤20%%")
    p.add_argument("--trade-value", type=float, default=40000.0)
    p.add_argument("--grid", type=float, default=0.008)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--exit-time", default="14:45")
    args = p.parse_args()

    import pandas as pd

    if args.symbols:
        syms = [x.strip() for x in args.symbols.split(",") if x.strip()]
        names = {}
    else:
        if not RANKED.exists():
            print(f"缺少 {RANKED}，先跑 screen_t0_minute.py")
            return 1
        df = pd.read_csv(RANKED, encoding="utf-8-sig")
        syms = df["vt_symbol"].astype(str).tolist()[: args.top]
        names = dict(zip(df["vt_symbol"].astype(str),
                         df.get("name", pd.Series(dtype=str)).astype(str)))

    print("=" * 82)
    print("做 T 回放（配对损益，已剔除底仓自身涨跌）")
    print("=" * 82)
    print(f"  底仓 {args.base_value:,.0f}   格距 {args.grid:.2%}   "
          f"分 {args.layers} 档   回看 {args.days} 日")
    print(f"  成本按双边 {COST_RATE:.2%}（佣金+印花税+滑点）")
    print()
    print(f"  {'标的':<14}{'名称':<10}{'笔/日':>7}{'毛差价':>10}"
          f"{'成本':>10}{'净收益':>10}{'占底仓':>9}{'盈利日':>8}")

    rows = []
    for vt in syms:
        r = replay_symbol(vt, args)
        if not r:
            continue
        rows.append(r)
        print(f"  {vt:<14}{str(names.get(vt,''))[:8]:<10}"
              f"{r['fills_per_day']:>7.1f}{r['gross']:>10.0f}"
              f"{r['cost']:>10.0f}{r['net']:>+10.0f}"
              f"{r['net_pct']:>+8.2%}{r['win_days']:>8.0%}")

    if rows:
        import statistics
        print()
        print("-" * 82)
        med = statistics.median(r["net_pct"] for r in rows)
        pos = sum(1 for r in rows if r["net"] > 0)
        print(f"  {len(rows)} 只中位净收益 {med:+.2%}（占底仓，{args.days} 日）"
              f"   盈利标的 {pos}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
