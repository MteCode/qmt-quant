"""策略注册表 —— 把「策略是什么 / 怎么回测 / 结果在哪 / 是否在跑」统一起来。

## 为什么需要这层

在此之前，管理台的页面是按「有什么产物」堆的：有 ensemble_result.json
就做一个页面读它，有 equity.csv 就再做一个。结果是同一个策略的信息散在
三四个页面里，而「这个策略到底是干什么的」根本没地方看。

这里反过来，以**策略**为中心组织：每个策略声明自己的说明、回测入口、
结果位置、实盘信号来源。页面只管渲染，不再各自 hardcode 文件路径。

## 结果格式统一

不同策略的产物格式差异很大（JSON / CSV / 嵌套结构）。每个策略提供一个
loader，把自己的产物翻译成统一的 dict，页面据此渲染同一套图表：

    {"has_result": bool, "metrics": {...}, "equity": {...},
     "period": str, "source": str, "note": str}

缺产物时返回 has_result=False，页面显示「未回测」并给出触发按钮 ——
不编造数字，也不因为缺文件就白屏。

## 实盘状态

`live_status()` 只读快照文件，不连 miniQMT。连接要 QMT 在线、会阻塞请求、
失败时整个页面打不开。没有快照就如实显示「未运行」。
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- 工具

def _read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    try:
        with p.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _norm_dd(v) -> float | None:
    """回撤统一为负数。

    各产物的符号约定不一致：ensemble_result.json 存正数 0.1706，
    t0 summary 存负数 -0.2674。不在这里统一，页面就得逐处判断，
    迟早有一处漏掉，把 -27% 显示成 +27%。
    """
    if v is None:
        return None
    try:
        return -abs(float(v))
    except (TypeError, ValueError):
        return None


def _metrics_from_equity(dates: list, values: list[float],
                         capital: float | None = None) -> dict:
    """从净值序列算统一指标。年化按 244 交易日折算。"""
    if len(values) < 2:
        return {}
    init = capital or values[0]
    total = values[-1] / init - 1
    n = len(values)
    peak, mdd = values[0], 0.0
    rets = []
    for i, v in enumerate(values):
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
        if i:
            prev = values[i - 1]
            if prev:
                rets.append(v / prev - 1)
    af = 244 / n if n else 0
    ann = (1 + total) ** af - 1 if total > -1 else -1.0
    if len(rets) > 1:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        vol = math.sqrt(var) * math.sqrt(244)
    else:
        vol = 0.0
    return {
        "total_return": total,
        "annual_return": ann,
        "max_drawdown": mdd,
        "sharpe": (ann / vol) if vol > 1e-9 else 0.0,
        "volatility": vol,
        "n_periods": n,
    }


# --------------------------------------------------------------- 数据结构

@dataclass
class Strategy:
    id: str
    name: str
    category: str                 # 日内 / 日频 / 组合 / 研究
    summary: str                  # 一句话说明，列表页显示
    how: list[str] = field(default_factory=list)     # 展开后的工作原理
    inputs: list[str] = field(default_factory=list)  # 依赖的数据
    risk: list[str] = field(default_factory=list)    # 风控约束
    code: str = ""                # 主要代码位置
    backtest_task: str = ""       # registry.py 中的 task_id
    live_task: str = ""           # 实盘/信号生成的 task_id
    status: str = "research"      # research / backtest_only / live_ready
    caveat: str = ""              # 已知问题或结论，红字显示

    #: 返回统一格式的回测结果
    loader: object = None

    def result(self) -> dict:
        if self.loader is None:
            return {"has_result": False}
        try:
            r = self.loader()
        except Exception as e:                      # noqa: BLE001
            return {"has_result": False, "error": str(e)}
        return r or {"has_result": False}


# --------------------------------------------------------------- 各策略 loader

def _load_intraday_gbm() -> dict:
    d = ROOT / "models" / "intraday_gbm"
    metrics = _read_json(d / "metrics.json")
    bt = _read_json(d / "backtest" / "summary.json")
    if not metrics and not bt:
        return {"has_result": False}

    out = {"has_result": True, "source": str(d.relative_to(ROOT)),
           "model": {}, "modes": []}
    if metrics:
        out["model"] = {
            "test_auc": metrics.get("test_auc"),
            "test_accuracy": metrics.get("test_accuracy"),
            "train_auc": metrics.get("train_auc"),
            "n_features": metrics.get("n_features"),
            "train_samples": metrics.get("train_samples"),
            "horizon": metrics.get("horizon_bars"),
            "period": " ~ ".join(
                [str(x)[:10] for x in metrics.get("train_date_range", [])]),
        }
        out["importance"] = _read_csv(d / "feature_importance.csv")[:10]

    if bt and bt.get("results"):
        for mode, r in bt["results"].items():
            tg = r.get("targets", {})
            out["modes"].append({
                "mode": mode,
                "total_return": r.get("total_return"),
                "annual_return": r.get("annual_return"),
                "monthly_return": r.get("monthly_return"),
                "max_drawdown": _norm_dd(r.get("max_drawdown")),
                "sharpe": r.get("sharpe"),
                "n_trades": r.get("n_trades"),
                "win_rate": r.get("win_rate"),
                "targets_passed": sum(1 for v in tg.values() if v.get("pass")),
                "targets_total": len(tg),
            })
        out["config"] = bt.get("config", {})
        out["generated_at"] = bt.get("generated_at")
        # 取第一个模式作为主指标
        if out["modes"]:
            m = out["modes"][0]
            out["metrics"] = {
                "total_return": m["total_return"],
                "annual_return": m["annual_return"],
                "max_drawdown": m["max_drawdown"],
                "sharpe": m["sharpe"],
            }
    return out


def _load_t0_single() -> dict:
    d = ROOT / "models" / "t0_single"
    s = _read_json(d / "600711_summary.json")
    if not s:
        return {"has_result": False}
    pf = s.get("performance", {})
    rows = _read_csv(d / "600711_best_daily.csv")
    eq = {"dates": [r["date"] for r in rows],
          "values": [float(r["equity"]) for r in rows],
          "close": [float(r["close"]) for r in rows]} if rows else None
    return {
        "has_result": True,
        "source": str((d / "600711_summary.json").relative_to(ROOT)),
        "symbol": s.get("symbol"),
        "period": " ~ ".join(s.get("date_range", [])),
        "metrics": {
            "total_return": pf.get("total_return"),
            "annual_return": pf.get("annual_return"),
            "max_drawdown": _norm_dd(pf.get("max_drawdown")),
            "sharpe": pf.get("sharpe"),
        },
        "alpha_beta": {
            "t_annual": pf.get("t_annual"),
            "base_return": pf.get("base_return"),
            "buyhold_annual": pf.get("buyhold_annual"),
        },
        "robustness": s.get("robustness", {}),
        "walkforward": s.get("walkforward", []),
        "cost": s.get("cost_model", {}),
        "conclusion": s.get("conclusion", {}),
        "equity": eq,
        "n_trades": pf.get("n_trades"),
        "win_rate": pf.get("win_rate"),
        "generated_at": s.get("generated_at"),
    }


def _load_t0_divergence() -> dict:
    d = ROOT / "models" / "t0_divergence"
    s = _read_json(d / "600711_summary.json")
    if not s:
        return {"has_result": False}
    pf = s.get("performance", {})
    rows = _read_csv(d / "600711_best_daily.csv")
    eq = {"dates": [r["date"] for r in rows],
          "values": [float(r["equity"]) for r in rows],
          "close": [float(r["close"]) for r in rows]} if rows else None
    return {
        "has_result": True,
        "source": str((d / "600711_summary.json").relative_to(ROOT)),
        "symbol": s.get("symbol"),
        "period": " ~ ".join(s.get("date_range", [])),
        "metrics": {
            "total_return": pf.get("total_return"),
            "annual_return": pf.get("annual_return"),
            "max_drawdown": _norm_dd(pf.get("max_drawdown")),
            "sharpe": pf.get("sharpe"),
        },
        "alpha_beta": {
            "t_annual": pf.get("t_annual"),
            "buyhold_annual": pf.get("buyhold_annual"),
        },
        "target": s.get("target_100pct", {}),
        "robustness": s.get("robustness", {}),
        "walkforward": s.get("walkforward", []),
        "best_params": s.get("best_params", {}),
        "equity": eq,
        "n_trades": pf.get("n_trades"),
        "coverage": pf.get("coverage"),
        "generated_at": s.get("generated_at"),
    }


def _load_alstm_ensemble() -> dict:
    d = ROOT / "strategies" / "alstm_ppo_csi1000" / "backtest"
    ens = _read_json(d / "ensemble_result.json")
    if not ens or not ens.get("ensemble"):
        return {"has_result": False}
    e = ens["ensemble"]
    cfg = ens.get("config", {})
    return {
        "has_result": True,
        "source": str((d / "ensemble_result.json").relative_to(ROOT)),
        "period": " ~ ".join(cfg.get("test", [])),
        "capital": cfg.get("capital"),
        "metrics": {
            "total_return": e.get("total_return"),
            "annual_return": e.get("annual_return"),
            "max_drawdown": _norm_dd(e.get("max_drawdown")),
            "sharpe": e.get("sharpe"),
        },
        "n_trades": e.get("total_trades"),
        "seeds": ens.get("singles", []),
        "config": cfg,
    }


def _load_report(name: str, capital: float = 1_000_000):
    """从 reports/<name>/equity.csv 读净值并算指标。"""
    def _fn():
        p = ROOT / "reports" / name / "equity.csv"
        rows = _read_csv(p)
        if not rows:
            return {"has_result": False}
        dcol = "date" if "date" in rows[0] else list(rows[0])[0]
        vcol = next((c for c in ("equity", "total", "value", "nav")
                     if c in rows[0]), list(rows[0])[-1])
        dates, vals = [], []
        for r in rows:
            try:
                vals.append(float(r[vcol]))
                dates.append(r[dcol][:10])
            except (ValueError, KeyError, TypeError):
                continue
        if len(vals) < 2:
            return {"has_result": False}
        return {
            "has_result": True,
            "source": str(p.relative_to(ROOT)),
            "period": f"{dates[0]} ~ {dates[-1]}",
            "capital": capital,
            "metrics": _metrics_from_equity(dates, vals, capital),
            "equity": {"dates": dates, "values": vals},
        }
    return _fn


def _load_intraday_920368() -> dict:
    d = ROOT / "strategies" / "intraday_t_920368"
    s = _read_json(d / "backtest" / "summary.json")
    if not s:
        return {"has_result": False}
    rows = _read_csv(d / "backtest" / "equity.csv")
    eq = None
    if rows:
        dcol = list(rows[0])[0]
        vcol = next((c for c in ("equity", "total", "value")
                     if c in rows[0]), list(rows[0])[-1])
        try:
            eq = {"dates": [r[dcol][:10] for r in rows],
                  "values": [float(r[vcol]) for r in rows]}
        except (ValueError, KeyError):
            eq = None
    return {
        "has_result": True,
        "source": str((d / "backtest" / "summary.json").relative_to(ROOT)),
        "metrics": {
            "total_return": s.get("total_return"),
            "annual_return": s.get("annual_return"),
            "max_drawdown": _norm_dd(s.get("max_drawdown")),
            "sharpe": s.get("sharpe"),
        },
        "n_trades": s.get("n_trades") or s.get("trades"),
        "period": s.get("period", ""),
        "equity": eq,
        "raw": s,
    }


# --------------------------------------------------------------- 注册表

STRATEGIES: list[Strategy] = [
    Strategy(
        id="intraday_gbm",
        name="全市场日内 GBM 选股",
        category="日内",
        summary="用 1 分钟 K 线训练横截面模型，判断当前时刻全市场哪些票适合日内操作，"
                "再按做T/均值回归/打板三种模式执行。",
        how=[
            "读取全市场 1m 清洗数据（已排除北交所、科创板、股价 >500 元的标的）",
            "计算 23 维日内特征：多周期动量、均线偏离、量能 z-score、ATR、"
            "RSI、VWAP 偏离、订单不平衡、Amihud 非流动性、日内位置、时间编码",
            "标签为「未来 N 根 bar 收益率是否超过阈值」，LightGBM 二分类",
            "按日期 70/30 分割，训练集与验证集不重叠，避免前视",
            "盘中对全市场打分排序，下游策略按概率筛选标的",
        ],
        inputs=["data/clean/1m/ 全市场分钟线"],
        risk=["单票日内止损", "组合回撤分档停止开仓", "尾盘强制平仓，不留隔夜"],
        code="strategies/intraday_gbm/strategy.py",
        backtest_task="backtest_intraday_gbm",
        live_task="predict_intraday",
        status="backtest_only",
        loader=_load_intraday_gbm,
    ),
    Strategy(
        id="t0_single",
        name="单票做 T（VWAP 偏离）",
        category="日内",
        summary="底仓 + 现金做 T+0 回转，以 VWAP 偏离为信号。"
                "已验证：样本外无 alpha。",
        how=[
            "严格实现 T+1 约束：区分 base_shares（可卖）与 locked_shares（今日买入锁定）",
            "正T = 低点用现金买入、高点从底仓卖出；反T = 高点先卖底仓、低点买回",
            "信号为价格相对日内累计 VWAP 的偏离幅度",
            "止盈/止损/尾盘强平三重出场",
            "内置样本外验证：前半段选参数，后半段检验",
        ],
        inputs=["单标的 1m 数据"],
        risk=["单笔止损", "最长持有时长", "尾盘强制平仓"],
        code="scripts/optimize_t0_600711.py",
        backtest_task="",
        status="research",
        caveat="3600 组参数中仅 5 组做T为正（0.1%），样本内前 10 组样本外无一存活。"
               "毛 edge 0.298% 对成本 0.25%，净 edge 仅 0.048%。不建议实盘。",
        loader=_load_t0_single,
    ),
    Strategy(
        id="t0_divergence",
        name="单票做 T（量价背离）",
        category="日内",
        summary="以量价背离为信号的 T+0 回转，测试 corr / OBV / 量能脉冲三类背离。",
        how=[
            "corr 背离：滚动窗口内价格变化与成交量变化的相关系数显著为负",
            "OBV 背离：能量潮与价格的短期斜率符号相反",
            "量能脉冲：量能 z-score 突增但价格反向走",
            "每类信号都测正向与反向（背离后是反转还是延续）",
            "加入最长持有时长，避免腿死等到尾盘",
        ],
        inputs=["单标的 1m 数据"],
        risk=["止盈止损", "最长持有时长", "尾盘强制平仓"],
        code="scripts/optimize_t0_divergence.py",
        backtest_task="",
        status="research",
        loader=_load_t0_divergence,
    ),
    Strategy(
        id="intraday_t_920368",
        name="920368 日内做 T",
        category="日内",
        summary="单票 LightGBM 做 T 模型，9 个基础特征预测短期方向。",
        how=[
            "针对 920368.BSE 单只标的训练",
            "9 维基础特征（动量、量能、波动）",
            "LightGBM 二分类预测未来短期方向",
        ],
        inputs=["920368 1m 数据"],
        risk=["单票止损", "尾盘平仓"],
        code="strategies/intraday_t_920368/strategy.py",
        backtest_task="",
        status="backtest_only",
        loader=_load_intraday_920368,
    ),
    Strategy(
        id="alstm_ensemble",
        name="ALSTM 八模型集成",
        category="日频",
        summary="ALSTM 多 seed 集成选股 + PPO 仓位管理，中证 1000 池。",
        how=[
            "8 个不同 seed 的 ALSTM 模型对全池打分",
            "集成打分排序选股，固定持仓数与调仓周期",
            "PPO 智能体负责仓位管理",
        ],
        inputs=["日线行情", "因子数据"],
        risk=["回撤控制", "单票权重上限"],
        code="strategies/alstm_ppo_csi1000/",
        backtest_task="train_ensemble",
        live_task="generate_signal",
        status="live_ready",
        loader=_load_alstm_ensemble,
    ),
    Strategy(
        id="lgb_enhanced",
        name="LightGBM 增强选股",
        category="日频",
        summary="LightGBM 因子模型选股，50 只持仓 / 20 日调仓。",
        how=[
            "多因子输入训练 LightGBM 排序模型",
            "按预测收益排序取前 N 只",
            "固定周期调仓",
        ],
        inputs=["日线行情", "财务数据"],
        risk=["回撤控制"],
        code="scripts/run_portfolio_backtest.py",
        backtest_task="sweep_portfolio",
        status="backtest_only",
        loader=_load_report("lgb_enhanced"),
    ),
    Strategy(
        id="qlib_ml",
        name="Qlib ML 基线",
        category="日频",
        summary="Qlib 框架下的机器学习选股基线，用于对照。",
        how=["Qlib 标准 Alpha158 因子集", "GBDT 模型", "TopK 选股"],
        inputs=["Qlib 格式行情"],
        risk=["标准回撤控制"],
        code="scripts/run_portfolio_backtest.py",
        backtest_task="",
        status="backtest_only",
        loader=_load_report("qlib_ml"),
    ),
]

BY_ID = {s.id: s for s in STRATEGIES}
CATEGORIES = ["日内", "日频", "组合", "研究"]


def list_strategies(category: str = "") -> list[dict]:
    """列表页数据：策略元信息 + 结果摘要。"""
    out = []
    for s in STRATEGIES:
        if category and s.category != category:
            continue
        r = s.result()
        out.append({"s": s, "r": r})
    return out


# --------------------------------------------------------------- 实盘

def live_status() -> dict:
    """实盘运行状态。只读快照，不连 miniQMT。

    从未运行过实盘时如实返回 running=[]，页面显示「未运行」，
    不编造持仓或收益。
    """
    out = {"running": [], "positions": None, "equity": None,
           "signals": [], "has_any": False, "state_dir": None}

    # 只认策略**自己**目录下的快照。曾经写过「找不到就退回 alstm 的目录」，
    # 结果是每个策略都显示「有快照」，实际读的是别人的数据 —— 实盘页面上
    # 这种张冠李戴比空白危险得多。
    #
    # 同样，目录存在 ≠ 跑过实盘：这些 state/ 目录是建仓库时就有的空壳。
    # 必须真的读到 positions.json 或 equity.csv 才算。
    for s in STRATEGIES:
        if not s.live_task:
            continue
        sd = ROOT / "strategies" / s.id / "state"
        if not sd.exists():
            continue
        pos = _read_json(sd / "positions.json")
        eq = _read_csv(sd / "equity.csv")
        if pos is None and not eq:
            continue
        out["running"].append({
            "id": s.id, "name": s.name,
            "positions": pos, "equity_rows": len(eq),
            "state_dir": str(sd.relative_to(ROOT)),
        })
        out["has_any"] = True
        out["state_dir"] = str(sd.relative_to(ROOT))

    return out


def live_capable() -> list[Strategy]:
    """具备实盘条件的策略（登记了 live_task）。"""
    return [s for s in STRATEGIES if s.live_task]
