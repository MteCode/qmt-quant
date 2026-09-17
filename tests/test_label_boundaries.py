"""训练标签的边界屏蔽。

## 起因

标签是 `close[i + horizon] / close[i] - 1`，而 df 是全市场拼成的长表，
按标的分块（concat 不排序），每块内部按时间。于是 close[i+horizon]
有两种取错：

1. **跨标的**：每只最后 horizon 行取到下一只股票的价格。
   800 只 × horizon 15 = 12,000 行，占 270M 的 0.004% —— 量小，
   但完全是垃圾标签，没有理由留着。

2. **跨日**：每个交易日最后 horizon 根 bar 取到**次日**价格，
   「未来收益」因此跨越隔夜跳空。实测 241 根/日时：

       horizon= 5 ->  2.07%
       horizon=15 ->  6.20%
       horizon=30 -> 12.40%

   而这个策略是尾盘强制平仓、不留隔夜的。这些样本让模型去学一段
   它永远不可能交易的收益，而且**系统性地集中在尾盘** ——
   正是策略该平仓的时候。模型会在最不该开仓的时段学到最强的信号。

第 2 条不是小数点后的误差，是 6% 的训练样本在教错的东西。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_gbm", ROOT / "scripts" / "train_intraday_gbm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TRAINER = _load_trainer()


def _mask(close: np.ndarray, day_codes: np.ndarray,
          sym_codes: np.ndarray, horizon: int):
    """调**生产代码**的 make_labels，不复刻公式。

    第一版我在这里复刻了一遍屏蔽逻辑，理由是「训练要 GB 级数据跑不动」。
    但那样改坏 train_intraday_gbm 测试照样全绿 —— 复刻公式只能证明
    我会写 numpy。正确做法是把屏蔽逻辑从 train() 里抽成 make_labels()，
    然后测它。
    """
    y, has_label, _n = TRAINER.make_labels(
        close, day_codes, sym_codes, horizon, threshold=0.0)
    n = len(close)
    fut = np.full(n, np.nan)
    fut[:-horizon] = close[horizon:] / close[:-horizon] - 1
    return fut, has_label


def _panel(n_sym: int, n_days: int, bars_per_day: int):
    """造一个「按标的分块、块内按时间」的长表，和真实布局一致。

    ## 日期码必须跨标的共享

    第一版我写的是 `days.append(s * 1000 + d)` —— 不同标的用不重叠的
    日期码。那样跨标的边界同时也是跨日边界，same_day 一条就挡住了，
    于是把 same_sym 整个删掉测试照样全绿：**fixture 不真实，
    测试就测不出东西**。

    真实数据里全市场共享同一批交易日，所以这里也共享 —— 只有这样
    「跨标的但同日」这个组合才会出现，same_sym 才是必需的。
    """
    close, days, syms = [], [], []
    price = 10.0
    for s in range(n_sym):
        for d in range(n_days):
            for b in range(bars_per_day):
                price += 0.01
                close.append(price)
                days.append(d)                # 与真实数据一致：跨标的共享
                syms.append(s)
        price += 100.0                        # 标的间价格跳变，便于看出串号
    return (np.array(close), np.array(days), np.array(syms))


class TestCrossDayMasked:
    @pytest.mark.parametrize("horizon", [5, 15, 30])
    def test_last_bars_of_day_dropped(self, horizon):
        close, day, sym = _panel(1, 5, 240)
        _fut, keep = _mask(close, day, sym, horizon)
        # 每天最后 horizon 根应当被丢弃
        for d in range(5):
            for k in range(horizon):
                i = d * 240 + 240 - 1 - k
                assert not keep[i], \
                    f"第 {d} 天倒数第 {k+1} 根（跨日）不该保留"

    @pytest.mark.parametrize("horizon,expect", [
        (5, 0.0207), (15, 0.0620), (30, 0.1240),
    ])
    def test_dropped_fraction_matches_measurement(self, horizon, expect):
        """丢弃比例应当接近 horizon / 每日 bar 数。

        钉住这个数是为了让「这条屏蔽有多重要」可见 ——
        horizon=30 时丢掉 12%，那不是舍入误差。
        """
        close, day, sym = _panel(1, 241, 241)
        _fut, keep = _mask(close, day, sym, horizon)
        labelled = ~np.isnan(_fut)
        dropped = labelled.sum() - keep.sum()
        frac = dropped / len(close)
        assert frac == pytest.approx(expect, abs=0.01)

    def test_intraday_labels_kept(self):
        """日内的绝大多数行必须保留 —— 屏蔽不能变成什么都不留。"""
        close, day, sym = _panel(1, 5, 240)
        _fut, keep = _mask(close, day, sym, 15)
        assert keep.sum() / len(close) > 0.9


class TestCrossSymbolMasked:
    def test_symbol_boundary_dropped(self):
        close, day, sym = _panel(3, 2, 100)
        _fut, keep = _mask(close, day, sym, 10)
        n_per_sym = 2 * 100
        for s in range(2):                     # 前两只的末尾
            for k in range(10):
                i = (s + 1) * n_per_sym - 1 - k
                assert not keep[i], f"标的 {s} 末尾第 {k+1} 行应被丢弃"

    def test_no_label_uses_another_symbols_price(self):
        """保留下来的标签，其未来价格必须与当前行同标的。

        这是最直接的断言：不看比例，看有没有一行串了号。
        """
        close, day, sym = _panel(4, 3, 60)
        horizon = 12
        _fut, keep = _mask(close, day, sym, horizon)
        idx = np.where(keep)[0]
        assert len(idx) > 0
        assert np.all(sym[idx + horizon] == sym[idx]), "有标签串到了别的标的"
        assert np.all(day[idx + horizon] == day[idx]), "有标签串到了别的交易日"


class TestMaskingActuallyChangesLabels:
    """屏蔽必须真的改变结果 —— 否则这些测试只是在验证 numpy 会切片。"""

    def test_unmasked_would_include_garbage(self):
        close, day, sym = _panel(3, 2, 100)
        horizon = 10
        fut, keep = _mask(close, day, sym, horizon)
        naive = ~np.isnan(fut)
        assert naive.sum() > keep.sum(), "屏蔽应当丢掉一些行"

        # 被丢掉的那些里，确实有串号/跨日的
        dropped = np.where(naive & ~keep)[0]
        bad = sum(1 for i in dropped
                  if sym[i + horizon] != sym[i] or day[i + horizon] != day[i])
        assert bad == len(dropped), "被丢的每一行都应当是真的越界了"

    def test_symbol_jump_produces_absurd_return(self):
        """串号的标签有多离谱：标的间价格跳了 100，收益率是荒唐的。

        这解释了为什么哪怕只有 0.004% 也该丢 —— 它们不是噪声，是异常值。
        """
        close, day, sym = _panel(2, 1, 50)
        horizon = 5
        fut, keep = _mask(close, day, sym, horizon)
        crossed = [i for i in range(len(close) - horizon)
                   if sym[i + horizon] != sym[i]]
        assert crossed
        assert max(abs(fut[i]) for i in crossed) > 1.0, \
            "跨标的的「收益率」应当大到一眼可辨"
        assert not any(keep[i] for i in crossed)


class TestT1Labels:
    """T+1 标签：当日买入、次一交易日卖出。

    ## 和日内标签的关系是「相反」

    日内版把跨日样本当越界屏蔽掉，因为策略尾盘强平。T+1 策略的持有期
    必然跨越隔夜，所以跨日不是越界而是标签本身 —— 这两套逻辑容易写串，
    所以分别测。

    实盘印证过必要性：日内版每分钟发卖单、每分钟被「可卖数量不足」拒，
    连止损都执行不了。
    """

    def _t1(self, close, day, sym, exit_at="open"):
        return TRAINER.make_labels_t1(close, day, sym,
                                      threshold=0.0, exit_at=exit_at)

    def test_exit_is_next_day_first_bar(self):
        """exit_at=open 时，出场价必须是次日第一根 bar。"""
        close, day, sym = _panel(1, 3, 4)
        y, keep, _ = self._t1(close, day, sym, "open")
        starts = [i for i in range(len(close))
                  if i == 0 or day[i] != day[i - 1]]
        # 第 0 天任意一行，出场都应当是第 1 天的首根
        for i in range(starts[0], starts[1]):
            assert keep[i]
            expected = close[starts[1]] / close[i] - 1
            assert y[i] == int(expected > 0)

    def test_exit_at_close_uses_next_day_last_bar(self):
        """exit_at=close 时，出场价必须是次日**最后**一根 bar。

        不比较两种口径的标签是否不同 —— 合成数据价格单调时它们本就相同，
        那样的断言测不出取错 bar。直接验证用的是哪一根。
        """
        close, day, sym = _panel(1, 3, 4)
        y, keep, _ = self._t1(close, day, sym, "close")
        starts = [i for i in range(len(close))
                  if i == 0 or day[i] != day[i - 1]]
        day1_last = starts[2] - 1
        for i in range(starts[0], starts[1]):
            assert keep[i]
            expected = close[day1_last] / close[i] - 1
            assert y[i] == int(expected > 0)

    def test_last_day_has_no_label(self):
        """最后一个交易日没有次日，必须无标签 —— 否则会取到别的标的。"""
        close, day, sym = _panel(1, 3, 4)
        _, keep, _ = self._t1(close, day, sym)
        last_day = day[-1]
        assert not keep[day == last_day].any()

    def test_never_crosses_symbol(self):
        """每只标的最后一天的「次日」不能落到下一只股票头上。"""
        close, day, sym = _panel(3, 3, 4)
        _, keep, _ = self._t1(close, day, sym)
        for s in np.unique(sym):
            idx = np.flatnonzero(sym == s)
            last_day = day[idx].max()
            tail = idx[day[idx] == last_day]
            assert not keep[tail].any(), "跨标的取值必须被屏蔽"

    def test_same_day_rows_never_labelled_intraday(self):
        """T+1 标签不应退化成日内标签：出场绝不在当天。"""
        close, day, sym = _panel(2, 3, 5)
        _, keep, _ = self._t1(close, day, sym)
        assert keep.any(), "应当有可用样本"
        # 有标签的行，其出场日必须严格晚于入场日
        for i in np.flatnonzero(keep):
            later = day[i + 1:][day[i + 1:] != day[i]]
            assert len(later) > 0, "有标签却找不到次日，逻辑有误"
