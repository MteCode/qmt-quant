"""强化学习 PPO 择时 —— RL 只控制仓位水平，选股用截面因子排序。

思路：
- 选股：每 20 天用截面因子（动量、波动率等）排序，等权选 Top-30
- 择时：RL 决定整体仓位水平 [0, 1]，1 = 满仓，0 = 空仓
- State：市场聚合特征（均值动量、均值波动率、近期收益等）—— 低维
- Action：1 维连续 [0, 1]，学习效率高

用法::

    python strategies/alstm_ppo_csi1000/train_ppo.py
    python strategies/alstm_ppo_csi1000/train_ppo.py --timesteps 200000
"""
import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

TRAIN = ("2016-01-01", "2019-12-31")
VALID = ("2020-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-08-27")

HOLD_K = 30
REBAL_PERIOD = 20


def prepare_data(market: str):
    """准备日频数据：收盘价 + 截面因子 + 次日收益。"""
    from qlib.data import D

    from qmtquant.config import get_config
    cfg = get_config()
    qlib_dir = Path(cfg.data.store_dir) / "qlib_data"

    instruments = D.instruments(market)

    print("提取收盘价...")
    close_df = D.features(
        instruments, ["$close"],
        start_time=TRAIN[0], end_time=TEST[1],
    )
    close_df.columns = ["close"]

    factor_exprs = [
        "$close/Ref($close,1)-1",       # 日收益
        "$close/Ref($close,5)-1",        # 5 日动量
        "$close/Ref($close,20)-1",       # 20 日动量
        "Std($close,5)/Mean($close,5)",  # 5 日波动率
        "Std($close,20)/Mean($close,20)",  # 20 日波动率
        "$volume/Ref($volume,1)-1",      # 成交量变化
        "Mean($volume,5)/Mean($volume,20)-1",  # 量比
        "($high-$low)/$close",           # 振幅
        "($close-$low)/($high-$low+1e-8)",  # 收盘位置
        "Mean($close,5)/Mean($close,20)-1",  # 均线偏离
    ]
    factor_names = [
        "ret1", "mom5", "mom20", "vol5", "vol20",
        "vol_chg", "vol_ratio", "amplitude", "close_pos", "ma_dev",
    ]

    print("提取因子特征...")
    feat_df = D.features(
        instruments, factor_exprs,
        start_time=TRAIN[0], end_time=TEST[1],
    )
    feat_df.columns = factor_names

    # 加载大资金因子
    factor_dir = qlib_dir / "money_factors"
    for name in ["dragon_count_20"]:
        path = factor_dir / f"{name}.parquet"
        if path.exists():
            s = pd.read_parquet(path).iloc[:, 0].swaplevel().sort_index()
            aligned = s.reindex(feat_df.index)
            feat_df[name] = aligned.values
            print(f"  {name}: {aligned.notna().sum():,} 有效值")

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    # 截面 ZScore
    print("截面 ZScore 标准化...")
    feat_df = feat_df.groupby(level="datetime").transform(
        lambda x: ((x - x.mean()) / (x.std() + 1e-8)).clip(-3, 3)
    ).fillna(0)

    # 次日收益率
    label_df = D.features(
        instruments, ["Ref($close,-1)/$close-1"],
        start_time=TRAIN[0], end_time=TEST[1],
    )
    label_df.columns = ["ret_next"]

    return close_df, feat_df, label_df


class TimingEnv(gymnasium.Env):
    """RL 择时环境：action = 仓位水平 [0,1]，选股用因子排序。

    State (15 维)：
    - 市场聚合：截面均值/中位数的动量、波动率、量比等
    - 组合状态：累计收益、近 5 日收益、回撤、当前仓位
    """
    metadata = {"render_modes": []}

    def __init__(self, dates, daily_features, daily_returns,
                 hold_k=HOLD_K, rebal_period=REBAL_PERIOD,
                 initial_amount=500_000,
                 buy_cost=0.0005, sell_cost=0.0015,
                 scores=None):
        super().__init__()

        self.dates = dates
        self.daily_features = daily_features
        self.daily_returns = daily_returns
        self.hold_k = hold_k
        self.rebal_period = rebal_period
        self.initial_amount = initial_amount
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.scores = scores

        self.n_features = next(iter(daily_features.values())).shape[1]

        # State: 市场聚合(n_features * 2) + 组合状态(5)
        self.state_dim = self.n_features * 2 + 5
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32,
        )
        # Action: 仓位水平 [0, 1]
        self.action_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32,
        )

        self.day = 0
        self.portfolio_value = initial_amount
        self.exposure = 0.0
        self.held_stocks = []
        self.asset_memory = []

    def _select_stocks(self, date):
        """有分数用分数选 Top-K，无分数用等权全市场"""
        if self.scores is not None:
            ts = pd.Timestamp(date)
            pos = self.scores.index.searchsorted(ts, side="right") - 1
            if pos >= 0:
                row = self.scores.iloc[pos].dropna().sort_values(ascending=False)
                # scores 列名可能是 vt 格式(600007.SSE)，转成 Qlib(sh600007)
                top_vt = list(row.index[:self.hold_k])
                if top_vt and "." in str(top_vt[0]):
                    from qmtquant.datafeed.qlib_export import to_qlib_code
                    if not hasattr(self, "_vt2qlib"):
                        self._vt2qlib = {}
                        for c in self.scores.columns:
                            try:
                                self._vt2qlib[c] = to_qlib_code(c)
                            except ValueError:
                                pass
                    return [self._vt2qlib.get(s, s) for s in top_vt
                            if s in self._vt2qlib]
                return top_vt

        if date not in self.daily_returns:
            return []
        rets = self.daily_returns[date].dropna()
        valid = rets[rets.abs() < 0.11].index
        return list(valid)

    def _market_state(self, date):
        """市场聚合特征：截面均值 + 截面标准差"""
        if date not in self.daily_features:
            return np.zeros(self.n_features * 2, dtype=np.float32)
        feat = self.daily_features[date]
        means = feat.mean().values.astype(np.float32)
        stds = feat.std().values.astype(np.float32)
        return np.concatenate([means, stds])

    def _portfolio_state(self):
        total_ret = self.portfolio_value / self.initial_amount - 1
        if len(self.asset_memory) >= 5:
            recent = np.array(self.asset_memory[-5:])
            ret_5d = recent[-1] / recent[0] - 1
            vol_5d = np.std(np.diff(recent) / recent[:-1])
        else:
            ret_5d = 0.0
            vol_5d = 0.0
        peak = max(self.asset_memory) if self.asset_memory else self.initial_amount
        dd = (peak - self.portfolio_value) / peak
        return np.array([total_ret, ret_5d, vol_5d, dd, self.exposure],
                        dtype=np.float32)

    def _get_state(self, date):
        mkt = self._market_state(date)
        port = self._portfolio_state()
        return np.concatenate([mkt, port])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.exposure = 0.0
        self.held_stocks = []
        self.asset_memory = [self.initial_amount]
        return self._get_state(self.dates[0]), {}

    def step(self, action):
        date = self.dates[self.day]
        new_exposure = float(np.clip(action[0], 0, 1))

        # 每天更新持仓（等权全市场，无选股成本）
        self.held_stocks = self._select_stocks(date)

        # 仓位变化成本
        exposure_change = abs(new_exposure - self.exposure)
        cost = exposure_change * (self.buy_cost + self.sell_cost) / 2
        self.portfolio_value *= (1 - cost)
        self.exposure = new_exposure

        # 等权组合收益
        returns = self.daily_returns.get(date, pd.Series(dtype=float))
        port_ret = 0.0
        n_valid = 0
        for stock in self.held_stocks:
            if stock in returns.index:
                r = returns[stock]
                if np.isfinite(r) and abs(r) < 0.11:
                    port_ret += r
                    n_valid += 1

        if n_valid > 0:
            port_ret /= n_valid  # 等权

        # 按仓位水平缩放收益
        actual_ret = self.exposure * port_ret
        self.portfolio_value *= (1 + actual_ret)
        self.asset_memory.append(self.portfolio_value)

        self.day += 1
        terminated = self.day >= len(self.dates) - 1

        if not terminated:
            state = self._get_state(self.dates[self.day])
        else:
            state = np.zeros(self.state_dim, dtype=np.float32)

        # Reward: 风险调整后收益
        reward = actual_ret * 100

        # 回撤惩罚
        peak = max(self.asset_memory)
        drawdown = (peak - self.portfolio_value) / peak
        if drawdown > 0.10:
            reward -= (drawdown - 0.10) * 20
        if drawdown > 0.18:
            reward -= (drawdown - 0.18) * 50

        return state, float(reward), terminated, False, {
            "portfolio_value": self.portfolio_value,
            "daily_return": actual_ret,
            "drawdown": drawdown,
            "exposure": self.exposure,
        }


def build_env(close_df, feat_df, label_df, start, end, scores=None, **kwargs):
    idx = feat_df.index.get_level_values("datetime")
    mask = (idx >= start) & (idx <= end)

    feat_sub = feat_df[mask]
    label_sub = label_df[mask]

    dates = sorted(feat_sub.index.get_level_values("datetime").unique())

    daily_features = {}
    daily_returns = {}
    for date in dates:
        if date in feat_sub.index.get_level_values("datetime"):
            daily_features[date] = feat_sub.xs(date, level="datetime")
        if date in label_sub.index.get_level_values("datetime"):
            daily_returns[date] = label_sub.xs(date, level="datetime")["ret_next"]

    return TimingEnv(
        dates=dates,
        daily_features=daily_features,
        daily_returns=daily_returns,
        scores=scores,
        **kwargs,
    )


def main():
    p = argparse.ArgumentParser(description="RL PPO 择时")
    p.add_argument("--market", default="csi1000")
    p.add_argument("--capital", type=float, default=500_000)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--report", default=str(paths.BACKTEST_DIR))
    p.add_argument("--scores", default=None,
                    help="选股分数面板路径，默认用本策略的 ALSTM 分数")
    args = p.parse_args()

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = str(Path(cfg.data.store_dir) / "qlib_data")

    from qmtquant.datafeed.qlib_init import init_qlib
    init_qlib(uri, n_expressions=32)

    print("=" * 62)
    print("强化学习 PPO 择时（stable-baselines3）")
    print("=" * 62)
    print(f"市场: {args.market}  本金: {args.capital:,.0f}")
    print(f"训练: {TRAIN[0]}~{VALID[1]}  测试: {TEST[0]}~{TEST[1]}")
    print(f"持仓: {HOLD_K} 只  调仓: 每 {REBAL_PERIOD} 日")
    print(f"RL Action: 仓位水平 [0, 1]（1 维）")
    print(f"训练步数: {args.timesteps:,}\n")

    t0 = time.time()

    close_df, feat_df, label_df = prepare_data(args.market)
    print(f"数据准备完成: {feat_df.shape[0]:,} 行 x {feat_df.shape[1]} 列\n")

    print("构建训练环境...")
    train_env = build_env(
        close_df, feat_df, label_df,
        TRAIN[0], VALID[1],
        initial_amount=args.capital,
    )
    print(f"  训练日数: {len(train_env.dates)}")
    print(f"  State dim: {train_env.state_dim}")
    print(f"  Action dim: 1（仓位水平）\n")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    class LogCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_count = 0

        def _on_step(self):
            if self.locals.get("dones") is not None and any(self.locals["dones"]):
                self.episode_count += 1
                info = self.locals.get("infos", [{}])[0]
                pv = info.get("portfolio_value", 0)
                dd = info.get("drawdown", 0)
                exp = info.get("exposure", 0)
                ret = pv / args.capital - 1 if pv else 0
                if self.episode_count % 10 == 0 or self.episode_count <= 5:
                    print(f"  Episode {self.episode_count}: "
                          f"终值 {pv:,.0f}  收益 {ret:+.2%}  "
                          f"回撤 {dd:.2%}  末仓位 {exp:.1%}")
            return True

    print("训练 PPO ...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=42,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps, callback=LogCallback())

    train_time = time.time() - t0
    print(f"\n训练完成，耗时 {train_time:.0f}s")

    # 测试：加载 ALSTM 分数做选股
    print("\n" + "-" * 62)
    print("测试段回测（样本外）—— ALSTM 选股 + PPO 择时")
    print("-" * 62)

    # 分数来源可切换 —— 第 1 层换成 LightGBM 后需要指向它的产出
    scores_path = Path(args.scores) if args.scores else paths.ALSTM_SCORES
    alstm_scores = None
    if scores_path.exists():
        alstm_scores = pd.read_parquet(scores_path)
        print(f"加载 ALSTM 分数: {alstm_scores.shape[0]} 期 x {alstm_scores.shape[1]} 只")
    else:
        print("未找到 ALSTM 分数，使用等权全市场")

    test_env = build_env(
        close_df, feat_df, label_df,
        TEST[0], TEST[1],
        initial_amount=args.capital,
        scores=alstm_scores,
    )
    print(f"测试日数: {len(test_env.dates)}")

    obs, _ = test_env.reset()
    daily_returns = []
    exposures = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        daily_returns.append(info["daily_return"])
        exposures.append(info["exposure"])
        if terminated or truncated:
            break

    # 结果统计
    asset_curve = np.array(test_env.asset_memory)
    daily_ret = np.array(daily_returns)
    exp_arr = np.array(exposures)
    total_return = asset_curve[-1] / asset_curve[0] - 1
    n_years = len(daily_ret) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 1 else 0
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    peak = np.maximum.accumulate(asset_curve)
    drawdowns = (peak - asset_curve) / peak
    max_dd = np.max(drawdowns)

    print(f"\n{'='*46}")
    print(f"回测区间      : {TEST[0]} ~ {TEST[1]} ({len(daily_ret)} 交易日)")
    print(f"初始资金      : {args.capital:,.2f}")
    print(f"期末资金      : {asset_curve[-1]:,.2f}")
    print(f"总收益率      : {total_return:+.2%}")
    print(f"年化收益率    : {annual_return:+.2%}")
    print(f"最大回撤      : {max_dd:.2%}   "
          f"{'✓' if max_dd <= 0.2 else '✗'} "
          f"{'达标' if max_dd <= 0.2 else '超过上限 20%'}")
    print(f"年化波动率    : {annual_vol:.2%}")
    print(f"Sharpe        : {sharpe:.3f}")
    print(f"平均仓位      : {exp_arr.mean():.1%}")
    print(f"仓位标准差    : {exp_arr.std():.1%}")
    print(f"{'='*46}")

    # 保存
    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)

    equity = pd.DataFrame({
        "date": test_env.dates[:len(asset_curve)],
        "equity": asset_curve,
    })
    equity.to_csv(out / "ppo_equity.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "date": test_env.dates[:len(daily_ret)],
        "daily_return": daily_ret,
        "exposure": exp_arr,
    }).to_csv(out / "daily_returns.csv", index=False, encoding="utf-8-sig")

    # 权重存到 models/，与回测结果分开 —— models/ 走 LFS 版本化
    paths.ensure_dirs()
    model.save(str(paths.PPO_MODEL.with_suffix("")))
    print(f"\n模型已保存: {paths.PPO_MODEL}")
    print(f"回测明细: {out.resolve()}")
    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
