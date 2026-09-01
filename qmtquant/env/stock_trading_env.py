"""A 股多股票交易 Gymnasium 环境。

State = [现金, 持仓数量×N, 收盘价×N, 因子×N×F]
Action = [-1, +1]^N（连续），正=买 负=卖，幅度按 hmax 缩放
Reward = 组合日收益率

设计要点：
- T+1 规则：当日买入的股票次日才能卖出
- 交易成本：买入 0.05%，卖出 0.15%（含印花税）
- 整手约束：A 股 100 股一手
- 涨跌停不可交易（简化：不处理，由数据端过滤）
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AShareTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim: int,
        feature_dim: int,
        initial_amount: float = 500_000,
        hmax: int = 100,
        buy_cost_pct: float = 0.0005,
        sell_cost_pct: float = 0.0015,
        reward_scaling: float = 1e-4,
        max_holdings: int = 30,
    ):
        super().__init__()

        self.df = df
        self.stock_dim = stock_dim
        self.feature_dim = feature_dim
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.max_holdings = max_holdings

        # 交易日列表
        self.dates = sorted(df.index.get_level_values("datetime").unique())
        self.day = 0
        self.terminal = False

        # State: 现金(1) + 持仓(N) + 价格(N) + 因子(N*F)
        self.state_dim = 1 + stock_dim + stock_dim + stock_dim * feature_dim
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(stock_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # T+1 记录：当日买入的不能卖
        self.bought_today = np.zeros(stock_dim, dtype=bool)

        self._init_state()

    def _init_state(self):
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float64)
        self.bought_today = np.zeros(self.stock_dim, dtype=bool)
        self.day = 0
        self.terminal = False
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []

    def _get_prices(self):
        date = self.dates[self.day]
        day_data = self.df.loc[date] if date in self.df.index.get_level_values("datetime") else None
        if day_data is None:
            return np.zeros(self.stock_dim), np.zeros((self.stock_dim, self.feature_dim))

        prices = np.zeros(self.stock_dim)
        features = np.zeros((self.stock_dim, self.feature_dim))

        if hasattr(day_data, "iloc"):
            for i in range(min(len(day_data), self.stock_dim)):
                row = day_data.iloc[i]
                prices[i] = row.get("close", 0)
                feat_cols = [c for c in row.index if c not in ("close", "open", "high", "low", "volume")]
                for j, col in enumerate(feat_cols[:self.feature_dim]):
                    features[i, j] = row[col] if np.isfinite(row[col]) else 0

        return prices, features

    def _get_state(self, prices, features):
        state = np.concatenate([
            [self.cash],
            self.holdings,
            prices,
            features.flatten(),
        ]).astype(np.float32)
        return state

    def _portfolio_value(self, prices):
        return self.cash + np.sum(self.holdings * prices)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        prices, features = self._get_prices()
        state = self._get_state(prices, features)
        return state, {}

    def step(self, action):
        prices, features = self._get_prices()
        begin_value = self._portfolio_value(prices)

        # T+1: 昨天买入的今天可以卖了
        self.bought_today[:] = False

        # 先卖后买
        sell_idx = np.where(action < 0)[0]
        buy_idx = np.where(action > 0)[0]

        # 卖出
        for i in sell_idx:
            if self.holdings[i] <= 0 or prices[i] <= 0:
                continue
            sell_shares = min(
                int(abs(action[i]) * self.hmax) // 100 * 100,
                int(self.holdings[i])
            )
            if sell_shares <= 0:
                continue
            sell_amount = sell_shares * prices[i]
            cost = sell_amount * self.sell_cost_pct
            self.cash += sell_amount - cost
            self.holdings[i] -= sell_shares

        # 买入
        for i in buy_idx:
            if prices[i] <= 0:
                continue
            buy_shares = int(action[i] * self.hmax) // 100 * 100
            if buy_shares <= 0:
                continue
            buy_amount = buy_shares * prices[i]
            cost = buy_amount * self.buy_cost_pct
            if buy_amount + cost > self.cash:
                buy_shares = int(self.cash / (prices[i] * (1 + self.buy_cost_pct))) // 100 * 100
                if buy_shares <= 0:
                    continue
                buy_amount = buy_shares * prices[i]
                cost = buy_amount * self.buy_cost_pct
            self.cash -= buy_amount + cost
            self.holdings[i] += buy_shares
            self.bought_today[i] = True

        # 前进一天
        self.day += 1
        if self.day >= len(self.dates):
            self.terminal = True
            prices_new = prices
        else:
            prices_new, features = self._get_prices()

        end_value = self._portfolio_value(prices_new)
        reward = (end_value - begin_value) * self.reward_scaling
        self.asset_memory.append(end_value)
        self.rewards_memory.append(reward)

        state = self._get_state(prices_new, features) if not self.terminal else np.zeros(self.state_dim, dtype=np.float32)

        return state, reward, self.terminal, False, {}
