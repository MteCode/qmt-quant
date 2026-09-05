# 换机器迁移清单

## 结论：拷贝 `E:\qmt` 不够，还有三处依赖在目录外

| 依赖 | 位置 | 拷目录能带走 |
|---|---|---|
| 系统 Python 3.11.0 | `C:\Users\DELL\AppData\Local\Programs\Python\Python311` | 否 |
| miniQMT 客户端 | `D:/qmtApp/userdata_mini` | 否 |
| 项目路径本身 | 必须仍放在 `E:\qmt` | 见下 |

`.venv` 不是自包含的。`pyvenv.cfg` 里写死了

    home = C:\Users\DELL\AppData\Local\Programs\Python\Python311
    command = ... -m venv E:\qmt\.venv

Windows 的 venv 会把绝对路径烧进 `Scripts/*.exe`。**换盘符或换目录名，
`.venv` 直接失效**，报错通常是 `ModuleNotFoundError` 或找不到解释器 ——
看着像装漏了包，实际是路径不对。

---

## 方案 A：原样搬（最省事，18 GB）

新机器先装好两样，再拷目录：

1. **Python 3.11.0**，装到同一路径
   `C:\Users\DELL\AppData\Local\Programs\Python\Python311`
   （用户名不同就装到新机器的对应位置，然后走方案 B 重建 venv）
2. **miniQMT 客户端**。装完确认数据目录路径，若不是 `D:/qmtApp/userdata_mini`，
   改 `config/config.yaml` 的 `qmt_path`

然后整个 `E:\qmt` 拷到新机器的 **`E:\qmt`**（盘符和目录名都不能变）。

验证：

```bash
cd /e/qmt && .venv/Scripts/python.exe -c "import qmtquant, xtquant, qlib, torch; print('OK')"
```

---

## 方案 B：只搬数据，重建环境（12.5 GB，更稳）

新机器路径可以随便放。

**必须拷的（GitHub 上没有）**

| 目录 | 体积 | 内容 |
|---|---|---|
| `config/config.yaml` | 2 KB | **Tushare token + 资金账号**，gitignored |
| `data/` | 9.6 GB | 行情、财报、清洗层、`market.db`、`qlib_data` |
| `strategies/**/models/` | 1.7 GB | 模型权重、分数面板、候选池 |
| `data/risk_state.json` | 237 B | 回撤峰值，跨重启累积 |
| `data/state.db` | 44 KB | 运行时状态 |
| `strategies/**/state/` | 小 | 实盘净值曲线、持仓快照 |

后三项是**本机实盘记录，没有任何办法重建**。删掉 `risk_state.json`
等于抹掉回撤记忆，风控会从零重新记峰值（见 `qmtquant/risk/drawdown.py`）。
净值曲线漏一天补不回来 —— 券商查不到历史序列。

前两项在 `data/` 里，整个目录拷过去就带上了；
`strategies/**/state/` 要单独确认。

**不用拷的**

- `.venv/`（5.5 GB）—— 重建
- `.git/`（66 MB）—— 从 GitHub clone
- `logs/`（132 MB）—— 运行日志
- `strategies/**/models/features_*.pkl`（1.6 GB）—— Alpha158 特征缓存，
  20 分钟可重算，已 gitignore

重建步骤：

```bash
git clone git@github.com:MteCode/qmt-quant.git qmt
```

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements.txt` 里 **xtquant 不是 pip 包**，要从 miniQMT 安装目录
复制到 `site-packages`（当前版本 250807.1.2）。

`torch` 装的是 CUDA 版（2.11.0+cu128）。新机器没有 N 卡就装 CPU 版，
只影响 ALSTM 训练速度，树模型和回测不受影响。

然后把上面「必须拷的」四项覆盖回去。

---

## 迁移后自检

```bash
cd /e/qmt && .venv/Scripts/python.exe scripts/check_migration.py
```

本机实测的正常输出：

    Tushare token : 已配置
    QMT 路径      : D:/qmtApp/userdata_mini 存在
    1d / clean / qlib_data / financial / factor/daily_basic  全部有
    market.db     : 有

再确认数据日期没断（这一步会连 miniQMT，要先启动客户端）：

```bash
cd /e/qmt && .venv/Scripts/python.exe scripts/update_market_data.py --check-only
```

---

## 别漏的东西

- **`config/config.yaml` 不在 GitHub 上**（含 token 和资金账号，故意 gitignore）。
  只 clone 不拷这个文件，所有需要联网取数的脚本都会失败。
- **`data/` 也不在 GitHub 上**。重下一遍全市场行情要几小时，财报 22 分钟。
- 换机器后 miniQMT 要重新登录，实盘下单前先用模拟盘验证一遍链路。
