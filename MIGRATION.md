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

## 方案 A：原样搬（18 GB，15.4 万个文件）—— 采用这个

### 第 1 步：旧机器上生成对账清单

```bash
cd /e/qmt && .venv/Scripts/python.exe scripts/migration_manifest.py --save
```

18 GB、15 万个小文件，**静默丢文件是常态** —— 路径超长、文件名含特殊字符、
中途断开、目标盘 FAT32 的 4 GB 单文件上限，都会让个别文件悄悄没过去，
而且不会报错。丢一个 parquet 要跑到某次回测才发现某只股票没数据。

清单会落在项目根目录，跟着一起拷过去。

### 第 2 步：新机器先装两样，再拷

**① Python 3.11.0，装到完全相同的路径**

    C:\Users\DELL\AppData\Local\Programs\Python\Python311

⚠️ **新机器的 Windows 用户名不是 `DELL` 的话，这条路走不通** ——
`.venv/pyvenv.cfg` 里写死了上面这个路径。两个办法：

- 新机器建一个同名用户 `DELL`（最省事）
- 或者装完 Python 后改 `.venv/pyvenv.cfg` 的两行：

      home = C:\Users\<新用户名>\AppData\Local\Programs\Python\Python311
      executable = C:\Users\<新用户名>\...\Python311\python.exe

  Windows 的 venv 靠 `pyvenv.cfg` 找基础解释器，改完通常就能用。
  改完必须验证（见第 4 步），不行就走方案 B 重建。

**② miniQMT 客户端**。装完确认数据目录，若不是 `D:/qmtApp/userdata_mini`，
改 `config/config.yaml` 里的 `qmt_path`（填 `userdata_mini`，不是安装根目录）。

**③ 项目必须放在 `E:\qmt`**。盘符和目录名都不能变 ——
Windows 的 venv 把绝对路径烧进了 `Scripts/*.exe`。

### 第 3 步：拷贝

别用拖拽。用 `robocopy`，它有重试和完整的失败报告
（在**新机器**的 CMD 或 PowerShell 里跑，源换成实际的移动硬盘盘符）：

```
robocopy X:\qmt E:\qmt /E /R:2 /W:2 /MT:8 /NP /LOG:E:\qmt_copy.log
```

`/E` 含空目录，`/R:2` 失败重试 2 次，`/MT:8` 八线程。
跑完看日志末尾的 `Failed` 一栏，**必须是 0**。

### 第 4 步：验证（两步都要跑）

```bash
cd /e/qmt && .venv/Scripts/python.exe scripts/migration_manifest.py --verify
```

对账文件完整性。关键文件逐个哈希，data/ 对数量与字节数，
不符时会定位到具体子目录。

```bash
cd /e/qmt && .venv/Scripts/python.exe scripts/check_migration.py
```

查配置、数据目录、行情新鲜度、实盘状态。本机实测的正常输出见下方。

最后确认包能导入：

```bash
cd /e/qmt && .venv/Scripts/python.exe -c "import qmtquant, xtquant, qlib, torch, lightgbm; print('OK')"
```

### 关于显卡

当前装的是 CUDA 版 torch（`2.11.0+cu128`，本机 RTX 3050）。
新机器没有 N 卡也**不用改**：torch 会正常导入，`cuda.is_available()`
返回 False，自动退回 CPU。只影响 ALSTM 训练速度，
树模型（当前主力）、回测、实盘都不受影响。

---

## 方案 B：只搬数据，重建环境（12.5 GB）

**只在方案 A 的 venv 起不来时才走这条**（比如新机器用户名不同且改
`pyvenv.cfg` 无效）。路径可以随便放。

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
