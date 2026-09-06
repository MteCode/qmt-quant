# 换机器迁移清单

> 2026-09-06 实际执行过一次：`E:\qmt`（用户 DELL）→ `D:\qmt\qmt`（用户 51453）。
> 下面是照着做出来的结论，不是推测。原先那版文档的核心判断是错的，见「被证伪的两条」。

## 被证伪的两条

原文档说「项目必须仍放在 `E:\qmt`，盘符和目录名都不能变」，以及「新机器用户名
不是 `DELL` 就走不通」。**两条都不成立。**

Windows 的 venv 确实把绝对路径烧进了文件，但烧的位置只有两处，都能修：

| 烧在哪 | 影响 | 修法 |
|---|---|---|
| `.venv/pyvenv.cfg` 的 `home` / `executable` | `python.exe` 报 `No Python at ...` | `python -m venv --upgrade .venv` |
| `.venv/Scripts/*.exe` 里的 shebang（67 个） | `pip.exe` / `pytest.exe` 静默退出 1 | `scripts/fix_venv_launchers.py --apply` |

`site-packages` 本身**完全没有绝对路径**，486 个包一个都不用重装 —— 包括
`torch 2.11.0+cu128`、`pyqlib`、`lightgbm`，以及不能 pip 装的 `xtquant`。
换句话说，5.5 GB 的 `.venv` 是可以整个搬走的，只是要修上面两处。

`activate` / `activate.bat` / `Activate.ps1` 由 `--upgrade` 一并重写，不用管。

---

## 目录外的依赖：只有两处

| 依赖 | 拷目录能带走 | 说明 |
|---|---|---|
| Python 3.11 x64 | 否 | 必须自己装，见下 |
| QMT 客户端 | 否 | 必须自己装并**以极简模式登录一次** |

VC++ 运行库、CUDA、cuDNN 都**不用单独装**：CUDA 全套在 `torch/lib` 里自带，
VC++ 运行库 Win11 通常已有（LightGBM 的 `vcomp140.dll` 是唯一没被 wheel 兜住的
原生依赖，靠系统运行库）。webui 是 Flask 服务端渲染，**不需要 Node/npm**。

---

## 方案 A：原样搬（18 GB，15.4 万个文件）—— 采用这个

### 第 1 步：旧机器上生成对账清单

```bash
.venv/Scripts/python.exe scripts/migration_manifest.py --save
```

18 GB、15 万个小文件，**静默丢文件是常态** —— 路径超长、文件名含特殊字符、
中途断开、目标盘 FAT32 的 4 GB 单文件上限，都会让个别文件悄悄没过去，
而且不会报错。丢一个 parquet 要跑到某次回测才发现某只股票没数据。

清单落在项目根目录，跟着一起拷过去。它存的是**相对路径**，换目录后照样能用。

### 第 2 步：拷贝

别用拖拽。用 `robocopy`，它有重试和完整的失败报告（源盘符换成实际的）：

```
robocopy X:\qmt D:\qmt\qmt /E /R:2 /W:2 /MT:8 /NP /LOG:D:\qmt_copy.log
```

`/E` 含空目录，`/R:2` 失败重试 2 次，`/MT:8` 八线程。
跑完看日志末尾的 `Failed` 一栏，**必须是 0**。

放哪个目录都行，路径不必和旧机器一致。

### 第 3 步：装 Python 3.11 x64

装哪都行，不需要和旧机器同路径、同用户名，补丁号也不必一致
（实测 3.11.0 建的 venv 复用 3.11.9 的基础解释器没问题，`python311.dll` 同 ABI）。

```
winget install --id Python.Python.3.11 -e --source winget --scope user
```

> `--source winget` 不能省。不加的话 winget 会去问 msstore 源，
> 那边证书校验失败会让整条命令以 `0x8a15005e` 退出，看着像包不存在。

**必须是 64 位** —— miniQMT 是 64 位，32 位 Python 会在 `import xtquant` 时报 DLL 错误。

### 第 4 步：修 venv

```bash
"$LOCALAPPDATA/Programs/Python/Python311/python.exe" -m venv --upgrade D:\qmt\qmt\.venv
```

> **不要手改 `pyvenv.cfg` 了事。** 只改配置文件的话，`Scripts/` 下缺 venv 需要的
> 那几个 DLL，报错是 `api-ms-win-crt-heap-l1-1-0.dll` 找不到 —— 完全看不出跟路径有关。
> `--upgrade` 会一并把 `python.exe` 和 activate 脚本换成新解释器的版本，
> 且**不动 `site-packages`**。

再修那 67 个控制台启动器：

```bash
.venv/Scripts/python.exe scripts/fix_venv_launchers.py --apply
```

不修也能跑（`python -m pip`、`python -m pytest` 不走启动器），但 `pip.exe`、
`pytest.exe`、`qrun.exe`、`mlflow.exe` 会**静默退出 1，一个字都不打**，
排障时极具误导性。

### 第 5 步：装 QMT 客户端，并以极简模式登录一次

装完确认数据目录，若不是 `D:/qmtApp/userdata_mini`，改 `config/config.yaml` 的
`qmt_path`（填 `userdata_mini`，不是安装根目录，用正斜杠）。

**`userdata_mini` 是极简模式首次登录成功后才创建的。** 只装客户端、或者只登大
QMT，这个目录不会出现，`check_migration.py` 会报「QMT 数据目录不存在」。

登录框里勾「独立交易 / 启动极简模式」再登录。验证：

```
Get-Process XtMiniQmt,miniquote
Test-Path D:\qmtApp\userdata_mini
(Test-NetConnection 127.0.0.1 -Port 58610).TcpTestSucceeded
```

三个都要通。**58610 只有极简模式监听**（大 QMT 开的是 58600，那是给内置公式用的，
不是 xtquant 的口），`xtdata` 的地址写死在 `xtquant/xtdata.py` 里。
所以不是「交易要极简、行情不用」—— **行情和交易都要**。

> 回测不受影响：`XtDataFeed.load_bars` 只读本地 Parquet，不开客户端照样跑。
> 受阻的是数据回源下载、财报下载、以及实盘/模拟盘取价。

### 第 6 步：验证（四步都要跑）

```bash
.venv/Scripts/python.exe scripts/migration_manifest.py --verify
```

对账文件完整性。关键文件逐个哈希，`data/` 对数量与字节数，不符时定位到具体子目录。

```bash
.venv/Scripts/python.exe scripts/check_migration.py
```

查配置、数据目录、行情新鲜度、实盘状态。

```bash
.venv/Scripts/python.exe scripts/check_env.py
```

查 Python 版本位数、依赖、xtquant 连通性、配置安全。

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

2026-09-06 实测 616 passed。

### 关于显卡

装的是 CUDA 版 torch（`2.11.0+cu128`）。新机器没有 N 卡也**不用改**：
torch 会正常导入，`cuda.is_available()` 返回 False，自动退回 CPU。
只影响 ALSTM 训练速度，树模型（当前主力）、回测、实盘都不受影响。

CUDA runtime 和 cuDNN 都在 wheel 里自带，机器上**不需要装 CUDA Toolkit**。

---

## 方案 B：只搬数据，重建环境（12.5 GB）

方案 A 已证明可行，这条基本用不上了。真要重建：

**必须拷的（GitHub 上没有）**

| 目录 | 体积 | 内容 |
|---|---|---|
| `config/config.yaml` | 2 KB | **Tushare token + 资金账号**，gitignored |
| `data/` | 9.6 GB | 行情、财报、清洗层、`market.db`、`qlib_data` |
| `strategies/**/models/` | 1.7 GB | 模型权重、分数面板、候选池 |
| `data/state.db` | 44 KB | 运行时状态 |
| `strategies/**/state/` | 小 | 实盘净值曲线、回撤峰值、持仓快照 |

后两项是**本机实盘记录，没有任何办法重建**。净值曲线漏一天补不回来 ——
券商查不到历史序列。回撤峰值删掉等于抹掉回撤记忆，风控会从零重新记
（见 `qmtquant/risk/drawdown.py`）。

> `data/risk_state.json` 已无任何代码读写，是历史遗留。风控状态实际存在
> `strategies/<策略>/state/risk_state.json`，路径由各策略的 `paths.py` 定死。

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

`requirements.txt` 里 **xtquant 不是 pip 包**，要从 miniQMT 安装目录复制到
`site-packages`（当前版本 250807.1.2）。注意有些客户端安装目录下没有
`bin.x64\Lib\site-packages` —— 这时只能从旧机器的 `.venv` 里拷。

`torch` 装的是 CUDA 版（2.11.0+cu128）。没有 N 卡就装 CPU 版。

然后把上面「必须拷的」五项覆盖回去。

---

## 别漏的东西

- **`config/config.yaml` 不在 GitHub 上**（含 token 和资金账号，故意 gitignore）。
  只 clone 不拷这个文件，所有需要联网取数的脚本都会失败。
- **`data/` 也不在 GitHub 上**。重下一遍全市场行情要几小时，财报 22 分钟。
- **`qlib_data` 会落后于 `data/1d`**。搬完先看 `check_migration.py` 的新鲜度一栏：
  2026-09-06 实测原始 1d 到 `20260904`，而 qlib 日历只到 `2026-08-31`，差 4 个交易日。
  跑任何基于 qlib 的回测或出信号前，先 `python scripts/export_qlib.py` 重导。
- **对账清单看不见空目录**。`strategies/*/state/` 如果旧机器上有内容而拷丢了，
  `--verify` 不会报 —— 靠 `check_migration.py` 逐策略列出来确认。
- 换机器后 miniQMT 要重新登录，实盘下单前先用模拟盘验证一遍链路。
- 盘前/盘后行情更新**没有开机自启**，只在 webui 进程活着时才走
  （`webui/scheduler.py` 的后台线程）。新机器上要么常驻 webui，要么手工跑。
