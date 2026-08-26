"""环境自检。

上线前先跑这个，把 xtquant 接入的坑一次性暴露出来。

用法：
    python scripts/check_env.py
"""
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def check_python() -> bool:
    ver = sys.version_info
    bits = platform.architecture()[0]
    ok = ver >= (3, 9) and bits == "64bit"
    tag = OK if ok else FAIL
    print(f"{tag} Python {ver.major}.{ver.minor}.{ver.micro} {bits}")
    if bits != "64bit":
        print("       miniQMT 是 64 位，Python 必须也是 64 位，否则 import xtquant 会报 DLL 错误")
    return ok


def check_deps() -> bool:
    all_ok = True
    for mod in ["pandas", "numpy", "yaml", "pyarrow"]:
        try:
            __import__(mod)
            print(f"{OK} {mod}")
        except ImportError:
            print(f"{FAIL} {mod} 未安装 —— pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_xtquant() -> bool:
    try:
        from xtquant import xtdata
    except ImportError:
        print(f"{WARN} xtquant 未安装 —— 无法接入实盘，但可用 --mock 跑回测")
        print("       从 <QMT安装目录>\\bin.x64\\Lib\\site-packages\\xtquant 复制到本环境")
        return False

    print(f"{OK} xtquant: {xtdata.__file__}")
    try:
        # 能取到交易日历说明客户端连接正常
        dates = xtdata.get_trading_dates("SH", "20240101", "20240110")
        print(f"{OK} 客户端连接正常，取到 {len(dates)} 个交易日")
        return True
    except Exception as e:
        print(f"{WARN} xtquant 已导入但取数失败: {e}")
        print("       通常是 QMT 客户端未启动或未登录")
        return False


def check_config() -> bool:
    from qmtquant.config import CONFIG_DIR, get_config

    real = CONFIG_DIR / "config.yaml"
    if not real.exists():
        print(f"{WARN} config/config.yaml 不存在，当前使用 config.example.yaml 的默认值")
        print("       实盘前请复制模板并填入真实账号")
    else:
        print(f"{OK} config/config.yaml 已存在")

    cfg = get_config()
    print(f"       网关={cfg.gateway.name} 数据源={cfg.data.provider} "
          f"复权={cfg.data.dividend_type}")
    print(f"       风控: 单笔上限={cfg.risk.max_order_value:,.0f} "
          f"单票占比={cfg.risk.max_position_ratio:.0%} "
          f"日亏线={cfg.risk.daily_loss_limit_ratio:.1%}")
    return True


def check_gitignore() -> bool:
    """确认真实配置不会被误提交 —— 这是最容易出事的地方"""
    from qmtquant.config import CONFIG_DIR, ROOT_DIR

    if not (CONFIG_DIR / "config.yaml").exists():
        return True
    gi = ROOT_DIR / ".gitignore"
    if gi.exists() and "config/config.yaml" in gi.read_text(encoding="utf-8"):
        print(f"{OK} config.yaml 已被 .gitignore 排除，不会误提交账号信息")
        return True
    print(f"{FAIL} config.yaml 未被 .gitignore 排除，有泄露资金账号的风险！")
    return False


def main() -> int:
    print("=" * 52)
    print("qmtquant 环境自检")
    print("=" * 52)

    results = []
    for title, fn in [
        ("Python 环境", check_python),
        ("依赖包", check_deps),
        ("xtquant / QMT 客户端", check_xtquant),
        ("配置", check_config),
        ("配置安全", check_gitignore),
    ]:
        print(f"\n--- {title} ---")
        results.append(fn())

    print("\n" + "=" * 52)
    # xtquant 缺失不算致命（可离线回测），Python/依赖/安全必须过
    critical = [results[0], results[1], results[4]]
    if all(critical):
        print("核心检查通过。" + ("" if results[2] else "（xtquant 未就绪，仅可离线回测）"))
        return 0
    print("存在致命问题，请按上面提示处理后重试。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
