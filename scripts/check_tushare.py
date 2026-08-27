"""检查 Tushare token 与积分是否可用。

在跑长时间下载之前先跑这个 —— 积分未到账时 ``index_weight`` 会
**静默返回空 DataFrame** 而不是报错，直接开跑会白等十分钟才发现。

用法::

    python scripts/check_tushare.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import get_config  # noqa: E402
from qmtquant.datafeed.tushare_feed import (  # noqa: E402
    TushareClient,
    TushareError,
)


def main() -> int:
    cfg = get_config()
    print("=== Tushare Pro 连通性检查 ===\n")

    try:
        client = TushareClient(cfg.tushare)
    except TushareError as e:
        print(f"[×] {e}")
        return 1

    # 只显示头尾，避免 token 出现在终端记录或截图里
    tok = client._token
    print(f"token          : {tok[:6]}...{tok[-4:]}（长度 {len(tok)}）")
    print(f"限流           : {cfg.tushare.calls_per_minute} 次/分钟")
    print("\n正在探活...\n")

    info = client.check()

    if info["token_ok"]:
        print(f"[√] token 有效，全市场 {info.get('stock_count', 0):,} 只在市股票")
    else:
        print("[×] token 无效或基础接口不可用")

    if info["points_2000_ok"]:
        print("[√] 2000 积分接口可用（index_weight 返回正常）")
    else:
        print("[×] 2000 积分接口不可用 —— 历史成分股拉不了")

    for n in info["notes"]:
        print(f"    · {n}")

    if info["token_ok"] and info["points_2000_ok"]:
        print("\n一切就绪。下一步：")
        print("  python scripts/download_index_weight.py --index 000300.SH")
        return 0

    print("\n排查：")
    print("  · token 在 https://tushare.pro/user/token")
    print("  · 积分在 https://tushare.pro/user/info 查看，充值后可能需几分钟到账")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
