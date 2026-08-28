"""验证导出的 Qlib 数据可读。

⚠ **必须作为真实文件运行，不能用 ``python - <<EOF`` 从 stdin 喂进去。**
Qlib 内部用 joblib 并行，Windows 上 multiprocessing 是 spawn 模式，
子进程要重新 import ``__main__`` —— 从 stdin 运行时没有可导入的主模块，
表现为**永久挂起而不是报错**（实测卡满 10 分钟无输出）。

同理，所有逻辑必须放在 ``if __name__ == "__main__":`` 之下。

用法::

    python scripts/check_qlib.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    p = argparse.ArgumentParser(description="验证 Qlib 数据")
    p.add_argument("--uri", default=None, help="qlib 数据目录")
    p.add_argument("--index", default="csi300")
    args = p.parse_args()

    from qmtquant.config import get_config

    uri = args.uri or str(Path(get_config().data.store_dir) / "qlib_data")
    if not Path(uri).exists():
        print(f"数据目录不存在: {uri}\n请先运行 scripts/export_qlib.py")
        return 1

    import qlib
    from qlib.data import D

    # 单进程：Windows 上并行读会显著变慢甚至挂起，验证阶段不需要并行
    qlib.init(provider_uri=uri, region="cn", joblib_backend="threading")
    print(f"qlib {qlib.__version__}  provider_uri={uri}\n")

    cal = D.calendar(start_time="2016-01-01", end_time="2026-12-31")
    print(f"日历: {len(cal)} 个交易日  {cal[0].date()} ~ {cal[-1].date()}")

    print(f"\n成分股（{args.index}，point-in-time）:")
    for d in ("2017-06-30", "2021-06-30", "2026-07-31"):
        inst = D.list_instruments(D.instruments(args.index),
                                  start_time=d, end_time=d, as_list=True)
        print(f"  {d}: {len(inst):>4} 只   例 {sorted(inst)[:4]}")

    print("\n行情:")
    df = D.features(["SH600000", "SZ000001"],
                    ["$close", "$volume", "$factor"],
                    start_time="2024-01-02", end_time="2024-01-08")
    print(df.to_string())

    print("\n表达式引擎（未来 1 日收益，Ref 用负数看未来）:")
    r = D.features(["SH600000"], ["Ref($close, -1)/$close - 1"],
                   start_time="2024-01-02", end_time="2024-01-08")
    print(r.to_string())

    print("\n✓ Qlib 能正常读取本项目导出的数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
