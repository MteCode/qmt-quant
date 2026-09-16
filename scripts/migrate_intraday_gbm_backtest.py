"""把 intraday_gbm 的历史回测产物迁成标准 run。

旧产物躺在 models/intraday_gbm/backtest/，是固定覆写的「最近一次」，
没有 manifest 也没有参数快照。策略页为了读它，得在 strategies.py 里
专门写一个 _load_intraday_gbm —— 12 个策略就是 12 个这样的函数。

这个脚本把那份产物补成带 manifest 的 run，之后通用 reader 就能读它，
专用 loader 可以删掉。

一次性脚本：run 已存在时直接跳过，可重复执行。

用法::

    python scripts/migrate_intraday_gbm_backtest.py
    python scripts/migrate_intraday_gbm_backtest.py --dry-run
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LEGACY_DIR = ROOT / "models" / "intraday_gbm" / "backtest"
RUNS_DIR = ROOT / "strategies" / "intraday_gbm" / "runs"

_CORE = ("total_return", "annual_return", "monthly_return",
         "max_drawdown", "sharpe", "volatility",
         "n_trades", "win_rate", "trading_days")


def main() -> int:
    p = argparse.ArgumentParser(description="迁移历史回测产物为标准 run")
    p.add_argument("--dry-run", action="store_true", help="只看不写")
    p.add_argument("--force", action="store_true",
                   help="run 已存在时重新生成（指标口径变更后用）")
    args = p.parse_args()

    summary_path = LEGACY_DIR / "summary.json"
    if not summary_path.exists():
        print(f"[ERROR] 找不到历史产物: {summary_path}")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary.get("results") or {}
    if not results:
        print("[ERROR] summary.json 里没有 results")
        return 1

    from webui.model_registry import (create_manifest, discover_backtests,
                                      save_manifest)

    # run_id 用产物自己的生成时间，重跑本脚本不会产生第二个 run
    gen = str(summary.get("generated_at") or "")
    stamp = gen.replace("-", "").replace(":", "").replace(" ", "_")[:15] \
        or "unknown"
    run_id = f"bt_legacy_{stamp}"

    if any(r.get("run_id") == run_id for r in discover_backtests()):
        if not args.force:
            print(f"run 已存在，跳过: {run_id}（要重建加 --force）")
            return 0
        print(f"run 已存在，--force 重建: {run_id}")

    best_mode = max(results, key=lambda m: results[m].get("sharpe") or 0)
    best = results[best_mode]
    metrics = {k: best.get(k) for k in _CORE}
    metrics["best_mode"] = best_mode
    metrics["modes"] = sorted(results)
    metrics["by_mode"] = {}
    for mode, r in results.items():
        tg = r.get("targets") or {}
        row = {k: r.get(k) for k in _CORE}
        row["targets_passed"] = sum(1 for v in tg.values() if v.get("pass"))
        row["targets_total"] = len(tg)
        metrics["by_mode"][mode] = row

    artifacts = {"summary_json": "summary.json"}
    copies = [(summary_path, "summary.json")]
    for mode in results:
        for suffix, key in (("daily", "daily_csv"), ("trades", "trades_csv")):
            src = LEGACY_DIR / f"{mode}_{suffix}.csv"
            if src.exists():
                copies.append((src, src.name))
                artifacts[f"{mode}_{key}"] = src.name

    print(f"run_id   : {run_id}")
    print(f"模式     : {', '.join(sorted(results))}（主 {best_mode}）")
    print(f"顶层指标 : 总收益 {metrics['total_return']:.2%}  "
          f"夏普 {metrics['sharpe']:.2f}  交易 {metrics['n_trades']}")
    print(f"拷贝文件 : {len(copies)} 个")

    if args.dry_run:
        print("\n-- dry-run，未写入 --")
        return 0

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for src, name in copies:
        shutil.copy2(src, run_dir / name)

    manifest = create_manifest(
        run_id=run_id,
        kind="backtest",
        model_type="IntradayGBM",
        strategy_id="intraday_gbm",
        params={**(summary.get("config") or {}),
                "modes": sorted(results),
                **{f"model_{k}": v
                   for k, v in (summary.get("model") or {}).items()}},
        metrics=metrics,
        artifacts=artifacts,
        notes=f"由 {LEGACY_DIR.relative_to(ROOT)} 迁移，原生成于 {gen}",
    )
    # 迁移产物保留原始生成时间，否则历史 run 会排到今天新跑的前面
    if gen:
        manifest["created_at"] = gen.replace(" ", "T")
    save_manifest(run_dir, manifest)

    print(f"\n已写入: {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
