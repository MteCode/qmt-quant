"""搬家清单对账 —— 搬之前生成，搬之后核对。

## 为什么需要

项目 18 GB、15 万个文件。拷贝这种量级的小文件，无论走 U 盘、
移动硬盘还是网络共享，**静默丢文件是常态**：路径超长、文件名含特殊字符、
中途断开、目标盘 FAT32 单文件 4 GB 上限，都会让个别文件悄悄没过去。

丢了不会立刻报错。丢一个 parquet，要跑到某次回测才发现某只股票没数据；
丢了 config.yaml，要跑到取数才报「未配置 token」。

## 对账策略

15 万个文件全算哈希太慢（要十几分钟）。分两档：

**关键文件逐个哈希** —— 配置、代码、模型产物。这些丢一个就出事，
数量少（几千个），算得起。

**数据文件只对数量和大小** —— data/ 下十几万个 parquet，
逐个哈希不划算。文件数 + 总字节数能抓住绝大多数拷贝事故：
少文件必然数量对不上，截断必然字节数对不上。

用法::

    # 旧机器（搬之前）
    python scripts/migration_manifest.py --save

    # 新机器（搬之后）
    python scripts/migration_manifest.py --verify
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "migration_manifest.json"

#: 逐个算哈希的目录。丢一个就出事，且数量可控
CRITICAL = [
    "config",
    "qmtquant",
    "scripts",
    "strategies",
    "webui",
    "tests",
]

#: 只对数量与大小的目录。十几万个文件，逐个哈希不划算
BULK = ["data"]

#: 不参与对账 —— 本机运行痕迹或可重算的东西
SKIP_PARTS = {
    "__pycache__", ".git", ".venv", "logs", "mlruns",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
}


def iter_files(base: Path):
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if SKIP_PARTS & set(p.parts):
            continue
        yield p


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build() -> dict:
    """生成清单。"""
    critical, bulk = {}, {}
    t0 = time.time()

    for name in CRITICAL:
        base = ROOT / name
        if not base.exists():
            continue
        n = 0
        for p in iter_files(base):
            rel = p.relative_to(ROOT).as_posix()
            sz = p.stat().st_size
            # 大文件（模型权重、分数面板）只记大小 —— 算哈希太慢
            critical[rel] = {"size": sz,
                             "sha256": sha256(p) if sz < 50 << 20 else None}
            n += 1
        print(f"  {name:<12s} {n:>6,} 个文件  ({time.time() - t0:.0f}s)")

    for name in BULK:
        base = ROOT / name
        if not base.exists():
            continue
        n = total = 0
        by_dir: dict[str, list] = {}
        for p in iter_files(base):
            n += 1
            total += p.stat().st_size
            # 按二级目录汇总，定位丢失范围
            parts = p.relative_to(ROOT).parts
            key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
            d = by_dir.setdefault(key, [0, 0])
            d[0] += 1
            d[1] += p.stat().st_size
        bulk[name] = {"files": n, "bytes": total,
                      "by_dir": {k: {"files": v[0], "bytes": v[1]}
                                 for k, v in sorted(by_dir.items())}}
        print(f"  {name:<12s} {n:>6,} 个文件  "
              f"{total / 1024**3:.1f} GB  ({time.time() - t0:.0f}s)")

    return {"root": str(ROOT), "critical": critical, "bulk": bulk,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def verify(old: dict) -> int:
    """与清单核对。返回问题数。"""
    bad = 0
    print("\n" + "-" * 62)
    print("关键文件（逐个哈希）")
    print("-" * 62)

    missing, changed = [], []
    for rel, meta in old["critical"].items():
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        if p.stat().st_size != meta["size"]:
            changed.append((rel, "大小不同"))
            continue
        if meta["sha256"] and sha256(p) != meta["sha256"]:
            changed.append((rel, "内容不同"))

    total = len(old["critical"])
    if not missing and not changed:
        print(f"[ OK ] {total:,} 个文件全部一致")
    else:
        bad += len(missing) + len(changed)
        if missing:
            print(f"[缺失] {len(missing)} 个文件没拷过来：")
            for r in missing[:15]:
                print(f"       {r}")
            if len(missing) > 15:
                print(f"       ... 还有 {len(missing) - 15} 个")
        if changed:
            print(f"[损坏] {len(changed)} 个文件不一致：")
            for r, why in changed[:15]:
                print(f"       {r}  ({why})")

    print("\n" + "-" * 62)
    print("数据文件（对数量与大小）")
    print("-" * 62)

    for name, meta in old["bulk"].items():
        base = ROOT / name
        if not base.exists():
            print(f"[缺失] {name}/ 整个目录不存在")
            bad += 1
            continue
        n = total_b = 0
        by_dir: dict[str, list] = {}
        for p in iter_files(base):
            n += 1
            sz = p.stat().st_size
            total_b += sz
            parts = p.relative_to(ROOT).parts
            key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
            d = by_dir.setdefault(key, [0, 0])
            d[0] += 1
            d[1] += sz

        dn, db = n - meta["files"], total_b - meta["bytes"]
        if dn == 0 and db == 0:
            print(f"[ OK ] {name}/  {n:,} 个文件  {total_b / 1024**3:.1f} GB")
            continue

        bad += 1
        print(f"[不符] {name}/  文件数 {meta['files']:,} -> {n:,} ({dn:+,})"
              f"  字节 {db:+,}")
        # 定位到具体子目录
        for k, exp in meta["by_dir"].items():
            got = by_dir.get(k, {"files": 0, "bytes": 0})
            if isinstance(got, list):
                got = {"files": got[0], "bytes": got[1]}
            if got["files"] != exp["files"] or got["bytes"] != exp["bytes"]:
                print(f"       {k:<28s} {exp['files']:>7,} -> "
                      f"{got['files']:>7,} 个")

    return bad


def main() -> int:
    p = argparse.ArgumentParser(description="搬家清单对账")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true", help="旧机器：生成清单")
    g.add_argument("--verify", action="store_true", help="新机器：核对")
    args = p.parse_args()

    print("=" * 62)
    print("搬家清单对账")
    print(f"项目: {ROOT}")
    print("=" * 62)

    if args.save:
        print("\n扫描中...")
        m = build()
        MANIFEST.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        n = len(m["critical"]) + sum(v["files"] for v in m["bulk"].values())
        print(f"\n清单已写入 {MANIFEST.name}（{n:,} 个文件）")
        print("\n这个文件**必须跟着一起拷过去**，新机器上跑：")
        print("  python scripts/migration_manifest.py --verify")
        return 0

    if not MANIFEST.exists():
        print(f"\n找不到 {MANIFEST.name} —— 它应该跟项目一起拷过来")
        print("若已丢失，只能回旧机器重新 --save")
        return 1

    old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"\n清单生成于 {old.get('generated_at')}")
    print(f"原路径 {old.get('root')}")
    bad = verify(old)

    print("\n" + "=" * 62)
    if bad:
        print(f"发现 {bad} 处不一致 —— 回旧机器重拷对应文件")
        return 1
    print("拷贝完整。下一步跑环境自检：")
    print("  python scripts/check_migration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
