"""搬家后修复 .venv/Scripts/*.exe 里烧死的解释器路径。

Windows 上 pip/setuptools 生成的控制台脚本是这个结构::

    [launcher.exe 桩][#!<解释器绝对路径>\r\n][__main__.py 的 zip]

路径是**写死**的。项目从 E:\qmt 搬到别处后，pip.exe / pytest.exe /
jupyter.exe 全部静默失败（退出码 1，一个字都不打），看着像装漏了包。

`python -m pip`、`python -m pytest` 不受影响 —— 它们不走这些启动器。
所以这个脚本是可选的便利修复，不是能不能跑的前提。

用法::

    .venv/Scripts/python.exe scripts/fix_venv_launchers.py          # 只看不改
    .venv/Scripts/python.exe scripts/fix_venv_launchers.py --apply
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / ".venv" / "Scripts"
TARGET = (SCRIPTS / "python.exe").resolve()

ZIP_MAGIC = b"PK\x03\x04"


def repair(path: Path, apply: bool) -> str | None:
    """返回一句话说明，None 表示这个文件不用动。"""
    blob = path.read_bytes()
    zip_at = blob.find(ZIP_MAGIC)
    if zip_at < 0:
        return None  # python.exe / pythonw.exe 之类，不是控制台脚本

    # shebang 紧贴在 zip 前面，从 zip 往回找最后一个 b"#!"
    sb_at = blob.rfind(b"#!", 0, zip_at)
    if sb_at < 0:
        return None
    old = blob[sb_at:zip_at]
    new = b"#!" + str(TARGET).encode("utf-8") + b"\r\n"
    if old == new:
        return None

    if apply:
        path.write_bytes(blob[:sb_at] + new + blob[zip_at:])
    return f"{path.name}: {old[2:].rstrip().decode('utf-8', 'replace')}"


def main() -> int:
    apply = "--apply" in sys.argv
    if not SCRIPTS.is_dir():
        print(f"没找到 {SCRIPTS}")
        return 1

    changed = [msg for p in sorted(SCRIPTS.glob("*.exe"))
               if (msg := repair(p, apply)) is not None]

    if not changed:
        print(f"{len(list(SCRIPTS.glob('*.exe')))} 个 exe，路径都已指向 {TARGET}")
        return 0

    for msg in changed:
        print(("  已改 " if apply else "  待改 ") + msg)
    print(f"\n{len(changed)} 个启动器指向了不存在的解释器 -> {TARGET}")
    if not apply:
        print("加 --apply 实际写入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
