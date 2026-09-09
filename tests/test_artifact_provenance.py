"""产物的来源可追溯性。

## 起因：一份从没被重跑过的文件

我重跑全部旧成本产物，跑完检查「管理台红标为 0、未接入产物为 0」，
以为干净了。实际 `models/t0_market/gating_experiment.csv` 从头到尾
没被重写 —— 因为 `models/t0_market/` 有**三个**生产者：

    optimize_t0_market.py   -> grid.csv, summary.json, best_*.csv
    test_market_gating.py   -> gating_experiment.csv, gating_summary.json
    analyze_t0_deep.py      -> （也往这个目录读写）

我的重跑链里只有第一个。于是目录里有一半文件是新的、一半是旧的，
`ls` 看上去这个目录刚跑过，实际 analyze_t0_deep 和管理台 loader
读的正是那个没更新的文件。我据此给出了一份「新成本下的门控分析」，
数字其实是旧的。

## 这类错误为什么难发现

目录级的新鲜度检查会说「刚更新过」。只有落到**文件**级、
并且知道**谁**该写它，才能发现缺口。

所以这里钉两件事：
1. 每个被读取的产物文件，都要能找到一个声明写它的脚本
2. 同一目录下的文件不应该有过大的时间跨度（一半新一半旧）
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 扫描这些位置找生产者
_SRC_GLOBS = ("scripts/*.py", "strategies/**/*.py", "webui/*.py")

#: 只有这些位置算「生产者」。webui/ 只读产物不写，把它算进来会让
#: 每个目录都变成「多生产者」，判据就没意义了。
_PRODUCER_GLOBS = ("scripts/*.py", "strategies/**/*.py")

#: 目录内文件时间跨度的上限（秒）。超过说明目录里混着不同批次的产物。
#: 一整轮重跑（6 个脚本串行）实测约 2.5 小时，取 6 小时留足余量 ——
#: 这条要抓的是「跨天的陈旧文件」，不是同一轮里的先后差异。
_MAX_SPREAD = 6 * 3600


# 曾经还有一条「每个被读取的产物都要能找到写它的脚本」。删掉了：
# 生产者的文件名多半是 f-string（`f"{args.symbol}_summary.json"`），
# 正则只能抓到模板抓不到解析后的名字，于是对一批**正常产出**的文件报错。
# 一条会稳定误报的测试，最终只会被加 skip 或删掉，不如现在就不留。
#
# 真正的失败模式由下面的 TestNoMixedBatches 覆盖 —— 它不需要知道
# 谁写了哪个文件，只看目录里的文件是不是同一批。


class TestNoMixedBatches:
    """同一目录里不应混着不同批次的产物。"""

    @pytest.mark.parametrize(
        "d", sorted(p.name for p in (ROOT / "models").iterdir()
                    if p.is_dir()) if (ROOT / "models").is_dir() else [])
    def test_files_in_one_dir_are_one_batch(self, d):
        files = [p for p in (ROOT / "models" / d).iterdir()
                 if p.is_file() and p.suffix in (".csv", ".json")]
        if len(files) < 2:
            pytest.skip("文件太少")
        times = sorted(p.stat().st_mtime for p in files)
        spread = times[-1] - times[0]
        if spread <= _MAX_SPREAD:
            return
        names = sorted(files, key=lambda p: p.stat().st_mtime)
        import datetime as _dt

        def _t(p):
            return _dt.datetime.fromtimestamp(
                p.stat().st_mtime).strftime("%m-%d %H:%M")

        pytest.fail(
            f"models/{d} 里的文件跨了 {spread/3600:.1f} 小时，"
            f"可能混着不同批次（有文件没被重跑）：\n  "
            + "\n  ".join(f"{_t(p)}  {p.name}" for p in names))


class TestMultiProducerDirsAreKnown:
    """一个目录被多个脚本写，是上面那个错的根源 —— 重跑时容易只跑其中一个。

    这条不禁止多生产者（有时是合理的），只要求它被显式记录下来，
    让下次做重跑的人知道要跑几个脚本。
    """

    #: 已知的多生产者目录 -> 写它的全部脚本。改动产物布局时同步更新。
    KNOWN = {
        "t0_market": {"optimize_t0_market.py", "test_market_gating.py",
                      "analyze_t0_deep.py"},
        "t0_divergence": {"optimize_t0_divergence.py", "analyze_t0_deep.py"},
        "intraday_gbm": {"train_intraday_gbm.py", "backtest_intraday_gbm.py",
                         "predict_intraday.py"},
    }

    def test_multi_producer_dirs_are_documented(self):
        pat = re.compile(
            r"""ROOT\s*/\s*["']models["']\s*/\s*["']([a-z0-9_]+)["']""")
        found: dict[str, set[str]] = {}
        for g in _PRODUCER_GLOBS:
            for f in ROOT.glob(g):
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for d in pat.findall(src):
                    found.setdefault(d, set()).add(f.name)

        multi = {d: s for d, s in found.items() if len(s) > 1}
        undocumented = {d: s for d, s in multi.items() if d not in self.KNOWN}
        assert not undocumented, (
            "以下目录有多个生产者但没记录在 KNOWN 里 —— "
            "重跑时容易只跑其中一个，留下半新半旧的目录：\n  "
            + "\n  ".join(f"{k}: {sorted(v)}"
                          for k, v in sorted(undocumented.items())))

        for d, expected in self.KNOWN.items():
            if d in found:
                assert found[d] == expected, (
                    f"models/{d} 的生产者变了：\n"
                    f"  记录的: {sorted(expected)}\n"
                    f"  实际的: {sorted(found[d])}")
