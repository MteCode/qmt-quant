"""对账的下单意图读取。

## 起因：一个永远通过的对账

`reconcile.load_intent()` 原先 glob `*{date}*.json`，而 paper_trade.py
写的是 `executions/exec_{date}.csv`。格式对不上，函数恒返回 []。

intent 在对账里只有一个用途：算 `missing` ——「我下了单，券商侧查无此单」。
委托丢了却没人发现，是实盘最坏的失败模式之一，而这条检查一直是死的，
且表现为「一切正常」。

第二处不匹配藏在后面：CSV 里是 `vt_symbol`（688403.SSE），
reconcile() 比对的是 `xt_code`（688403.SH）。就算把格式改对，
字段名也对不上，missing 仍然恒空。修一个不修另一个等于没修。

## 为什么这些测试要造数据而不是读现有文件

仓库里现存的 exec_2026-09-08.csv 十三行全是预览模式，
真读出来也是 0 笔 —— 和 bug 的表现一模一样。
用现有文件测，改好改坏都是绿的。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRAT = ROOT / "strategies" / "alstm_ppo_csi1000"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STRAT))

COLUMNS = ["time", "run_id", "remark", "vt_symbol", "name", "direction",
           "volume", "price", "amount", "result", "reason", "order_id",
           "mode"]


@pytest.fixture(scope="module")
def rec():
    spec = importlib.util.spec_from_file_location(
        "reconcile_mod", STRAT / "reconcile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_dir: Path, date: str, rows: list[dict]) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"exec_{date}.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    return p


def _row(**kw) -> dict:
    base = {"time": "10:44:57", "run_id": "104457", "remark": "104457_001",
            "vt_symbol": "688403.SSE", "name": "汇成股份", "direction": "卖",
            "volume": "300", "price": "27.08", "amount": "8124.0",
            "result": "已提交", "reason": "清仓", "order_id": "X1",
            "mode": "实盘"}
    base.update(kw)
    return base


@pytest.fixture
def sandbox(rec, tmp_path, monkeypatch):
    """把 STRATEGY_DIR 指到临时目录，避免测试读写真实执行记录。"""
    monkeypatch.setattr(rec.paths, "STRATEGY_DIR", tmp_path, raising=False)
    return tmp_path / "executions"


class TestReadsTheFormatPaperTradeWrites:
    def test_reads_csv(self, rec, sandbox):
        _write(sandbox, "2026-09-08", [_row(), _row(vt_symbol="600353.SSE")])
        got = rec.load_intent("2026-09-08")
        assert len(got) == 2, "应当读到 CSV 里的两笔实盘委托"

    def test_maps_vt_symbol_to_xt_code(self, rec, sandbox):
        """reconcile() 用 xt_code 比对，CSV 里只有 vt_symbol。

        少了这层映射，missing 依然恒空 —— 换个方式坏掉而已。
        """
        _write(sandbox, "2026-09-08",
               [_row(vt_symbol="688403.SSE"), _row(vt_symbol="000001.SZSE")])
        got = rec.load_intent("2026-09-08")
        codes = {g["xt_code"] for g in got}
        assert codes == {"688403.SH", "000001.SZ"}, \
            f"vt_symbol 应转成 xt 代码，实际 {codes}"

    def test_keeps_vt_symbol_too(self, rec, sandbox):
        _write(sandbox, "2026-09-08", [_row()])
        got = rec.load_intent("2026-09-08")
        assert got[0]["vt_symbol"] == "688403.SSE"


class TestPreviewRowsExcluded:
    """预览模式的委托从未提交给券商，不该参与对账。

    把它们算进 missing 会得到一堆假异常，很快就会让人学会忽略这个字段
    —— 结果和恒空一样没用。
    """

    def test_preview_mode_skipped(self, rec, sandbox):
        _write(sandbox, "2026-09-08",
               [_row(mode="预览", result="已预览"),
                _row(mode="预览", result="已预览")])
        assert rec.load_intent("2026-09-08") == []

    def test_mixed_keeps_only_real(self, rec, sandbox):
        _write(sandbox, "2026-09-08", [
            _row(mode="预览", result="已预览", vt_symbol="600353.SSE"),
            _row(mode="实盘", result="已提交", vt_symbol="688403.SSE"),
        ])
        got = rec.load_intent("2026-09-08")
        assert len(got) == 1
        assert got[0]["vt_symbol"] == "688403.SSE"


class TestMissingCheckActuallyWorks:
    """端到端：意图里有、券商侧没有的委托，必须被 reconcile 报出来。

    这是整条链路存在的理由。前面几条测的是解析，这条测的是
    「委托丢了会不会被发现」。
    """

    def test_lost_order_is_reported(self, rec, sandbox):
        _write(sandbox, "2026-09-08", [
            _row(vt_symbol="688403.SSE", order_id="X1"),
            _row(vt_symbol="000001.SZSE", order_id="X2"),
        ])
        intent = rec.load_intent("2026-09-08")
        # 券商侧只认得其中一笔
        broker = {"orders": [{"order_id": "X1", "stock_code": "688403.SH",
                              "direction": "卖出", "order_volume": 300,
                              "status": 56}],
                  "trades": []}
        r = rec.reconcile(intent, broker, {"688403.SH": 27.0})
        lost = {m["xt_code"] for m in r["missing"]}
        assert lost == {"000001.SZ"}, \
            f"丢失的委托应被报出来，实际 missing={r['missing']}"

    def test_nothing_missing_when_all_present(self, rec, sandbox):
        _write(sandbox, "2026-09-08", [_row(vt_symbol="688403.SSE")])
        intent = rec.load_intent("2026-09-08")
        broker = {"orders": [{"order_id": "X1", "stock_code": "688403.SH",
                              "direction": "卖出", "order_volume": 300,
                              "status": 56}],
                  "trades": []}
        r = rec.reconcile(intent, broker, {"688403.SH": 27.0})
        assert r["missing"] == []


class TestUnparsableFilesAreLoud:
    def test_raises_when_nothing_parses(self, rec, sandbox):
        """目录里有文件却一行都读不出，说明格式又变了。

        静默返回 [] 会让对账继续报「一切正常」—— 正是原来那个 bug
        的形态。宁可对账跑不完，也不要一个永远通过的对账。
        """
        sandbox.mkdir(parents=True, exist_ok=True)
        (sandbox / "exec_2026-09-08.json").write_text(
            "{ 这不是合法 json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="解析不出"):
            rec.load_intent("2026-09-08")

    def test_empty_dir_is_fine(self, rec, sandbox):
        sandbox.mkdir(parents=True, exist_ok=True)
        assert rec.load_intent("2026-09-08") == []
