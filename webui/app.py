"""量化研究管理台。

用途：查看回测与实验结果、监控运行中的任务、触发训练/回测/下单。

## 安全约束

默认只监听 127.0.0.1。这不是保守 —— 管理台能触发真实下单，
绑到 0.0.0.0 等于把交易接口暴露给整个局域网。

即便如此，可执行的动作仍然全部走 registry.py 的白名单：浏览器里任何
页面都能向 localhost 发请求，若接口能执行任意命令，一个恶意网页就能
在你机器上跑任意程序。前端只能提交 task_id 与已登记的参数。

下单类任务额外要求二次确认，且预览（dry-run）是默认值。

启动::

    python -m webui.app
    python -m webui.app --port 8800
"""
import argparse
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_file
from pathlib import Path
from plotly.offline import get_plotlyjs

from . import data_browser, jobs, loaders, scheduler
from . import attribution
from . import strategies as strat
from .registry import TASKS, TASK_BY_ID

app = Flask(__name__)
# 模板改动即时生效。默认只在 debug 下重载，而 debug 会起两个调度线程
# 造成定时任务重复触发（见 main()），所以单独把模板重载打开。
app.config["TEMPLATES_AUTO_RELOAD"] = True
ROOT = Path(__file__).resolve().parents[1]
INTRADAY_DIR = ROOT / "strategies" / "intraday_t_920368" / "backtest"
INTRADAY_MODEL_DIR = ROOT / "strategies" / "intraday_t_920368" / "models"
GBM_MODEL_DIR = ROOT / "models" / "intraday_gbm"


@app.template_filter("dur")
def _fmt_duration(seconds) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s} 秒"
    if s < 3600:
        return f"{s // 60} 分 {s % 60} 秒"
    return f"{s // 3600} 时 {(s % 3600) // 60} 分"


@app.route("/plotly.js")
def plotly_js():
    """内联提供 plotly，不依赖 CDN —— 离线环境也要能用。"""
    return app.response_class(get_plotlyjs(), mimetype="application/javascript")


@app.route("/")
def index():
    return render_template(
        "index.html",
        overview=loaders.strategy_overview(),
        equity=loaders.equity_figure(),
        drawdown=loaders.drawdown_figure(),
        running=jobs.running_jobs(),
        recent=jobs.list_jobs(limit=8),
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


@app.route("/experiments")
def experiments():
    return render_template(
        "experiments.html",
        seed=loaders.seed_figure(),
        scaling=loaders.scaling_figure(),
        sweep=loaders.sweep_figure(),
        subperiod=loaders.subperiod_table(),
    )


@app.route("/backtests")
def backtests():
    """已保存模型回测成绩单。"""
    return render_template("backtests.html", results=loaders.backtest_results())


@app.route("/intraday-t-920368")
def intraday_t_page():
    import json
    summary = {}
    metrics = {}
    metric_path = INTRADAY_MODEL_DIR / "gbm_metrics.json"
    if not metric_path.exists():
        metric_path = INTRADAY_MODEL_DIR / "model_metrics.json"
    for p, target in [(INTRADAY_DIR / "summary.json", summary), (metric_path, metrics)]:
        if p.exists():
            try:
                target.update(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
    return render_template("intraday_t.html", summary=summary, metrics=metrics)


@app.route("/intraday-t-920368/report.html")
def intraday_t_report():
    report = INTRADAY_DIR / "report.html"
    if not report.exists():
        return "报告尚未生成", 404
    return send_file(report)


@app.route("/api/intraday-t-920368")
def api_intraday_t():
    import json
    f = request.args.get("file", "")
    if f in {"equity", "trades"}:
        p = INTRADAY_DIR / f"{f}.csv"
        if not p.exists():
            return jsonify(ok=False, error="文件不存在"), 404
        return send_file(p, mimetype="text/csv", as_attachment=False)
    out = {}
    metric_path = INTRADAY_MODEL_DIR / "gbm_metrics.json"
    if not metric_path.exists():
        metric_path = INTRADAY_MODEL_DIR / "model_metrics.json"
    for name, p in (("summary", INTRADAY_DIR / "summary.json"), ("model", metric_path)):
        if p.exists():
            try: out[name] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError): out[name] = {}
    return jsonify(out)


@app.route("/intraday-gbm")
def intraday_gbm_page():
    """全市场日内 GBM 选股模型 —— 训练指标与特征重要性。"""
    import csv
    import json

    metrics, importance, grid = {}, [], []
    mp = GBM_MODEL_DIR / "metrics.json"
    if mp.exists():
        try:
            metrics = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    fp = GBM_MODEL_DIR / "feature_importance.csv"
    if fp.exists():
        try:
            with fp.open(encoding="utf-8") as f:
                importance = [
                    {"feature": r["feature"], "importance": int(r["importance"])}
                    for r in csv.DictReader(f)
                ]
        except (OSError, ValueError, KeyError):
            pass

    gp = GBM_MODEL_DIR / "grid_search_results.csv"
    if gp.exists():
        try:
            with gp.open(encoding="utf-8") as f:
                grid = list(csv.DictReader(f))
        except OSError:
            pass

    backtest = {}
    bp = GBM_MODEL_DIR / "backtest" / "summary.json"
    if bp.exists():
        try:
            backtest = json.loads(bp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    return render_template("intraday_gbm.html", metrics=metrics,
                           importance=importance, grid=grid,
                           backtest=backtest,
                           has_model=(GBM_MODEL_DIR / "model.joblib").exists())


@app.route("/strategies")
def strategies_page():
    """策略中心 —— 所有策略的统一入口：说明、回测、结果。"""
    cat = request.args.get("category", "")
    rows = strat.list_strategies(cat)
    return render_template(
        "strategies.html", rows=rows, category=cat,
        categories=sorted({s.category for s in strat.STRATEGIES}),
        running={r["task_id"] for r in jobs.running_jobs()},
        task_by_id=TASK_BY_ID)


@app.route("/strategies/<sid>")
def strategy_detail(sid):
    s = strat.BY_ID.get(sid)
    if s is None:
        return render_template("strategies.html",
                               rows=strat.list_strategies(), category="",
                               categories=sorted({x.category for x in
                                                  strat.STRATEGIES}),
                               running=set(), task_by_id=TASK_BY_ID,
                               error=f"策略不存在: {sid}"), 404
    return render_template(
        "strategy_detail.html", s=s, r=s.result(),
        task=TASK_BY_ID.get(s.backtest_task),
        live_task=TASK_BY_ID.get(s.live_task),
        running={r["task_id"] for r in jobs.running_jobs()},
        recent=[j for j in jobs.list_jobs(limit=30)
                if j.get("task_id") in (s.backtest_task, s.live_task)][:8])


@app.get("/api/strategy/<sid>/equity")
def api_strategy_equity(sid):
    s = strat.BY_ID.get(sid)
    if s is None:
        return jsonify(ok=False, error="策略不存在"), 404
    r = s.result()
    eq = r.get("equity")
    if not eq:
        return jsonify(ok=False, error="无净值数据"), 404
    return jsonify(eq)


@app.route("/trading")
def trading_page():
    """实盘交易台 —— 按策略归因的委托、成交、持仓与盈亏。

    以**策略**为主轴组织，而不是把所有委托平铺：量化系统最基本的问题
    是「哪个策略在赚钱」，平铺展示回答不了这个。
    """
    from datetime import date as _d

    day = request.args.get("date") or _d.today().isoformat()
    data = attribution.by_strategy(None if day == "all" else day)
    from qmtquant.core.constants import Status
    return render_template(
        "trading.html", day=day,
        BUY=attribution.BUY, SELL=attribution.SELL,
        DEAD={Status.CANCELLED.value, Status.REJECTED.value},
        dates=attribution.order_dates(),
        rows=data["rows"],
        orders=data["orders"], trades=data["trades"],
        positions=loaders.positions(),
        running={r["task_id"] for r in jobs.running_jobs()},
        error=data.get("error"))


@app.get("/api/trading/refresh")
def api_trading_refresh():
    """交易台轮询接口：只回统计数字，够判断要不要重载页面。"""
    from datetime import date as _d
    day = request.args.get("date") or _d.today().isoformat()
    try:
        from qmtquant.config import DATA_DIR
        from qmtquant.store.database import StateStore
        store = StateStore(DATA_DIR / "state.db")
        orders = store.load_orders(day)
        trades = store.load_trades(day)
    except Exception as e:                          # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 500
    from qmtquant.core.constants import Status
    done = {Status.ALLTRADED.value, Status.CANCELLED.value,
            Status.REJECTED.value}
    n_active = sum(1 for o in orders
                   if (o.get("status") or "") not in done)
    return jsonify(ok=True, orders=len(orders), trades=len(trades),
                   active=n_active)


@app.route("/live")
def live_page():
    """实盘监控 —— 在跑的策略、持仓、收益、信号。"""
    return render_template(
        "live.html",
        live=strat.live_status(),
        capable=strat.live_capable(),
        positions=loaders.positions(),
        signal=loaders.selection(request.args.get("signal") or None),
        ex=loaders.executions(request.args.get("date") or None),
        running={r["task_id"] for r in jobs.running_jobs()},
        task_by_id=TASK_BY_ID,
        recent=jobs.list_jobs(limit=10))


@app.route("/t0-single")
def t0_single_page():
    """单标的日内做 T 参数优化与样本外验证。"""
    import csv
    import json

    d = ROOT / "models" / "t0_single"
    summary, grid, wf, daily = {}, [], [], []

    sp = d / "600711_summary.json"
    if sp.exists():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    for name, target in (("600711_grid.csv", grid),
                         ("600711_best_daily.csv", daily)):
        p = d / name
        if p.exists():
            try:
                with p.open(encoding="utf-8") as f:
                    target.extend(list(csv.DictReader(f)))
            except OSError:
                pass

    wf = summary.get("walkforward", [])
    return render_template("t0_single.html", summary=summary,
                           grid=grid[:20], wf=wf, daily=daily,
                           has_result=bool(summary))


@app.get("/api/t0-single/equity")
def api_t0_single_equity():
    import csv
    p = ROOT / "models" / "t0_single" / "600711_best_daily.csv"
    if not p.exists():
        return jsonify(ok=False, error="回测结果不存在"), 404
    try:
        with p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        return jsonify(ok=False, error=str(e)), 500
    return jsonify(dates=[r["date"] for r in rows],
                   equity=[float(r["equity"]) for r in rows],
                   close=[float(r["close"]) for r in rows])


@app.get("/api/intraday-gbm/equity")
def api_intraday_gbm_equity():
    """净值曲线，供前端画图。"""
    import csv
    mode = request.args.get("mode", "momentum")
    if mode not in {"t_plus_0", "mean_reversion", "momentum"}:
        return jsonify(ok=False, error="未知模式"), 400
    p = GBM_MODEL_DIR / "backtest" / f"{mode}_daily.csv"
    if not p.exists():
        return jsonify(ok=False, error="回测结果不存在"), 404
    try:
        with p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        return jsonify(ok=False, error=str(e)), 500
    return jsonify(
        dates=[r["date"] for r in rows],
        equity=[float(r["equity"]) for r in rows],
    )


@app.get("/api/intraday-gbm")
def api_intraday_gbm():
    import json
    mp = GBM_MODEL_DIR / "metrics.json"
    if not mp.exists():
        return jsonify(ok=False, error="模型尚未训练"), 404
    try:
        return jsonify(json.loads(mp.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/tasks")
def tasks():
    return render_template(
        "tasks.html",
        tasks=TASKS,
        running={r["task_id"] for r in jobs.running_jobs()},
    )


@app.route("/selection")
def selection_page():
    """实盘执行 —— 策略实际下了什么单。"""
    ex = loaders.executions(request.args.get("date") or None)
    sel = loaders.selection(request.args.get("signal") or None)
    return render_template(
        "selection.html", ex=ex, sel=sel,
        pos=loaders.positions(),
        industry=loaders.industry_figure(sel) if sel else None,
        picks=loaders.backtest_picks(),
    )


@app.route("/data")
def data_page():
    src = request.args.get("source", "qlib")
    sym = request.args.get("symbol", "")
    q = request.args.get("q", "")

    result, issues = None, []
    if sym:
        result = (data_browser.read_qlib(sym) if src == "qlib"
                  else data_browser.read_parquet_source(src, sym))
        if result and "df" in result:
            issues = data_browser.check_quality(result["df"], src, sym)

    return render_template(
        "data.html",
        sources=data_browser.SOURCES,
        source=src, symbol=sym, query=q,
        symbols=data_browser.list_symbols(src, query=q),
        result=result, issues=issues,
        overview=data_browser.overview(),
        consistency=(data_browser.qlib_consistency()
                     if src == "qlib" and not sym else None),
    )


@app.get("/api/data/export")
def api_data_export():
    """导出为 CSV，供 Excel 等工具打开。"""
    src = request.args.get("source", "qlib")
    sym = request.args.get("symbol", "")
    if not sym:
        return jsonify(ok=False, error="缺少 symbol"), 400

    r = (data_browser.read_qlib(sym, tail=10 ** 9) if src == "qlib"
         else data_browser.read_parquet_source(src, sym, tail=10 ** 9))
    df = r.get("full") if r.get("full") is not None else r.get("df")
    if df is None:
        return jsonify(ok=False, error=r.get("error", "无数据")), 404

    # utf-8-sig 让 Excel 正确识别中文，否则打开是乱码
    csv = df.to_csv(encoding="utf-8-sig")
    return app.response_class(
        csv, mimetype="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="{src}_{sym}.csv"'})


@app.route("/schedule")
def schedule_page():
    return render_template(
        "schedule.html",
        schedules=scheduler.list_schedules(),
        scheduler_on=scheduler.is_running(),
        weekday_names=scheduler.WEEKDAY_NAMES,
    )


@app.post("/api/schedule/<sched_id>/enabled")
def api_schedule_enabled(sched_id):
    data = request.get_json(silent=True) or {}
    ok = scheduler.set_enabled(sched_id, bool(data.get("enabled")))
    return jsonify(ok=ok)


@app.post("/api/schedule/<sched_id>")
def api_schedule_update(sched_id):
    data = request.get_json(silent=True) or {}
    ok = scheduler.update(sched_id, data.get("time"),
                          data.get("weekdays"), data.get("catchup_hours"))
    return jsonify(ok=ok, error=None if ok else "时间格式应为 HH:MM")


@app.post("/api/schedule/<sched_id>/run")
def api_schedule_run(sched_id):
    try:
        job_id = scheduler.trigger_now(sched_id)
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 409
    if job_id is None:
        return jsonify(ok=False, error="计划不存在"), 404
    return jsonify(ok=True, job_id=job_id)


@app.route("/jobs")
def job_list():
    return render_template("jobs.html", rows=jobs.list_jobs(limit=50))


@app.route("/jobs/<job_id>")
def job_detail(job_id):
    row = jobs.get_job(job_id)
    if row is None:
        return render_template("jobs.html", rows=jobs.list_jobs(limit=50),
                               error=f"任务不存在: {job_id}"), 404
    return render_template("job_detail.html", job=row,
                           log=jobs.read_log(job_id))


# ------------------------------------------------------------------ API

@app.post("/api/run")
def api_run():
    data = request.get_json(silent=True) or request.form.to_dict()
    task_id = data.get("task_id", "")
    task = TASK_BY_ID.get(task_id)
    if task is None:
        return jsonify(ok=False, error="未登记的任务"), 400

    # 下单类任务必须显式确认，防止误点或跨站请求触发真实委托
    if task.dangerous:
        dry = data.get("dry_run") in (True, "true", "True", "1", 1, "on")
        if not dry and data.get("confirm") != "yes":
            return jsonify(ok=False,
                           error="该任务会产生真实委托，需要二次确认"), 400

    try:
        job = jobs.start(task_id, data)
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 409
    except (ValueError, OSError) as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, job_id=job.id)


@app.post("/api/stop/<job_id>")
def api_stop(job_id):
    return jsonify(ok=jobs.stop(job_id))


@app.get("/api/jobs")
def api_jobs():
    return jsonify(jobs.list_jobs(limit=30))


@app.get("/api/backtests")
def api_backtests():
    """以 JSON 形式提供成绩单，便于外部监控或二次展示。"""
    return jsonify(loaders.backtest_results())


@app.get("/api/log/<job_id>")
def api_log(job_id):
    row = jobs.get_job(job_id)
    return jsonify(status=row["status"] if row else "unknown",
                   log=jobs.read_log(job_id))


def main():
    p = argparse.ArgumentParser(description="量化研究管理台")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--host", default="127.0.0.1",
                    help="默认只监听本机。管理台能触发真实下单，"
                         "改成 0.0.0.0 等于把交易接口暴露给局域网")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"[警告] 监听 {args.host} —— 管理台可触发真实委托，"
              f"确认所在网络可信再继续")

    # debug 模式下 Flask 会重启进程，会起两个调度线程造成重复触发
    if not args.debug:
        scheduler.start_scheduler()
        n = sum(1 for s in scheduler.list_schedules() if s["enabled"])
        print(f"  定时调度已启动（{n} 条计划生效）")

    print(f"\n  管理台已启动: http://127.0.0.1:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True)


if __name__ == "__main__":
    main()
