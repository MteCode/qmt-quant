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

from flask import Flask, jsonify, render_template, request
from plotly.offline import get_plotlyjs

from . import jobs, loaders
from .registry import TASKS, TASK_BY_ID

app = Flask(__name__)


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


@app.route("/tasks")
def tasks():
    return render_template(
        "tasks.html",
        tasks=TASKS,
        running={r["task_id"] for r in jobs.running_jobs()},
    )


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

    print(f"\n  管理台已启动: http://127.0.0.1:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True)


if __name__ == "__main__":
    main()
