"""Qlib 初始化的统一入口。

存在的理由是一个具体的坑：Qlib 的内存缓存 `MemCacheUnit` 默认只存 **500 条**
（`mem_cache_size_limit`，按条数计），而它的 `__setitem__` 并不是线程安全的 ——
Qlib 自己的源码里就留着 `# TODO: thread safe?` 这行注释。

于是在多线程读取时会发生这样的竞争：

1. 线程 A 把 `('$close', 'sz002524', 0, 2587, 'day')` 写进缓存
2. 其它线程继续写入，把缓存撑过 500 条上限
3. LRU 淘汰掉了 A 刚写的那条
4. A 回头去读，`KeyError`

触发条件是**并发峰值需求超过缓存上限**。以 Alpha360 为例，它有 360 个表达式，
每个 worker 处理一只股票就要写读 360 条：

    26 个 worker x 360 个表达式 = 峰值 9360 条  >>  上限 500 条

必然触发淘汰。现象是随机某只股票、随机某个字段报 KeyError，重跑一次换一只，
索引部分固定不变（那是整个请求的 end_index，所有 key 都一样）。

核数越多的机器越容易中招 —— 并发峰值需求是线性放大的。

因此这里按「并发数 x 表达式数」估算上限并留出余量。代价是内存：一条约 20KB，
20000 条约 400MB，对跑训练的机器可以接受。
"""
import logging

logger = logging.getLogger(__name__)

#: 单个缓存条目的经验大小（一只股票 x 一个字段 x 全历史，约 2600 个 float）
_ENTRY_BYTES = 20 * 1024

#: 缓存上限的下限。低于这个数即使小 universe 也可能抖动
_MIN_CACHE_ENTRIES = 2000


def estimate_cache_limit(n_expressions: int, workers: int | None = None,
                         safety: float = 2.0) -> int:
    """按并发峰值估算所需的缓存条数。

    :param n_expressions: 一次请求里的表达式个数（Alpha360 是 360）
    :param workers: 并发 worker 数，None 表示读 Qlib 配置
    :param safety: 余量系数。worker 之间的进度并不整齐，取 2 倍留出空间
    """
    if workers is None:
        from qlib.config import C
        try:
            workers = C.get_kernels("day")
        except Exception:
            workers = 8
    need = int(workers * n_expressions * safety)
    return max(need, _MIN_CACHE_ENTRIES)


def init_qlib(provider_uri: str, n_expressions: int = 360,
              region: str = "cn", **kwargs):
    """初始化 Qlib，并把内存缓存上限调到不会触发淘汰竞争的水平。

    :param provider_uri: Qlib 数据目录
    :param n_expressions: 预计一次请求的表达式个数。Alpha360 用 360；
        只读少数几个因子的场景可以调小，省内存
    :param region: 市场，默认 cn
    :param kwargs: 透传给 ``qlib.init``，可覆盖这里的任何默认值

    用 ``joblib_backend="threading"``：Qlib 默认的多进程后端在 Windows 上
    要重复 pickle 数据，反而更慢。
    """
    import qlib

    limit = kwargs.pop("mem_cache_size_limit", None)
    if limit is None:
        limit = estimate_cache_limit(n_expressions)

    settings = {
        "provider_uri": provider_uri,
        "region": region,
        "joblib_backend": "threading",
        "mem_cache_size_limit": limit,
    }
    settings.update(kwargs)

    qlib.init(**settings)

    approx_mb = limit * _ENTRY_BYTES / 1024 / 1024
    logger.info("Qlib 已初始化 uri=%s 缓存上限=%d 条（约 %.0f MB）",
                provider_uri, limit, approx_mb)
    return limit
