-- 数据清洗核对查询集
--
-- 用法：
--   sqlite3 data/market.db < scripts/check_clean.sql
-- 或在任意 SQLite 客户端（DB Browser、DBeaver、VSCode 插件）里逐条执行。
--
-- 设计意图：清洗日志只能说明「脚本认为自己做了什么」，
-- 这些查询是在**结果数据上**独立验证，两者对得上才算清洗成功。

.headers on
.mode column

-- ============================================================
-- 1 脏数据是否真的没了（应全部为 0）
-- ============================================================
SELECT '日线非正价格' AS 检查项,
       COUNT(*) AS 残留行数
FROM daily_bar WHERE open<=0 OR high<=0 OR low<=0 OR close<=0
UNION ALL
SELECT '日线最高<最低', COUNT(*) FROM daily_bar WHERE high < low
UNION ALL
SELECT '日线最高<开盘或收盘', COUNT(*) FROM daily_bar
  WHERE high < open OR high < close
UNION ALL
SELECT '日线最低>开盘或收盘', COUNT(*) FROM daily_bar
  WHERE low > open OR low > close
UNION ALL
SELECT '日线成交量为负', COUNT(*) FROM daily_bar WHERE volume < 0
UNION ALL
SELECT '日线重复(标的+日期)',
       COUNT(*) - COUNT(DISTINCT vt_symbol || date) FROM daily_bar
UNION ALL
SELECT '财报公告日早于报告期', COUNT(*) FROM fin_income
  WHERE m_anntime < m_timetag
UNION ALL
SELECT '财报公告日在未来', COUNT(*) FROM fin_income
  WHERE m_anntime > date('now');

-- ============================================================
-- 2 清洗动作汇总 —— 脚本声称做了什么
-- ============================================================
SELECT dataset AS 数据集,
       rule AS 规则,
       CASE modified WHEN 1 THEN '已修改' ELSE '仅标记' END AS 类型,
       n_rows AS 行数
FROM clean_action
ORDER BY modified DESC, n_rows DESC;

-- ============================================================
-- 3 被清洗最多的标的 —— 定位异常数据源
-- ============================================================
SELECT d.dataset AS 数据集,
       d.vt_symbol AS 标的,
       COALESCE(i.name, '?') AS 名称,
       COALESCE(i.status, '?') AS 状态,
       d.rows_in AS 清洗前,
       d.rows_out AS 清洗后,
       d.rule AS 规则,
       d.n_rows AS 处理行数
FROM clean_detail d
LEFT JOIN instrument i ON i.vt_symbol = d.vt_symbol
ORDER BY d.n_rows DESC
LIMIT 20;

-- ============================================================
-- 4 交叉核对：清洗记录声称清空的标的，库里是否真的没有数据
--    对得上说明清洗动作与结果一致
-- ============================================================
SELECT d.vt_symbol AS 标的,
       COALESCE(i.name, '?') AS 名称,
       d.rows_in AS 清洗前行数,
       d.rows_out AS 清洗后行数,
       (SELECT COUNT(*) FROM daily_bar b
         WHERE b.vt_symbol = d.vt_symbol) AS 库中实际行数
FROM clean_detail d
LEFT JOIN instrument i ON i.vt_symbol = d.vt_symbol
WHERE d.dataset = '1d' AND d.rows_out = 0
ORDER BY d.rows_in DESC
LIMIT 15;

-- ============================================================
-- 5 数据新鲜度：各表最后日期
-- ============================================================
SELECT '日线' AS 数据, MAX(date) AS 最后日期, COUNT(*) AS 行数
FROM daily_bar
UNION ALL
SELECT '龙虎榜', MAX(trade_date), COUNT(*) FROM flow_dragon_tiger_list
UNION ALL
SELECT '两融明细', MAX(trade_date), COUNT(*) FROM flow_margin_detail
UNION ALL
SELECT '资金流因子', MAX(date), COUNT(*) FROM money_factor;

-- ============================================================
-- 6 因子可获得滞后 —— 防前视的关键元信息
--    使用因子时必须按 avail_lag 滞后，否则用到当天还拿不到的数据
-- ============================================================
SELECT factor AS 因子,
       avail_lag AS 可获得滞后天数,
       fill_value AS 缺失补值,
       n_stored AS 存储行数,
       n_dense AS 密集行数,
       first_date AS 起始, last_date AS 结束
FROM factor_meta ORDER BY factor;

-- ============================================================
-- 7 抽样检查：随机看几行真实数据是否合理
-- ============================================================
SELECT b.vt_symbol AS 标的, i.name AS 名称, b.date AS 日期,
       b.open AS 开, b.high AS 高, b.low AS 低, b.close AS 收,
       b.volume AS 成交量,
       ROUND(b.close / b.pre_close - 1, 4) AS 涨跌幅
FROM daily_bar b
LEFT JOIN instrument i ON i.vt_symbol = b.vt_symbol
WHERE b.date = (SELECT MAX(date) FROM daily_bar)
  AND b.pre_close > 0
ORDER BY RANDOM() LIMIT 10;
