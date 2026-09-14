@echo off
chcp 65001 >nul
title QMT 实时仪表盘
cd /d D:\qmt\qmt

echo ============================================
echo   QMT Quant 实时仪表盘
echo   启动后访问 http://127.0.0.1:8800/dashboard
echo   按 Ctrl+C 停止
echo ============================================
echo.

.venv\Scripts\python.exe -m webui.app --port 8800

pause
