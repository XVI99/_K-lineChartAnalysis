@echo off
chcp 65001 >nul
title AStockQuant ETF量化交易平台
cd /d "f:\_K-lineChartAnalysis\AStockQuant"
echo ============================================================
echo   AStockQuant ETF量化交易平台 - GOAL模式
echo ============================================================
echo   正在启动Web服务...
echo   访问地址: http://localhost:5001
echo   ETF数量: 103只
echo   持仓管理: 支持录入交易/查看详情/卖出建议
echo ============================================================
echo.
start http://localhost:5001
"F:\_K-lineChartAnalysis\.venv\Scripts\python.exe" web_app.py --port 5001
pause