@echo off
REM Quick benchmark script for LLAISYS performance testing (Windows)
REM Usage: quick_benchmark.bat [path\to\model] [device]

setlocal

set MODEL_PATH=%1
set DEVICE=%2

if "%DEVICE%"=="" set DEVICE=cpu

if "%MODEL_PATH%"=="" (
    echo Usage: quick_benchmark.bat [path\to\model] [device]
    echo   device: cpu ^(default^) or nvidia
    exit /b 1
)

echo ==========================================
echo LLAISYS Quick Performance Benchmark
echo ==========================================
echo Model: %MODEL_PATH%
echo Device: %DEVICE%
echo ==========================================
echo.

echo Checking dependencies...
python -c "import matplotlib, psutil, numpy" 2>nul
if errorlevel 1 (
    echo Installing benchmark dependencies...
    pip install -r test\benchmark_requirements.txt
)

echo.
echo Step 1: Operator-level benchmark
echo ==========================================
python test\benchmark_operators.py --device %DEVICE% --operators linear self_attention rms_norm

echo.
echo Step 2: End-to-end inference benchmark
echo ==========================================
python test\benchmark_inference.py --model "%MODEL_PATH%" --device %DEVICE% --max_tokens 64

echo.
echo ==========================================
echo Benchmark complete!
echo ==========================================
echo.
echo Results saved to:
echo   - benchmark_results\benchmark_results.json
echo   - benchmark_results\*.png ^(charts^)
echo   - operator_benchmark.json
echo.
echo Next steps:
echo   1. Review the speedup factors in the summary
echo   2. Identify bottleneck operators ^(lowest speedup^)
echo   3. See docs\PERFORMANCE_BENCHMARKING.md for optimization guide
echo.

endlocal
