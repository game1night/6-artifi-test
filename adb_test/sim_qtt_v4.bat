rem (0)这个版本是测试适用。
pause
@echo off
cd %~dp0
cd platform-tools_r28.0.1-windows/platform-tools/
adb start-server
adb devices
pause

set num = 0

:start
set /a num += 1

for /l %%i in (1, 1, 120) do (

rem (1)提示信息
echo.正在进行第 %num% 次循环，即将开始阅读第 %%i 篇文章！约耗时30s。

rem (2)刷新标签,3s
adb shell input tap 520 250
choice /t 2 /d y /n >nul

rem (3)开始阅读,2s
adb shell input tap 540 550
choice /t 1 /d y /n >nul

rem (4)开始滑屏
for /l %%j in (1, 1, 10) do (
rem 向上滑动屏幕1次
adb shell input swipe 565 1137 565 900
choice /t 2 /d y /n >nul
)
for /l %%k in (1, 1, 10) do (
rem 向下滑动屏幕1次
adb shell input swipe 565 700 565 937
choice /t 2 /d y /n >nul
)
rem (5)结束阅读,1s
adb shell input tap 47 131
choice /t 1 /d y /n >nul

)

goto start

adb kill-server
pause
