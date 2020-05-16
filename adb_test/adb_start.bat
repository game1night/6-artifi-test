rem (0)这个版本是测试适用。
pause
@echo off
cd %~dp0
cd platform-tools_r28.0.1-windows/platform-tools/
adb start-server
adb devices
pause
