rem (0)����汾�ǲ������á�
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

rem (1)��ʾ��Ϣ
echo.���ڽ��е� %num% ��ѭ����������ʼ�Ķ��� %%i ƪ���£�Լ��ʱ30s��

rem (2)ˢ�±�ǩ,3s
adb shell input tap 520 250
choice /t 2 /d y /n >nul

rem (3)��ʼ�Ķ�,2s
adb shell input tap 540 550
choice /t 1 /d y /n >nul

rem (4)��ʼ����
for /l %%j in (1, 1, 10) do (
rem ���ϻ�����Ļ1��
adb shell input swipe 565 1137 565 900
choice /t 2 /d y /n >nul
)
for /l %%k in (1, 1, 10) do (
rem ���»�����Ļ1��
adb shell input swipe 565 700 565 937
choice /t 2 /d y /n >nul
)
rem (5)�����Ķ�,1s
adb shell input tap 47 131
choice /t 1 /d y /n >nul

)

goto start

adb kill-server
pause
