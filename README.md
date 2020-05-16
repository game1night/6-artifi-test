6 - a application demo related to test-automotation via android devices

## 6-artifi-test

### Intro

图片识别？算法？这些都太难啦！试试从切割、探索、标注图片？具体的可以查阅上一个话题：[5-just-findout](https://github.com/game1night/5-just-findout)，本项目是“妥协”了！

Learn the core-calculate part from [airtest](https://github.com/AirtestProject/Airtest) and rewrite down the related-import-code-snippets for this test automation. It's amazing. I never have thought I could learn and rewrite down such a complex project. Wish you luck and fun.

Since the game is built under the LICENSE of GPL 3.0 so you can share it under it. If there were anything wrong, feel free to leave messages to let me know or fix. Thanks.

### Start

There're two parts.

First are the files ended with '.bat'. Also 'ps1', well, changed-version doesn't work well on my personal computer yet. They are used to be test-mechanically.

Second are the files under the same directory as 'adb.exe'. Need source images for test, which I didn't put them in the file. Also the core-recognize and classification  part is via the 'api', which is provided by the AI solution platform on [Baidu Cloud](https://cloud.baidu.com).

### Rebuild

准备工作

- 了解Windows的`bat`。
- 了解Android的`adb`;
- 阅读官方文档：https://developer.android.com/studio/command-line/adb?hl=zh-cn#top_of_page 。
- 下载所需的工具`android_sdk/platform-tools`。

检验工具
```
dir
cd platform-tools_r28.0.1-windows\platform-tools
adb devices
```

连接设备
- USB、WLAN，详见上面提到的文档。
- 打开`USB调试`模式（可以在“模拟点击”部分再打开`USB调试（安全设置）`；
- 测试命令：`adb devices`。

adb命令参考
- 您可以在开发计算机上从命令行发出 adb 命令，或通过脚本发出。
- 如果只有一个模拟器在运行或只连接了一个设备，则默认情况下将 adb 命令发送至该设备。如果有多个模拟器在运行和/或连接了多个设备，您需要使用 -d、-e 或 -s 选项指定应向其发送命令的目标设备。用法如下：
```
adb [-d|-e|-s serial_number] command
```

发出shell命令
- 您可以使用 shell 命令通过 adb 发出设备命令，可以进入或不进入模拟器/设备实例上的 adb 远程 shell，进入如下：
```
adb [-d|-e|-s serial_number] shell [shell_command]
```
- 当您准备退出远程 shell 时，按 Control + D 或输入 exit。
- shell 命令二进制文件存储在模拟器或设备的文件系统中，其路径为 /system/bin/。
- 说明：
```
..\platform-tools>adb shell
chiron:/ $ input
Usage: input [<source>] <command> [<arg>...]

The sources are:
      dpad
      keyboard
      mouse
      touchpad
      gamepad
      touchnavigation
      joystick
      touchscreen
      stylus
      trackball

The commands and default sources are:
      text <string> (Default: touchscreen)
      keyevent [--longpress] <key code number or name> ... (Default: keyboard)
      tap <x> <y> (Default: touchscreen)
      swipe <x1> <y1> <x2> <y2> [duration(ms)] (Default: touchscreen)
      draganddrop <x1> <y1> <x2> <y2> [duration(ms)] (Default: touchscreen)
      press (Default: trackball)
      roll <dx> <dy> (Default: trackball)
      tmode <tmode>
chiron:/ $
```

举例：截屏
- 给手机截屏，并将截图命名，并放在特定的文件夹内：
```
adb shell screencap /sdcard/screen.png
```
- 将截图复制、传送，并放入命令行本地的一个文件夹内：
```
adb pull /sdcard/screen.png
```
- 给图片重复命名，手机上自动起了新的名字，会导致从模拟器或设备复制失败。

模拟点击
- 打开`USB调试（安全设置）`；
- 打开`指针位置`，这样通过手在屏幕上的操作，能知道坐标系；
- 可以打开`显示布局边界`，虽然乱了点，但能辅助我们观察内容的布局。

adb服务器的开启和关闭
- `start-server`
- `kill-server`

批处理脚本
- 中文路径需要存为“ANSI”，否则会乱码，sublime不给力，下了各notepad++。
- `ping 192.0.2.2 -n 1 -w 10000 > nul`
- `choice /t 5 /d y /n >nul`

