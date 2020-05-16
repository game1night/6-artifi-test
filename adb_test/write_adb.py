#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/8/7 9:56

@author: tatatingting
"""

c0 = """
rem (0)This a new test.
@echo off
cd %~dp0
cd platform-tools_r28.0.1-windows/platform-tools/
adb start-server
adb devices
pause

"""


def init():
    with open('test.bat', 'w', encoding='utf-8-sig') as f:
        f.write(c0+'\n')


def add_content(c):
    with open('test.bat', 'a', encoding='utf-8-sig') as f:
        f.write(c+'\n')


def rem(c):
    add_content('rem {}'.format(c))


def sleep(n):
    add_content('choice /t {} /d y /n >nul'.format(n))


def go(list):
    for i in list:
        x = i[0]
        y = i[1]
        add_content('adb shell input tap {} {}'.format(x, y))
        add_content('choice /t 2 /d y /n >nul')


def swipe(list):
    x1 = list[0][0]
    y1 = list[0][1]
    x2 = list[1][0]
    y2 = list[1][1]
    add_content('adb shell input swipe {} {} {} {}'.format(x1, y1, x2, y2))
    add_content('choice /t 2 /d y /n >nul')


def loop_swipe(list, n):
    x1 = list[0][0]
    y1 = list[0][1]
    x2 = list[1][0]
    y2 = list[1][1]
    add_content('for /l %%j in (1, 1, {}) do ('.format(n))
    add_content('adb shell input swipe {} {} {} {}'.format(x1, y1, x2, y2))
    add_content('choice /t 2 /d y /n >nul')
    add_content(')')


def swipe_right_to_left():
    swipe([[789, 1261], [404, 1261]])


def swipe_left_to_right():
    swipe([[404, 1261], [789, 1261]])


def swipe_top_to_down():
    swipe([[565, 700], [565, 937]])


def swipe_down_to_top():
    swipe([[565, 1137], [565, 900]])


def end():
    add_content('adb kill-server')
    add_content('cd ..')
    add_content('cd ..')


def qtt():
    init()
    d = {
        5: [967, 2093],
        4: [749, 2093],
        3: [532, 2093],
        2: [310, 2093],
        1: [85, 2093],
        'vip': [503, 218],
        'daily': [348, 1380],
        'exp_tree': [348, 1661],
        'shake_exp_tree': [525, 1087],
        'back': [46, 146],
        'coin_bottle': [660, 1283],
        'tag_side': [922, 255],
        'tag_middle': [492, 255],
        'message_first': [492, 450],
    }



if __name__ == '__main__':
    init()
    if 1:
        1

    end()
