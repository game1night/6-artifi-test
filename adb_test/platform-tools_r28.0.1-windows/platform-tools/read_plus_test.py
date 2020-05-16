#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/5/6 12:45

@author: tatatingting
"""

import os
import subprocess
import sys
import time
from io import BytesIO
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import airtest.core.api as aci


def generate_result(middle_point, pypts, confi):
    """Format the result: 定义图像识别结果格式."""
    ret = dict(result=middle_point,
               rectangle=pypts,
               confidence=confi)
    return ret


def check_image_valid(im_source, im_search):
    """Check if the input images valid or not."""
    if im_source is not None and im_source.any() and im_search is not None and im_search.any():
        return True
    else:
        return False


def img_mat_rgb_2_gray(img_mat):
    """
    Turn img_mat into gray_scale, so that template match can figure the img data.
    "print(type(im_search[0][0])")  can check the pixel type.
    """
    assert isinstance(img_mat[0][0], np.ndarray), "input must be instance of np.ndarray"
    return cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)


def cal_ccoeff_confidence(im_source, im_search):
    """求取两张图片的可信度，使用TM_CCOEFF_NORMED方法."""
    im_source, im_search = img_mat_rgb_2_gray(im_source), img_mat_rgb_2_gray(im_search)
    res = cv2.matchTemplate(im_source, im_search, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    confidence = max_val
    return confidence


def cal_rgb_confidence(img_src_rgb, img_sch_rgb):
    """同大小彩图计算相似度."""
    # BGR三通道心理学权重:
    weight = (0.114, 0.587, 0.299)
    src_bgr, sch_bgr = cv2.split(img_src_rgb), cv2.split(img_sch_rgb)

    # 计算BGR三通道的confidence，存入bgr_confidence:
    bgr_confidence = [0, 0, 0]
    for i in range(3):
        res_temp = cv2.matchTemplate(src_bgr[i], sch_bgr[i], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_temp)
        bgr_confidence[i] = max_val

    # 加权可信度
    weighted_confidence = bgr_confidence[0] * weight[0] + bgr_confidence[1] * weight[1] + bgr_confidence[2] * weight[2]

    return weighted_confidence


class KeypointMatching(object):
    """基于特征点的识别基类: KAZE."""

    # 日志中的方法名
    METHOD_NAME = "KAZE"
    # 参数: FILTER_RATIO为SIFT优秀特征点过滤比例值(0-1范围，建议值0.4-0.6)
    FILTER_RATIO = 0.59
    # 参数: SIFT识别时只找出一对相似特征点时的置信度(confidence)
    ONE_POINT_CONFI = 0.5

    def __init__(self, im_search, im_source, threshold=0.8, rgb=True):
        super(KeypointMatching, self).__init__()
        self.im_source = im_source
        self.im_search = im_search
        self.threshold = threshold
        self.rgb = rgb

    def mask_kaze(self):
        """基于kaze查找多个目标区域的方法."""
        # 求出特征点后，self.im_source中获得match的那些点进行聚类
        raise NotImplementedError

    def find_all_results(self):
        """基于kaze查找多个目标区域的方法."""
        # 求出特征点后，self.im_source中获得match的那些点进行聚类
        raise NotImplementedError

    def find_best_result(self):
        """基于kaze进行图像识别，只筛选出最优区域."""
        # 第一步：检验图像是否正常：
        if not check_image_valid(self.im_source, self.im_search):
            return None

        # 第二步：获取特征点集并匹配出特征点对: 返回值 good, pypts, kp_sch, kp_src
        self.kp_sch, self.kp_src, self.good = self._get_key_points()

        # 第三步：根据匹配点对(good),提取出来识别区域:
        if len(self.good) in [0, 1]:
            # 匹配点对为0,无法提取识别区域;为1则无法获取目标区域,直接返回None作为匹配结果:
            return None
        elif len(self.good) in [2, 3]:
            # 匹配点对为2或3,根据点对求出目标区域,据此算出可信度:
            if len(self.good) == 2:
                origin_result = self._handle_two_good_points(self.kp_sch, self.kp_src, self.good)
            else:
                origin_result = self._handle_three_good_points(self.kp_sch, self.kp_src, self.good)
            # 某些特殊情况下直接返回None作为匹配结果:
            if origin_result is None:
                return origin_result
            else:
                middle_point, pypts, w_h_range = origin_result
        else:
            # 匹配点对 >= 4个，使用单矩阵映射求出目标区域，据此算出可信度：
            middle_point, pypts, w_h_range = self._many_good_pts(self.kp_sch, self.kp_src, self.good)

        # 第四步：根据识别区域，求出结果可信度，并将结果进行返回:
        # 对识别结果进行合理性校验: 小于5个像素的，或者缩放超过5倍的，一律视为不合法直接raise.
        self._target_error_check(w_h_range)
        # 将截图和识别结果缩放到大小一致,准备计算可信度
        x_min, x_max, y_min, y_max, w, h = w_h_range
        target_img = self.im_source[y_min:y_max, x_min:x_max]
        resize_img = cv2.resize(target_img, (w, h))
        confidence = self._cal_confidence(resize_img)

        best_match = generate_result(middle_point, pypts, confidence)
        return best_match if confidence >= self.threshold else None

    def show_match_image(self):
        """Show how the keypoints matches."""
        from random import random
        h_sch, w_sch = self.im_search.shape[:2]
        h_src, w_src = self.im_source.shape[:2]

        # first you have to do the matching
        self.find_best_result()
        # then initialize the result image:
        matching_info_img = np.zeros([max(h_sch, h_src), w_sch + w_src, 3], np.uint8)
        matching_info_img[:h_sch, :w_sch, :] = self.im_search
        matching_info_img[:h_src, w_sch:, :] = self.im_source
        # render the match image at last:
        for m in self.good:
            color = tuple([int(random() * 255) for _ in range(3)])
            cv2.line(matching_info_img, (int(self.kp_sch[m.queryIdx].pt[0]), int(self.kp_sch[m.queryIdx].pt[1])),
                     (int(self.kp_src[m.trainIdx].pt[0] + w_sch), int(self.kp_src[m.trainIdx].pt[1])), color)

        return matching_info_img

    def _cal_confidence(self, resize_img):
        """计算confidence."""
        if self.rgb:
            confidence = cal_rgb_confidence(self.im_search, resize_img)
        else:
            confidence = cal_ccoeff_confidence(self.im_search, resize_img)
        # confidence修正
        confidence = (1 + confidence) / 2
        return confidence

    def init_detector(self):
        """Init keypoint detector object."""
        self.detector = cv2.KAZE_create()
        # create BFMatcher object:
        self.matcher = cv2.BFMatcher(cv2.NORM_L1)  # cv2.NORM_L1 cv2.NORM_L2 cv2.NORM_HAMMING(not useable)

    def get_keypoints_and_descriptors(self, image):
        """获取图像特征点和描述符."""
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_keypoints(self, des_sch, des_src):
        """Match descriptors (特征值匹配)."""
        # 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
        return self.matcher.knnMatch(des_sch, des_src, k=2)

    def _get_key_points(self):
        """根据传入图像,计算图像所有的特征点,并得到匹配特征点对."""
        # 准备工作: 初始化算子
        self.init_detector()
        # 第一步：获取特征点集，并匹配出特征点对: 返回值 good, pypts, kp_sch, kp_src
        kp_sch, des_sch = self.get_keypoints_and_descriptors(self.im_search)
        kp_src, des_src = self.get_keypoints_and_descriptors(self.im_source)
        # When apply knnmatch , make sure that number of features in both test and
        #       query image is greater than or equal to number of nearest neighbors in knn match.
        if len(kp_sch) < 2 or len(kp_src) < 2:
            print("Not enough feature points in input images !")
        # match descriptors (特征值匹配)
        matches = self.match_keypoints(des_sch, des_src)

        # good为特征点初选结果，剔除掉前两名匹配太接近的特征点，不是独特优秀的特征点直接筛除(多目标识别情况直接不适用)
        good = []
        for m, n in matches:
            if m.distance < self.FILTER_RATIO * n.distance:
                good.append(m)
        # good点需要去除重复的部分，（设定源图像不能有重复点）去重时将src图像中的重复点找出即可
        # 去重策略：允许搜索图像对源图像的特征点映射一对多，不允许多对一重复（即不能源图像上一个点对应搜索图像的多个点）
        good_diff, diff_good_point = [], [[]]
        for m in good:
            diff_point = [int(kp_src[m.trainIdx].pt[0]), int(kp_src[m.trainIdx].pt[1])]
            if diff_point not in diff_good_point:
                good_diff.append(m)
                diff_good_point.append(diff_point)
        good = good_diff

        return kp_sch, kp_src, good

    def _handle_two_good_points(self, kp_sch, kp_src, good):
        """处理两对特征点的情况."""
        pts_sch1 = int(kp_sch[good[0].queryIdx].pt[0]), int(kp_sch[good[0].queryIdx].pt[1])
        pts_sch2 = int(kp_sch[good[1].queryIdx].pt[0]), int(kp_sch[good[1].queryIdx].pt[1])
        pts_src1 = int(kp_src[good[0].trainIdx].pt[0]), int(kp_src[good[0].trainIdx].pt[1])
        pts_src2 = int(kp_src[good[1].trainIdx].pt[0]), int(kp_src[good[1].trainIdx].pt[1])

        return self._get_origin_result_with_two_points(pts_sch1, pts_sch2, pts_src1, pts_src2)

    def _handle_three_good_points(self, kp_sch, kp_src, good):
        """处理三对特征点的情况."""
        # 拿出sch和src的两个点(点1)和(点2点3的中点)，
        # 然后根据两个点原则进行后处理(注意ke_sch和kp_src以及queryIdx和trainIdx):
        pts_sch1 = int(kp_sch[good[0].queryIdx].pt[0]), int(kp_sch[good[0].queryIdx].pt[1])
        pts_sch2 = int((kp_sch[good[1].queryIdx].pt[0] + kp_sch[good[2].queryIdx].pt[0]) / 2), int(
            (kp_sch[good[1].queryIdx].pt[1] + kp_sch[good[2].queryIdx].pt[1]) / 2)
        pts_src1 = int(kp_src[good[0].trainIdx].pt[0]), int(kp_src[good[0].trainIdx].pt[1])
        pts_src2 = int((kp_src[good[1].trainIdx].pt[0] + kp_src[good[2].trainIdx].pt[0]) / 2), int(
            (kp_src[good[1].trainIdx].pt[1] + kp_src[good[2].trainIdx].pt[1]) / 2)
        return self._get_origin_result_with_two_points(pts_sch1, pts_sch2, pts_src1, pts_src2)

    def _many_good_pts(self, kp_sch, kp_src, good):
        """特征点匹配点对数目>=4个，可使用单矩阵映射,求出识别的目标区域."""
        sch_pts, img_pts = np.float32([kp_sch[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2), np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # M是转化矩阵
        M, mask = self._find_homography(sch_pts, img_pts)
        matches_mask = mask.ravel().tolist()
        # 从good中间筛选出更精确的点(假设good中大部分点为正确的，由ratio=0.7保障)
        selected = [v for k, v in enumerate(good) if matches_mask[k]]

        # 针对所有的selected点再次计算出更精确的转化矩阵M来
        sch_pts, img_pts = np.float32([kp_sch[m.queryIdx].pt for m in selected]).reshape(
            -1, 1, 2), np.float32([kp_src[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
        M, mask = self._find_homography(sch_pts, img_pts)
        # 计算四个角矩阵变换后的坐标，也就是在大图中的目标区域的顶点坐标:
        h, w = self.im_search.shape[:2]
        h_s, w_s = self.im_source.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # trans numpy arrary to python list: [(a, b), (a1, b1), ...]
        def cal_rect_pts(dst):
            return [tuple(npt[0]) for npt in dst.astype(int).tolist()]

        pypts = cal_rect_pts(dst)
        # 注意：虽然4个角点有可能越出source图边界，但是(根据精确化映射单映射矩阵M线性机制)中点不会越出边界
        lt, br = pypts[0], pypts[2]
        middle_point = int((lt[0] + br[0]) / 2), int((lt[1] + br[1]) / 2)
        # 考虑到算出的目标矩阵有可能是翻转的情况，必须进行一次处理，确保映射后的“左上角”在图片中也是左上角点：
        x_min, x_max = min(lt[0], br[0]), max(lt[0], br[0])
        y_min, y_max = min(lt[1], br[1]), max(lt[1], br[1])
        # 挑选出目标矩形区域可能会有越界情况，越界时直接将其置为边界：
        # 超出左边界取0，超出右边界取w_s-1，超出下边界取0，超出上边界取h_s-1
        # 当x_min小于0时，取0。  x_max小于0时，取0。
        x_min, x_max = int(max(x_min, 0)), int(max(x_max, 0))
        # 当x_min大于w_s时，取值w_s-1。  x_max大于w_s-1时，取w_s-1。
        x_min, x_max = int(min(x_min, w_s - 1)), int(min(x_max, w_s - 1))
        # 当y_min小于0时，取0。  y_max小于0时，取0。
        y_min, y_max = int(max(y_min, 0)), int(max(y_max, 0))
        # 当y_min大于h_s时，取值h_s-1。  y_max大于h_s-1时，取h_s-1。
        y_min, y_max = int(min(y_min, h_s - 1)), int(min(y_max, h_s - 1))
        # 目标区域的角点，按左上、左下、右下、右上点序：(x_min,y_min)(x_min,y_max)(x_max,y_max)(x_max,y_min)
        pts = np.float32([[x_min, y_min], [x_min, y_max], [
            x_max, y_max], [x_max, y_min]]).reshape(-1, 1, 2)
        pypts = cal_rect_pts(pts)

        return middle_point, pypts, [x_min, x_max, y_min, y_max, w, h]

    def _get_origin_result_with_two_points(self, pts_sch1, pts_sch2, pts_src1, pts_src2):
        """返回两对有效匹配特征点情形下的识别结果."""
        # 先算出中心点(在self.im_source中的坐标)：
        middle_point = [int((pts_src1[0] + pts_src2[0]) / 2), int((pts_src1[1] + pts_src2[1]) / 2)]
        pypts = []
        # 如果特征点同x轴或同y轴(无论src还是sch中)，均不能计算出目标矩形区域来，此时返回值同good=1情形
        if pts_sch1[0] == pts_sch2[0] or pts_sch1[1] == pts_sch2[1] or pts_src1[0] == pts_src2[0] or pts_src1[1] == \
                pts_src2[1]:
            return None
        # 计算x,y轴的缩放比例：x_scale、y_scale，从middle点扩张出目标区域:(注意整数计算要转成浮点数结果!)
        h, w = self.im_search.shape[:2]
        h_s, w_s = self.im_source.shape[:2]
        x_scale = abs(1.0 * (pts_src2[0] - pts_src1[0]) / (pts_sch2[0] - pts_sch1[0]))
        y_scale = abs(1.0 * (pts_src2[1] - pts_src1[1]) / (pts_sch2[1] - pts_sch1[1]))
        # 得到scale后需要对middle_point进行校正，并非特征点中点，而是映射矩阵的中点。
        sch_middle_point = int((pts_sch1[0] + pts_sch2[0]) / 2), int((pts_sch1[1] + pts_sch2[1]) / 2)
        middle_point[0] = middle_point[0] - int((sch_middle_point[0] - w / 2) * x_scale)
        middle_point[1] = middle_point[1] - int((sch_middle_point[1] - h / 2) * y_scale)
        middle_point[0] = max(middle_point[0], 0)  # 超出左边界取0  (图像左上角坐标为0,0)
        middle_point[0] = min(middle_point[0], w_s - 1)  # 超出右边界取w_s-1
        middle_point[1] = max(middle_point[1], 0)  # 超出上边界取0
        middle_point[1] = min(middle_point[1], h_s - 1)  # 超出下边界取h_s-1

        # 计算出来rectangle角点的顺序：左上角->左下角->右下角->右上角， 注意：暂不考虑图片转动
        # 超出左边界取0, 超出右边界取w_s-1, 超出下边界取0, 超出上边界取h_s-1
        x_min, x_max = int(max(middle_point[0] - (w * x_scale) / 2, 0)), int(
            min(middle_point[0] + (w * x_scale) / 2, w_s - 1))
        y_min, y_max = int(max(middle_point[1] - (h * y_scale) / 2, 0)), int(
            min(middle_point[1] + (h * y_scale) / 2, h_s - 1))
        # 目标矩形的角点按左上、左下、右下、右上的点序：(x_min,y_min)(x_min,y_max)(x_max,y_max)(x_max,y_min)
        pts = np.float32([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]).reshape(-1, 1, 2)
        for npt in pts.astype(int).tolist():
            pypts.append(tuple(npt[0]))

        return middle_point, pypts, [x_min, x_max, y_min, y_max, w, h]

    def _find_homography(self, sch_pts, src_pts):
        """多组特征点对时，求取单向性矩阵."""
        try:
            M, mask = cv2.findHomography(sch_pts, src_pts, cv2.RANSAC, 5.0)
        except Exception:
            import traceback
            traceback.print_exc()
            print("OpenCV error in _find_homography()...")
        else:
            if mask is None:
                print("In _find_homography(), find no transfomation matrix...")
            else:
                return M, mask

    def _target_error_check(self, w_h_range):
        """校验识别结果区域是否符合常理."""
        x_min, x_max, y_min, y_max, w, h = w_h_range
        tar_width, tar_height = x_max - x_min, y_max - y_min
        # 如果src_img中的矩形识别区域的宽和高的像素数＜5，则判定识别失效。认为提取区域待不可能小于5个像素。(截图一般不可能小于5像素)
        if tar_width < 5 or tar_height < 5:
            print("In src_image, Taget area: width or height < 5 pixel.")
        # 如果矩形识别区域的宽和高，与sch_img的宽高差距超过5倍(屏幕像素差不可能有5倍)，认定为识别错误。
        if tar_width < 0.2 * w or tar_width > 5 * w or tar_height < 0.2 * h or tar_height > 5 * h:
            print("Target area is 5 times bigger or 0.2 times smaller than sch_img.")


def cd(n, m=0.8):
    time.sleep(np.random.uniform(n, n+m))
    return None


def cmd(lines):
    if type(lines) == list:
        for line in lines:
            os.system(line)
    else:
        os.system(lines)

    return None


def sim_init():
    lines = [
        # 'rem (0)这个版本是测试适用。',
        'pause',
        '@echo off',
        'cd platform-tools_r28.0.1-windows/platform-tools/',
        'adb start - server',
        'adb devices',
        'pause',
    ]
    cmd(lines)

    return None


def sim_tap(left, right, n=2, pix1=10, pix2=10):
    left = np.random.uniform(left - pix1, left + pix1)
    right = np.random.uniform(right - pix2, right + pix2)
    cmd('adb shell input tap {} {}'.format(left, right))
    cd(n)
    return None


def sim_swipe(left, right, left2, right2, n=2, pix1=10, pix2=10):
    left = np.random.uniform(left - pix1, left + pix1)
    right = np.random.uniform(right - pix2, right + pix2)
    left2 = np.random.uniform(left2 - pix1, left2 + pix1)
    right2 = np.random.uniform(right2 - pix2, right2 + pix2)
    cmd('adb shell input swipe {} {} {} {}'.format(left, right, left2, right2))
    cd(n)
    return None


def sim_input(text, n=1):
    cmd('adb shell input text "{}"'.format(text))
    cd(n)
    return None


def pull_screenshot():
    process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    screenshot = process.stdout.read()
    binary_screenshot = screenshot.replace(b'\r\n', b'\n')

    # os.system('adb shell screencap -p /sdcard/autojump.png')
    # os.system('adb pull /sdcard/autojump.png .')

    return binary_screenshot


def check_screenshot():
    n = 3
    if n < 0:
        print('out')
        sys.exit()

    binary_screenshot = pull_screenshot()

    try:
        # 内存IO
        # Image.open(BytesIO(binary_screenshot)).load()
        img = Image.open(BytesIO(binary_screenshot))
        plt.imshow(img)
        print('aaaaaok')
        plt.show()
    except Exception:
        n -= 1
        check_screenshot()

    return None


def get_img_src_rgb(url):
    img_src_rgb = Image.open(url)
    img_src_rgb = np.array(img_src_rgb)
    return img_src_rgb


def main_screenshot(img_src, img, h):
    # search
    confi = KeypointMatching(im_search=img_src, im_source=img).find_best_result()
    # print(confi)
    if confi:
        # print(confi)
        point_y = confi.get('result')[1]
        # 确认切割起点
        point_start = point_y - 50
        if point_start < 0:
            point_start = 0
        # 确认切割尾点
        point_end = point_start + 100
        if point_end > h:
            point_end = h
        img_temp = img[point_start: point_end, :]
        # plt.imshow(img_temp)
        # plt.show()
        img_url_name = 'img_temp.png'
        plt.imsave(img_url_name, img_temp)
        return get_number(img_url_name)

    return None


def loop_find(img_src, img_src2, img_src3):
    n = 0
    while n < 3:
        # 获取截图
        binary_screenshot = pull_screenshot()
        img = Image.open(BytesIO(binary_screenshot))
        w, h = img.size
        if w > h:
            img = img.rotate(-90, expand=True)
        # 操作截图 2160*1080, rgb, img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = np.array(img)
        try:
            number = main_screenshot(img_src, img, h)
            if number:
                return number
            else:
                number = main_screenshot(img_src3, img, h)
                if number:
                    print('fig_3.png')
                    return number
                else:
                    number = main_screenshot(img_src2, img, h)
                    if number:
                        print('fig_2.png')
                        return number
        except:
            pass
        n += 1
        for i in range(5):
           sim_swipe(536, 1897, 500, 1004, n=np.random.uniform(0.5, 0.8), pix1=100, pix2=100)

    return 0


def get_token():
    guan_ak = ''  # todo
    guan_sk = ''  # todo
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'.format(
        guan_ak, guan_sk)
    response = requests.get(host)
    if response:
        print(response.json())

    return None


def get_number(img_url_name):
    # get_token()
    guan_token = ''  # todo

    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/numbers"
    # 二进制方式打开图片文件
    f = open(img_url_name, 'rb')
    img = base64.b64encode(f.read())
    cd(1)

    params = {"image": img}
    access_token = guan_token
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        # print(response.json())
        response = response.json()
        number = response.get('words_result')[0].get('words')
        number = np.int(number)
        print(number)
        return number

    return None


def sim_fun_xxbxzhou(count_top=50):
    count = 0
    while count < count_top:
        count += 1
        print('这是第 {} / {}'.format(count, count_top))

        # 随机点击进入
        choose_rank = np.random.randint(0, 3)
        if choose_rank == 0:
            sim_tap(529, 1099, n=2, pix1=300)
        elif choose_rank == 1:
            sim_tap(590, 1376, n=2, pix1=300)
        else:
            sim_tap(572, 1624, n=2, pix1=300)

        # 缓冲加载页面
        cd(2)

        # 开始模拟阅读
        choose_count = np.random.randint(4, 9)
        for i in range(choose_count):
            sim_swipe(536, 1897, 500, 1404, n=np.random.uniform(2, 2.5), pix1=300, pix2=200)

        # 返回操作
        sim_swipe(1, 1207, 398, 1208, n=2, pix1=2)

    return None


def sim_fun_yueguang(count_top=50):
    count = 0
    img_src = get_img_src_rgb('fig_1.png')
    img_src2 = get_img_src_rgb('fig_2.png')
    img_src3 = get_img_src_rgb('fig_3.png')
    while count < count_top:
        count += 1
        print('这是第 {} / {}'.format(count, count_top))

        # 随机点击进入
        choose_rank = np.random.randint(0, 3)
        if choose_rank == 0:
            sim_tap(529, 1099, n=2, pix1=300)
        elif choose_rank == 1:
            sim_tap(590, 1376, n=2, pix1=300)
        else:
            sim_tap(572, 1624, n=2, pix1=300)

        # 缓冲加载页面
        cd(2)

        # 开始模拟阅读
        choose_count = 16
        for i in range(choose_count):
            sim_swipe(536, 1897, 500, 1004, n=np.random.uniform(0.5, 0.8), pix1=200, pix2=100)

        # 判断底部及反馈数据
        number = loop_find(img_src, img_src2, img_src3)

        # 返回操作
        sim_swipe(1, 1207, 398, 1208, n=3, pix1=2)

        # 缓冲加载2
        # cd(10)
        print(number)
        sim_tap(555, 1450, n=1, pix1=200, pix2=10)
        sim_input(number)
        sim_tap(1014, 1452, n=1, pix1=5, pix2=5)
        sim_tap(1014, 1452, n=1, pix1=5, pix2=5)
        sim_tap(719, 1652, n=1, pix1=50, pix2=5)
        cd(5)

    return None


def main(mode=1, count_top=50):
    # 初始化，准备工作
    sim_init()

    # 选择模式, 1='xxbxzhou', 2='yueguang'
    # mode = 1

    # 开始测试
    if mode == 1:
        sim_fun_xxbxzhou()
    elif mode == 2:
        sim_fun_yueguang()
    elif mode == 3:
        # loop_find()
        pass

    return None


if __name__ == '__main__':
    try:
        main(mode=2, count_top=50)
        os.system('adb kill-server')
        print('bye')
    except KeyboardInterrupt:
        os.system('adb kill-server')
        print('bye')
        exit(0)
