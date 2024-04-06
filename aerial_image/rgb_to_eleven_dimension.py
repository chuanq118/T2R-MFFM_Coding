import os

import cv2



def calculate_grayscale(rgb_image, a, b, c):
    """
    使用公式计算灰度图像
    """
    return a * rgb_image[..., 0] + b * rgb_image[..., 1] + c * rgb_image[..., 2]


def split_rgb_image(rgb_image_path):
    """
    由 RGB 三个维度变为十一个维度,保存在当前目录下z`
    :param rgb_image_path: 读取的原始 RGB 图像路径
    :return:
    """
    # 读取 RGB 图像
    rgb_image = cv2.imread(rgb_image_path)

    # 计算每个灰度图像
    gray_image_1 = calculate_grayscale(rgb_image, 0.299, 0.587, 0.114)
    gray_image_2 = calculate_grayscale(rgb_image, 0.2126, 0.7152, 0.0722)
    gray_image_3 = calculate_grayscale(rgb_image, 0.333, 0.333, 0.333)
    gray_image_4 = calculate_grayscale(rgb_image, 0.5, 0.25, 0.25)
    gray_image_5 = calculate_grayscale(rgb_image, 0.25, 0.5, 0.25)
    gray_image_6 = calculate_grayscale(rgb_image, 0.25, 0.25, 0.5)
    gray_image_7 = calculate_grayscale(rgb_image, 0.375, 0.375, 0.25)
    gray_image_8 = calculate_grayscale(rgb_image, 0.25, 0.375, 0.375)

    # 获取路径中的文件名以便保存不同的灰度图像
    filename = os.path.basename(rgb_image_path).split(".")[0]

    # 分离通道
    b_channel, g_channel, r_channel = cv2.split(rgb_image)

    # 保存通道图像到当前目录
    cv2.imwrite(filename + "_R.png", r_channel)
    cv2.imwrite(filename + "_G.png", g_channel)
    cv2.imwrite(filename + "_B.png", b_channel)

    # 保存灰度图像
    cv2.imwrite(filename + "_T4.png", gray_image_1)
    cv2.imwrite(filename + "_T5.png", gray_image_2)
    cv2.imwrite(filename + "_T6.png", gray_image_3)
    cv2.imwrite(filename + "_T7.png", gray_image_4)
    cv2.imwrite(filename + "_T8.png", gray_image_5)
    cv2.imwrite(filename + "_T9.png", gray_image_6)
    cv2.imwrite(filename + "_T10.png", gray_image_7)
    cv2.imwrite(filename + "_T11.png", gray_image_8)

    return


split_rgb_image(r'./3_116_sat.png')
