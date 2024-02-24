"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
import random

def randomcrop(image, mask, u=0.5):
    crop_rate = np.random.uniform(0.7,0.9)
    height = np.int32(image.shape[0]*crop_rate)
    width = height
    if np.random.random() < u:
        h, w, c = image.shape
        y = np.random.randint(0, h-height+1)
        x = np.random.randint(0, w-width+1)
        image = image[y:y+height,x:x+width,:]
        image = cv2.resize(image,(h, w), interpolation = cv2.INTER_CUBIC)
        mask = mask[y:y+height,x:x+width]
        mask = cv2.resize(mask,(h, w), interpolation = cv2.INTER_CUBIC)
    return image, mask


def randomHueSaturationValue(
    image,
    hue_shift_limit=(-30, 30),
    sat_shift_limit=(-5, 5),
    val_shift_limit=(-15, 15),
    u=0.5,
):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(
            hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(
    image,
    mask,
    shift_limit=(-0.1, 0.1),
    scale_limit=(-0.1, 0.1),
    aspect_limit=(-0.1, 0.1),
    rotate_limit=(-0, 0),
    borderMode=cv2.BORDER_CONSTANT,
    u=0.5,
):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy]
        )

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpPerspective(
            mask,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask

def randomRotate180(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(np.rot90(image))
        mask = np.rot90(np.rot90(mask))
    return image, mask

def randomRotate270(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(np.rot90(np.rot90(image)))
        mask = np.rot90(np.rot90(np.rot90(mask)))
    return image, mask

def randomCropResized(image, mask, u=0.5):
    ori_image = image
    if np.random.random() < u:
        gps_image = image[:, :, 3]
        image = image[:, :, :3]
        h, w, _ = image.shape
        resize_size = (h, w)
        i, j, h, w = transforms.RandomResizedCrop.get_params(Image.fromarray(image), scale=(0.7, 0.9), ratio=(0.7, 1.3))
        image = tf.resized_crop(Image.fromarray(image), i, j, h, w, resize_size)
        gps_image = tf.resized_crop(Image.fromarray(gps_image), i, j, h, w, resize_size)
        mask = tf.resized_crop(Image.fromarray(mask), i, j, h, w, resize_size)
        image = np.asarray(image)
        mask = np.asarray(mask)
        gps_image = np.asarray(gps_image)
        gps_image = gps_image[:, :, None]
        image = np.concatenate([image, gps_image], 2)
    return image, mask

def randomCrop(image, mask, u=0.5):
    gps_image = image[:, :, 3]
    image = image[:, :, :3]
    h, w, _ = image.shape

    trans_random_crop = transforms.RandomCrop(512)
    seed = torch.random.seed()
    torch.random.manual_seed(seed)
    image = trans_random_crop(Image.fromarray(image))
    torch.random.manual_seed(seed)
    gps_image = trans_random_crop(Image.fromarray(gps_image))
    torch.random.manual_seed(seed)
    mask = trans_random_crop(Image.fromarray(mask))

    image = np.asarray(image)
    mask = np.asarray(mask)
    gps_image = np.asarray(gps_image)
    gps_image = gps_image[:, :, None]
    image = np.concatenate([image, gps_image], 2)
    return image, mask

def randomMask(image, mask, u=0.5):

    if np.random.random() < u:
        gps_image = image[:, :, 3]
        image = image[:, :, :3]
        ori_image = image.copy()
        h, w, _ = image.shape
        iter = 5
        for i in range(iter):
            size_h = random.randint(int(h/12), int(h/4))
            size_w = random.randint(int(w / 12), int(w / 4))
            start_x = random.randint(0, w - size_w - 1)
            start_y = random.randint(0, h - size_h - 1)
            image[start_y:start_y+size_h, start_x:start_x+size_w, :] = 0
        image = np.asarray(image)
        mask = np.asarray(mask)
        gps_image = np.asarray(gps_image)
        gps_image = gps_image[:, :, None]
        image = np.concatenate([image, gps_image], 2)
    return image, mask

def randomExchange(image, mask, u=1):
    gps_image = image[:, :, 3]
    image = image[:, :, :3]
    if np.random.random() < u:
        h, w, _ = image.shape
        gps_image = gps_image[:, :, None].repeat(repeats=3, axis=2)
        def exchange_a_b(img1, img2):
            block_min, block_max= 4, 8
            block_num = random.randint(block_min, block_max)
            block_width = int(w / block_num)
            block_height = int(h / block_num)
            for i in range(block_num):
                for j in range(block_num):
                    if i % 2 == 0:
                        start_x = i * block_width
                        start_y = j * block_height
                        end_x = start_x + block_width
                        end_y = start_y + block_height
                        if w - end_x < block_width:
                            end_x = w
                        if h - start_y < block_height:
                            end_y = h
                        image_crop = img1[start_y:end_y, start_x:end_x, :].copy()
                        img1[start_y:end_y, start_x:end_x, :] = img2[start_y:end_y, start_x:end_x, :]
                        img2[start_y:end_y, start_x:end_x, :] = image_crop
                    if j % 2 == 0:
                        start_x = i * block_width
                        start_y = j * block_height
                        end_x = start_x + block_width
                        end_y = start_y + block_height
                        if w - end_x < block_width:
                            end_x = w
                        if h - start_y < block_height:
                            end_y = h
                        image_crop = img1[start_y:end_y, start_x:end_x, :].copy()
                        img1[start_y:end_y, start_x:end_x, :] = img2[start_y:end_y, start_x:end_x, :]
                        img2[start_y:end_y, start_x:end_x, :] = image_crop
            return img1, img2

        image2gps, gps2image = exchange_a_b(image.copy(), gps_image.copy())
        img_list = [image, gps_image, image2gps, gps2image]
        idx_1 = random.randint(0, 3)
        idx_2 = random.randint(0, 3)
        if (idx_1 == 0 and idx_2 == 1) or (idx_1 == 1 and idx_2 == 0):
            idx_2 = random.randint(2, 3)
        image = img_list[idx_1]
        gps_image = img_list[idx_2]

        image = np.concatenate([image, gps_image], 2)
        return image, mask
    gps_image = gps_image[:, :, None].repeat(repeats=3, axis=2)
    image = np.concatenate([image, gps_image], 2)
    return image, mask