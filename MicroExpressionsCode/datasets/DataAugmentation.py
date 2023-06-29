import os
import cv2
import numpy as np
import random
"""
use get functions to get random arguments and make them the same for all frames in each video 


            args_rotated = aug.get_rotate_args()
            # args_cutout = aug.get_cutout_args()
            args_colorjitter = aug.get_colorjitter_args()
            args_noisy = aug.get_noisy_args()
            args_filters = aug.get_filters_args()

            video_rotated = cv2.VideoWriter(output_path + "_rotated.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            # video_cutout = cv2.VideoWriter(output_path + "_cutout.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            video_colorjitter_brightness = cv2.VideoWriter(output_path + "_colorjitter_brightness.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            video_colorjitter_saturation = cv2.VideoWriter(output_path + "_colorjitter_saturation.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            video_colorjitter_contrast = cv2.VideoWriter(output_path + "_colorjitter_contrast.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            video_noisy = cv2.VideoWriter(output_path + "_noisy.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            video_filter_blur = cv2.VideoWriter(output_path + "_blur_filter.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            video_filter_gaussian = cv2.VideoWriter(output_path + "_gaussian_filter.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))
            video_filter_median = cv2.VideoWriter(output_path + "_median_filter.mp4", fourcc, 200, (REDUCED_WIDTH, REDUCED_HEIGHT))



                img_rotated = aug.rotate(img, **args_rotated)
                # img_cutout = aug.rotate(img, **args_cutout)
                img_colorjitter_brightness = aug.colorjitter(img, cj_type="b", **args_colorjitter)
                img_colorjitter_saturation = aug.colorjitter(img, cj_type="s", **args_colorjitter)
                img_colorjitter_contrast = aug.colorjitter(img, cj_type="c", **args_colorjitter)
                img_noisy = aug.noisy(img, noise_type="sp", **args_noisy)
                img_filter_blur = aug.filters(img, f_type="blur", **args_filters)
                img_filter_gaussian = aug.filters(img, f_type="gaussian", **args_filters)
                img_filter_median = aug.filters(img, f_type="median", **args_filters)


                video_rotated.write(img_rotated)  # write to video
                # video_cutout.write(img_cutout)
                video_colorjitter_brightness.write(img_colorjitter_brightness)
                video_colorjitter_saturation.write(img_colorjitter_saturation)
                video_colorjitter_contrast.write(img_colorjitter_contrast)
                video_noisy.write(img_noisy)
                video_filter_blur.write(img_filter_blur)
                video_filter_gaussian.write(img_filter_gaussian)
                video_filter_median.write(img_filter_median)
                
                
                
            video_rotated.release()
            # video_cutout.release()
            video_colorjitter_brightness.release()
            video_colorjitter_saturation.release()
            video_colorjitter_contrast.release()
            video_noisy.release()
            video_filter_blur.release()
            video_filter_gaussian.release()
            video_filter_median.release()

"""


def zoom(img, dim, zoom_factor):
    dim_zoomed = (dim[0]/zoom_factor, dim[1]/zoom_factor)
    crop_img = center_crop(img, dim_zoomed)
    scale_img = scale_image(crop_img, dim)
    return scale_img


def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def scale_image(img, dim):
    """Returns resize image by scale factor.
    This helps to retain resolution ratio while resizing.
    Args:
    img: image to be scaled
    factor: scale factor to resize
    """
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)



def get_rotate_args():
    args = {}
    args["angle"] = np.random.choice(np.array([-60, -45, -30, 30, 45, 60]))
    return args

def rotate(img, angle, dim, zoom_factor=1):
    image = img.copy()
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    image = zoom(image, dim, zoom_factor)
    return image


def get_cutout_args(gt_boxes, amount=0.5):
    args = {}
    args["ran_select"] = random.sample(gt_boxes, round(amount*len(gt_boxes)))
    return args


def cutout(img, ran_select):
    '''
    ### Cutout ###
    img: image
    gt_boxes: format [[obj x1 y1 x2 y2],...]
    amount: num of masks / num of objects
    '''
    out = img.copy()

    for box in ran_select:
        x1 = int(box[1])
        y1 = int(box[2])
        x2 = int(box[3])
        y2 = int(box[4])
        mask_w = int((x2 - x1)*0.5)
        mask_h = int((y2 - y1)*0.5)
        mask_x1 = random.randint(x1, x2 - mask_w)
        mask_y1 = random.randint(y1, y2 - mask_h)
        mask_x2 = mask_x1 + mask_w
        mask_y2 = mask_y1 + mask_h
        cv2.rectangle(out, (mask_x1, mask_y1), (mask_x2, mask_y2), (0, 0, 0), thickness=-1)
    return out


def get_colorjitter_args():
    args = {}
    # args["value"] = random.randint(-50, 50)
    args["value"] = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
    return args


def colorjitter(img, cj_type="b", value=0):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: contrast}
    '''
    if cj_type == "b":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        brightness = 10
        contrast = value
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img

def get_noisy_args():
    args = {}
    return args

def noisy(img, noise_type="sp"):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    # if noise_type == "gauss":
    #     image = img.copy()
    #     mean = 0
    #     st = 0.7
    #     gauss = np.random.normal(mean, st, image.shape)
    #     gauss = gauss.astype('uint8')
    #     image = cv2.add(image, gauss)
    #     return image

    if noise_type == "sp":
        image = img.copy()
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image


def get_filters_args():
    args = {}
    args["fsize"] = np.random.choice(np.array([7, 9, 11]))
    return args


def filters(img, f_type="blur", fsize=9):
    '''
    ### Filtering ###
    img: image
    f_type: {blur: blur, gaussian: gaussian, median: median}

    '''
    if f_type == "blur":
        image = img.copy()
        return cv2.blur(image, (fsize, fsize))

    elif f_type == "gaussian":
        image = img.copy()
        return cv2.GaussianBlur(image, (fsize, fsize), 0)

    elif f_type == "median":
        image = img.copy()
        return cv2.medianBlur(image, fsize)