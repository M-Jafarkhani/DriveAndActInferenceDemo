import torch
import numpy as np
from random import randint, seed
import math
import os
import PIL
import numbers
import cv2



def add_text_to_image(img, text, font = cv2.FONT_ITALIC, bottomLeftCornerOfText = (10,20), fontScale = 0.4,fontColor = (200,200,200),lineType = 1):

    color = np.random.randint(0, 255, size=(3, ))
    color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
    img_with_text = cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    return img_with_text




def extractFilesFromDirWhichMatchList(folder, string_match_filenames, string_not_match_filenames=[], full_path=False):
    result_files = []
    print (folder)
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            ok = True

            if full_path:

                check = os.path.join(dirpath, filename)
                # Only include if matches
                for s in string_match_filenames:
                    if (check.find(s) == -1):  # if not found
                        ok = False

                # Exclude if matches
                for s in string_not_match_filenames:
                    if (check.find(s) != -1):  # if found
                        ok = False

            else:
                # Only include if matches
                for s in string_match_filenames:
                    if (filename.find(s) == -1):  # if not found
                        ok = False

                # Exclude if matches
                for s in string_not_match_filenames:
                    if (filename.find(s) != -1):  # if found
                        ok = False

            if ok:
                file_path = os.path.join(dirpath, filename)
                result_files.append(file_path)

    return result_files


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return np.asarray(scaled)


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow

def normalize_color_input_zero_center_unit_range(frames, max_val = 255.0):

    frames = (frames / max_val) * 2 - 1
    return(frames)


class normalizeColorInputZeroCenterUnitRange(object):
    def __init__(self, max_val = 255.0):

        self.max_val = max_val


    def __call__(self, input_tensor):
        result = normalize_color_input_zero_center_unit_range(input_tensor, max_val = self.max_val)

        return result


def random_select(frames, n, seed = None):
    """
    Takes multiple frames as ndarray with shape
    (frame id, height, width, channels) and selects
    randomly n-frames. If n is greater than the number
    of overall frames, placeholder frames (zeros) will
    be added.

    frames: numpy
        all frames (e.g. video) with shape
        (frame id, height, width, channels)
    n: int
        number of desired randomly picked frames

    Returns
    -------
    Numpy: frames
        randomly picked frames with shape
        (frame id, height, width, channels)
    """
    #print("Frames shape:{}".format(np.shape(frames)))
    if seed is not None:
        seed(seed)

    number_of_frames = np.shape(frames)[0]
    if number_of_frames < n:
        # Add all frames
        selected_frames = []
        for i in range(number_of_frames):
            frame = frames[i, :, :, :]
            selected_frames.append(frame)

        # Fill up with 'placeholder' images
        frame = np.zeros(frames[0, :, :, :].shape)
        for i in range(n - number_of_frames):
            selected_frames.append(frame)

        return np.array(selected_frames)

    # Selected random frame ids
    frame_ids = set([])
    while len(frame_ids) < n:
        frame_ids.add(randint(0, number_of_frames - 1))

    # Sort the frame ids
    frame_ids = sorted(frame_ids)

    # Select frames
    selected_frames = []
    for id in frame_ids:
        #print (np.shape(frames))

        frame = frames[id, :, :, :]
        selected_frames.append(frame)

    return np.array(selected_frames)


def center_crop(frames, height, width, pad_zeros_if_too_small = True):
    """
    Takes multiple frames as ndarray with shape
    (frame id, height, width, channels) and crops all
    frames centered to desired width and height.

    frames: numpy
        all frames (e.g. video) with shape
        (frame id, height, width, channels)
    height: int
        height of the resulting crop
    width: int
        width of the resulting crop

    Returns
    -------
    Numpy: frames
        centered cropped frames with shape
        (frame id, height, width, channels)
    """

    frame_height = np.shape(frames)[1]
    frame_width = np.shape(frames)[2]

    t = np.shape(frames)[0]
    channels = np.shape(frames)[3]

    if pad_zeros_if_too_small and (height > frame_height or width > frame_width):
        # desired width
        frames_new = np.zeros((t, max(frame_height, height), max(frame_width, width), channels))
        # fill with the old data
        frames_new[0:t, 0:frame_height, 0:frame_width, 0:channels] = frames
        frames = frames_new
        frame_height = np.shape(frames)[1]
        frame_width = np.shape(frames)[2]


    origin_x = (frame_width - width) / 2
    origin_y = (frame_height - height) / 2

    # Floor origin (miss matching input sizes)
    # E.g. input width of 171 and crop width 112
    # would result in a float.
    origin_x = math.floor(origin_x)
    origin_y = math.floor(origin_y)

    return frames[:,
                  origin_y: origin_y + height,
                  origin_x: origin_x + width,
                  :]



class Rescale(object):
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):

        resized = resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class CenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, input_tensor):

        result = center_crop(input_tensor, self.height, self.width)

        return result




class RandomSelect(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, input_tensor):

        result = random_select(input_tensor, self.n)

        return result



class ToTensor(object):
    def __call__(self, input_tensor):


        # Swap color channels axis because
        # numpy frames: Frames ID x Height x Width x Channels
        # torch frames: Channels x Frame ID x Height x Width

        result = input_tensor.transpose(3, 0, 1, 2)
        result = np.float32(result)

        return torch.from_numpy(result)


