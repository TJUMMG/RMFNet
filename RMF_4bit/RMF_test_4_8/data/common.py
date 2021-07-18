import random
import numpy as np
import skimage.color as sc

import torch



def get_patch_test(img_in, img_tar, scale, patch_size_factor=8, multi_scale=False):
    ih, iw = img_in.shape[:2]
    p = scale if multi_scale else 1

    ix = (iw // patch_size_factor) * patch_size_factor
    iy = (ih // patch_size_factor) * patch_size_factor
    tx, ty = p * ix, p * iy

    img_in = img_in[0:iy, 0:ix, :]
    img_tar = img_tar[0:ty, 0:tx, :]
    return img_in, img_tar


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        type = img.dtype
        if type == 'uint8':
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose / 1.0).float()
            tensor.mul_((2**8))
        else:
            print('Please input correct data！')
        return tensor

    return [_np2Tensor(_l) for _l in l]



def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]
