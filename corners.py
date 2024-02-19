import os
from typing import Callable, Any

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img

import matplotlib.pyplot as plt


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    u = -u
    v = -v

    non: Callable[[int], int] = lambda s: s if s < 0 else None
    mom: Callable[[int], int] = lambda s: max(0, s)

    image_offset = np.zeros_like(image)
    image_offset[mom(u): non(u), mom(v): non(v)] = image[mom(-u):non(-u), mom(-v):non(-v)]
    square_difference = np.square(image_offset - image)

    kernel = np.ones(window_size, np.uint8)
    output = scipy.ndimage.convolve(square_difference, kernel, mode='constant')
    return output


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    kx = np.array([[-0.5, 0, 0.5]])
    ky = np.array([[-0.5, 0, 0.5]]).T
    Ix = scipy.ndimage.convolve(image, kx, mode='constant')
    Iy = scipy.ndimage.convolve(image, ky, mode='constant')
    """
    padding_x = int(np.ceil((window_size[0] - 1) / 2))
    padding_y = int(np.ceil((window_size[1] - 1) / 2))
    h, w = Ix.shape[:2]
    Ix_padded = np.pad(Ix, ((padding_x, padding_x), (padding_y, padding_y)), mode='constant')
    Iy_padded = np.pad(Iy, ((padding_x, padding_x), (padding_y, padding_y)), mode='constant')
    """
    # Ixx = np.zeros(shape=(h, w))
    # Iyy = np.zeros(shape=(h, w))
    # Ixy = np.zeros(shape=(h, w))
    kernel_sum = np.ones(window_size)
    """
    for i in range(h):
        for j in range(w):
            for x in range(window_size[0]):
                for y in range(window_size[1]):
                    Ixx[i, j] += np.square(Ix_padded[i+x, j+y])
                    Iyy[i, j] += np.square(Iy_padded[i+x, j+y])
                    Ixy[i, j] += Ix_padded[i+x, j+y] * Iy_padded[i+x, j+y]
    """
    Ixx = scipy.ndimage.convolve(np.square(Ix), kernel_sum, mode='constant')
    Iyy = scipy.ndimage.convolve(np.square(Iy), kernel_sum, mode='constant')
    Ixy = scipy.ndimage.convolve(Ix*Iy, kernel_sum, mode='constant')


    # For each image location, construct the structure tensor and calculate
    # the Harris response
    response = (Ixx*Iyy - np.square(Ixy)) - 0.05*np.square(Ixx+Iyy)

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 6: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    u, v, W = -5, 0, (5, 5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    # (c): No Code

    # -- TODO Task 7: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")
    plt.imshow(harris_corners, cmap='hot', aspect='auto')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
