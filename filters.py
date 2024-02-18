import os

import numpy as np

import scipy

from common import read_img, save_img

import matplotlib.pyplot as plt


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output = []
    for i in range(image.shape[0] // patch_size[0]):
        for j in range(image.shape[1] // patch_size[1]):
            output.append(image[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]])
    for i in range(len(output)):
        output[i] = (output[i] - np.mean(output[i])) / np.std(output[i])
    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """

    zero_padding_x = int(np.ceil((kernel.shape[0] - 1) / 2))
    zero_padding_y = int(np.ceil((kernel.shape[1] - 1) / 2))
    flipped_kernel = np.fliplr(np.flipud(kernel))
    H, W = image.shape
    output = np.zeros((H, W))
    image_padding = np.pad(image, ((zero_padding_x, zero_padding_x), (zero_padding_y, zero_padding_y)), "constant")
    # print(image_padding.shape)
    for i in range(image_padding.shape[0] - flipped_kernel.shape[0]):
        for j in range(image_padding.shape[1] - flipped_kernel.shape[1]):
            output[i, j] = np.sum(flipped_kernel * image_padding[i:i + flipped_kernel.shape[0], j:j + flipped_kernel.shape[1]])
    return output

    # return scipy.ndimage.convolve(image, kernel, mode='constant')


def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([[-1, 0, 1]])  # 1 x 3
    ky = np.array([[-1, 0, 1]]).T  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(np.square(Ix) + np.square(Iy))

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    Gx, Gy, grad_magnitude = None, None, None

    return Gx, Gy, grad_magnitude


def bilateral_filter(image, window_size, sigma_d, sigma_r):
    """
    Return filtered image using a bilateral filter

    Input-  image: H x W
            window_size: (h, w)
            sigma_d: sigma for the spatial kernel
            sigma_r: sigma for the range kernel
    Output- output: filtered image
    """
    # TODO: complete the bilateral filtering, assuming spatial and range kernels are gaussian
    h, w = image.shape[:2]
    output = np.zeros((h, w))
    padding_h = int(np.ceil((window_size[0] - 1) / 2))
    padding_w = int(np.ceil((window_size[1] - 1) / 2))
    image_padded = np.pad(image, ((padding_h, padding_h), (padding_w, padding_w)), mode='constant')
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):

            output[i, j] =

    return output


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = patches[2]
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    ax = np.linspace(-2 / 2., 2 / 2., 3)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(0.572))
    kernel = np.outer(gauss, gauss)
    kernel_gaussian = kernel / np.sum(kernel)

    filtered_gaussian = convolve(img, kernel_gaussian)
    # print("max: ", np.max(filtered_gaussian), ", min: ", np.min(filtered_gaussian))
    # plt.imsave('original.png', filtered_gaussian, cmap="gray", vmin = 0, vmax = 255)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")
    """
    ax = np.linspace(-2 / 2., 2 / 2., 3)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(2))
    kernel = np.outer(gauss, gauss)
    kernel_gaussian = kernel

    filtered_gaussian = convolve(img, kernel_gaussian)
    print("max: ", np.max(filtered_gaussian), ", min: ", np.min(filtered_gaussian))
    plt.imsave('modified.png', filtered_gaussian, cmap="gray", vmin=0, vmax=255)
    """
    # (d), (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")

    # (h) complete biliateral_filter()
    if not os.path.exists("./bilateral"):
        os.makedirs("./bilateral")

    image_bilataral_filtered = bilateral_filter(img, (5, 5), 3, 75)
    save_img(image_bilataral_filtered, "./bilateral/bilateral_output.png")

    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = None
    filtered_LoG2 = None
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
