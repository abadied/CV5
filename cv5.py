
import numpy as np
import cv2
import scipy
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import math
import scipy.signal
import scipy.io as sio
from math import pow
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ================
# Question 1
# ================

def bilinear_interpolation(image, x, y):
    frac_x, z_x = math.modf(x)
    frac_y, z_y = math.modf(y)
    x_ceiling, x_floor = math.ceil(x), math.floor(x)
    y_ceiling, y_floor = math.ceil(y), math.floor(y)
    interpolated_value = (1 - frac_x) * (1 - frac_y) * image[y_floor][x_floor] + \
                         (1 - frac_x) * frac_y * image[y_ceiling][x_floor] + \
                         frac_x * (1 - frac_y) * image[y_floor][x_ceiling] + \
                         frac_x * frac_y * image[y_ceiling][x_ceiling]
    return interpolated_value


# ================
# Question 2
# ================

def Q2(image, u, v):
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    map_y, map_x = (yy - v).astype(np.float32), (xx - u).astype(np.float32)
    I_warpped = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            m_x = map_x[i][j]
            m_y = map_y[i][j]
            if 0 <= m_x <= width - 1 and 0 <= m_y <= height - 1:
                I_warpped[i][j] = bilinear_interpolation(image, map_x[i][j],  map_y[i][j])
            else:
                I_warpped[i][j] = 0

    I_warpped_remap = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)

    return I_warpped, I_warpped_remap


# ================
# Question 3
# ================

def Q3(image):
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    u = 0.5 * xx + 0.2 * yy
    v = 0.5 * yy + 8
    map_x = np.ones((height, width)) * -1
    map_y = np.ones((height, width)) * -1
    I_warpped = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            u_ij = int(u[i][j])
            v_ij = int(v[i][j])
            map_x[v_ij][u_ij] = j
            map_y[v_ij][u_ij] = i

    for i in range(height):
        for j in range(width):
            m_x = map_x[i][j]
            m_y = map_y[i][j]
            if 0 <= m_x <= width - 1 and 0 <= m_y <= height - 1:
                I_warpped[i][j] = bilinear_interpolation(image, map_x[i][j],  map_y[i][j])
            else:
                I_warpped[i][j] = 0
    return I_warpped

# ================
# Question 4
# ================


def spatial_derivative(im_1, im_2):
    return im_2 - im_1

# ================
# Question 5
# ================

def get_derivatives():
    pass

def create_A():
    pass


def create_b(img_x, img_y, img_t):
    n_row = img_t.shape[0]
    n_col = img_t.shape[1]
    b = np.zeros(2 * n_row * n_col, dtype=np.float32)
    for i in range(n_row):
        for j in range(n_col):
            curr_row = (i * n_row) + j + 1
            b[2 * curr_row - 1] = - img_x[i, j] * img_t[i, j]
            b[2 * curr_row] = - img_y[i, j] * img_t[i, j]
    return b[1:]


def q5_all_images():
    pass


def q5_image():
    pass


def main():
    mdict = sio.loadmat("hw4_data/hw4_data/imgs_for_optical_flow.mat")
    image = mdict["img1"]
    u = mdict["u"]
    v = mdict["v"]

    # UV Flow
    image_w_bilinear_uv, image_w_remap_uv = Q2(image, u, v)

    # Affine flow
    flow_matrix = np.float32([[0.5, 0.2, 0], [0, 0.5, 8]])
    image_w_bilinear_affine = Q3(image)
    image_w_warp_affine = cv2.warpAffine(image, flow_matrix, image.shape)

    # Show images
    fig = plt.figure(5, figsize=(15, 10))
    ax = fig.add_subplot(1, 5, 1)
    ax.title.set_text('Original')
    ax.imshow(image, cmap='gray')

    ax = fig.add_subplot(152)
    ax.set_title('bilinear uv')
    ax.imshow(image_w_bilinear_uv, cmap='gray')

    ax = fig.add_subplot(153)
    ax.set_title('remap uv')
    ax.imshow(image_w_remap_uv, cmap='gray')

    ax = fig.add_subplot(154)
    ax.set_title('bilinear affine')
    ax.imshow(image_w_bilinear_affine, cmap='gray')

    ax = fig.add_subplot(155)
    ax.set_title('warpAffine')
    ax.imshow(image_w_warp_affine, cmap='gray')
    plt.show()

    # Question 4
    fig = plt.figure(5, figsize=(15, 10))
    ax = fig.add_subplot(1, 5, 1)
    ax.title.set_text('I2 - I1')
    ax.imshow(spatial_derivative(mdict['img1'], mdict['img2']), cmap='gray')

    ax = fig.add_subplot(152)
    ax.set_title('I3 - I1')
    ax.imshow(spatial_derivative(mdict['img1'], mdict['img3']), cmap='gray')

    ax = fig.add_subplot(153)
    ax.set_title('I4 - I1')
    ax.imshow(spatial_derivative(mdict['img1'], mdict['img4']), cmap='gray')

    ax = fig.add_subplot(154)
    ax.set_title('I5 - I1')
    ax.imshow(spatial_derivative(mdict['img1'], mdict['img5']), cmap='gray')

    ax = fig.add_subplot(155)
    ax.set_title('I6 - I1')
    ax.imshow(spatial_derivative(mdict['img1'], mdict['img6']), cmap='gray')
    plt.show()

    print('Done')


if __name__ == '__main__':
    main()
