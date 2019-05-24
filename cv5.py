
import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
import math
import scipy.signal
import scipy.io as sio
from cv2 import GaussianBlur

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
def warp_image(I, U, V):
    height, width = I.shape
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    yy, xx = np.mgrid[:height, :width]
    map_x = xx - U
    map_y = yy - V
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    I_warpped_remap = cv2.remap(I, map_x, map_y, cv2.INTER_CUBIC)
    return I_warpped_remap

def get_derivatives(img):
    I = GaussianBlur(src=img, ksize=(11, 11), sigmaX=7)
    h_x1, h_x2 = cv2.getDerivKernels(1, 0, 3, normalize=True)
    h_y1, h_y2 = cv2.getDerivKernels(0, 1, 3, normalize=True)
    img_x = cv2.sepFilter2D(I, -1, h_x1, h_x2)
    img_y = cv2.sepFilter2D(I, -1, h_y1, h_y2)
    return img_x, img_y


def get_A_row(lamb, location, i_s, i, j, N_rows, N_cols, factor, img_x, img_y):
    row_u_vals = np.zeros((2 * (N_rows * N_cols) + 1))
    row_v_vals = np.zeros((2 * (N_rows * N_cols) + 1))

    row_u_vals[2 * i_s - 1] = img_x[i][j] ** 2 + factor * lamb
    row_u_vals[2 * i_s] = img_x[i][j] * img_y[i][j]
    row_v_vals[2 * i_s - 1] = img_x[i][i] * img_y[i][j]
    row_v_vals[2 * i_s] = img_y[i][j] ** 2 + factor * lamb
    if location != 'upper' and location != 'up-left' and location != 'up-right':
        row_u_vals[2 * i_s - 2 * N_cols - 1] = -2 * lamb
        row_v_vals[2 * i_s - 2 * N_cols] = -2 * lamb
    if location != 'left' and location != 'up-left' and location != 'down-left':
        row_u_vals[2 * i_s - 3] = -2 * lamb
        row_v_vals[2 * i_s - 2] = -2 * lamb
    if location != 'right' and location != 'up-right' and location != 'down-right':
        row_u_vals[2 * i_s + 1] = -2 * lamb
        row_v_vals[2 * i_s + 2] = -2 * lamb
    if location != 'bottom' and location != 'down-right' and location != 'down-left':
        row_u_vals[2 * i_s + 2 * N_cols - 1] = -2 * lamb
        row_v_vals[2 * i_s + 2 * N_cols] = -2 * lamb
    return row_u_vals, row_v_vals


def create_A(im_x, im_y, lamb):
    n_rows, n_cols = im_x.shape
    N = n_rows * n_cols
    A = np.zeros((2 * N + 1, 2 * N + 1))

    # run on the image, for each pixel s create 2 rows in A:
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            i_s = (i * n_cols) + (j + 1)  # convert (i,j) to i_s in "1-based" index
            row_u = 2 * i_s - 1
            row_v = 2 * i_s
            row_u_vals = np.zeros((2 * (n_rows * n_cols) + 1))
            row_v_vals = np.zeros((2 * (n_rows * n_cols) + 1))
            # update A for non-border pixels
            if 1 <= i < n_rows - 1 and 1 <= j < n_cols - 1:
                row_u_vals, row_v_vals = get_A_row(lamb, 'non-border', i_s, i, j, n_rows, n_cols, 8, im_x, im_y)

            # update A for upper border pixels
            elif i == 0 and 1 <= j < n_cols - 1:
                row_u_vals, row_v_vals = get_A_row(lamb, 'upper', i_s, i, j, n_rows, n_cols, 6, im_x, im_y)

            # update A for bottom border pixels
            elif i == n_rows - 1 and 1 <= j < n_cols - 1:
                row_u_vals, row_v_vals = get_A_row(lamb, 'bottom', i_s, i, j, n_rows, n_cols, 6, im_x, im_y)

            # update A for right border pixels
            elif 1 <= i < n_rows - 1 and j == n_cols - 1:
                row_u_vals, row_v_vals = get_A_row(lamb, 'right', i_s, i, j, n_rows, n_cols, 6, im_x, im_y)

            # update A for left border pixels
            elif 1 <= i < n_rows - 1 and j == 0:
                row_u_vals, row_v_vals = get_A_row(lamb, 'left', i_s, i, j, n_rows, n_cols, 6, im_x, im_y)

            # update A for up-left corner
            elif i == 0 and j == 0:
                row_u_vals, row_v_vals = get_A_row(lamb, 'up-left', i_s, i, j, n_rows, n_cols, 4, im_x, im_y)

            # update A for up-right corner
            elif i == 0 and j == n_cols - 1:
                row_u_vals, row_v_vals = get_A_row(lamb, 'up-right', i_s, i, j, n_rows, n_cols, 4, im_x, im_y)

            # update A for down-left corner
            elif i == n_rows - 1 and j == 0:
                row_u_vals, row_v_vals = get_A_row(lamb, 'down-left', i_s, i, j, n_rows, n_cols, 4, im_x, im_y)

            # update A for down-right corner
            elif i == n_rows - 1 and j == n_cols - 1:
                row_u_vals, row_v_vals = get_A_row(lamb, 'down-right', i_s, i, j, n_rows, n_cols, 4, im_x, im_y)

            A[row_u][:] = row_u_vals
            A[row_v][:] = row_v_vals

    sA = scipy.sparse.csr_matrix(A[1:, 1:])

    return sA


def create_b(img_x, img_y, img_t):
    n_rows, n_cols = img_x.shape
    b = np.zeros((2 * n_rows * n_cols + 1))
    for i in range(n_rows):
        for j in range(n_cols):
            i_s = i * n_cols + j + 1
            b[2 * i_s-1] = -(img_x[i][j] * img_t[i][j])
            b[2 * i_s] = -(img_y[i][j] * img_t[i][j])
    return b[1:]


def open_figure(figure_num, figure_title, figsize):
    plt.figure(figure_num, figsize)
    plt.clf()
    plt.suptitle(figure_title, fontsize=13)


def plot_images(figure_num, rows, cols, ind, images, titles, cmap, axis=True, colorbar=True):
    plt.figure(figure_num)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.8, hspace=0.8)
    for i in range(len(images)):
        if images[i] is None:
            continue
        ax = plt.subplot(rows, cols, i + ind)
        ax.set_title(titles[i])
        img = ax.imshow(images[i], cmap=cmap, interpolation='None')
        if not axis:
            plt.axis('off')
        if colorbar:
            plt.colorbar(img, fraction=0.046, pad=0.04)


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

    # Question 5

    lamb = 0.05
    images = [mdict['img1'], mdict['img2'], mdict['img3'], mdict['img4'], mdict['img5'], mdict['img6']]
    img_x, img_y = get_derivatives(images[0])
    img_t_list = [images[1] - images[0], images[2] - images[0], images[3] - images[0], images[4] - images[0],
               images[5] - images[0]]
    display = []
    titles = []
    for i in range(5):
        A = create_A(img_x, img_y, lamb)
        b = create_b(img_x, img_y, img_t_list[i])
        x = scipy.sparse.linalg.lsqr(A, b)[0]
        U = x[::2].reshape(images[0].shape)
        V = x[1::2].reshape(images[0].shape)
        I1_w = warp_image(images[0], U, V)
        display = display + [images[0], images[i+1], U, V, I1_w]
        titles = titles + ['I1', 'I' + str(i+2), 'U', 'V', 'I1 warpped']
    open_figure(3, 'Q5', figsize=(10,10))
    plot_images(3, 5, 5, 1,
                display,
                titles,
               'gray', axis=True, colorbar=False)
    plt.show()
    print('Done')


if __name__ == '__main__':
    main()
