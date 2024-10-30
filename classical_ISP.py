import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def interpolate_bilinear(RGB, image, row, col):
    # row, col is 1 if odd, else 0
    for i in range(row, image.shape[0], 2):
        for j in range(col, image.shape[1], 2):
            sum = 0
            count = 0
            if i > 0:
                sum += image[i-1][j]
                count += 1
            if i < image.shape[0] - 1:
                sum += image[i+1][j]
                count += 1
            if j > 0:
                sum += image[i][j-1]
                count += 1
            if j < image.shape[1] - 1:
                sum += image[i][j+1]
                count += 1
            RGB[i,j] = sum / count

    return RGB

def demosaic(image):
    # GRBG
    R = np.zeros_like(image)
    G = np.zeros_like(image)
    B = np.zeros_like(image)

    height, width = image.shape

    # initialize
    G[::2, ::2] = image[::2, ::2]
    G[1::2, 1::2] = image[1::2, 1::2]
    R[::2, 1::2] = image[::2, 1::2]
    B[1::2, ::2] = image[1::2, ::2]

    # fill G
    # R position, G[::2, 1::2]
    G = interpolate_bilinear(G, image, 0, 1)
    # B position, G[1::2, ::2]
    G = interpolate_bilinear(G, image, 1, 0)


    # fill R
    for row in range(0, height, 2):
        for col in range(0, width, 2):
            if col == 0:  # Left boundary
                R[row, col] = image[row, col+1]
            elif col == width - 1:  # Right boundary
                R[row, col] = image[row, col-1]
            else:
                R[row, col] = (image[row, col-1] + image[row, col+1]) // 2

    for row in range(1, height, 2):
        for col in range(1, width, 2):
            if row == 0:  # Top boundary
                R[row, col] = image[row+1, col]
            elif row == height - 1:  # Bottom boundary
                R[row, col] = image[row-1, col]
            else:
                R[row, col] = (image[row-1, col] + image[row+1, col]) // 2

    # 짝수 행, 홀수 열
    R = interpolate_bilinear(R, R, 1, 0)

    # fill B
    for row in range(1, height, 2):
        for col in range(1, width, 2):
            if col == 0:  # Left boundary
                B[row, col] = image[row, col+1]
            elif col == width - 1:  # Right boundary
                B[row, col] = image[row, col-1]
            else:
                B[row, col] = (image[row, col-1] + image[row, col+1]) // 2

    for row in range(0, height, 2):
        for col in range(0, width, 2):
            if row == 0:  # Top boundary
                B[row, col] = image[row+1, col]
            elif row == height - 1:  # Bottom boundary
                B[row, col] = image[row-1, col]
            else:
                B[row, col] = (image[row-1, col] + image[row+1, col]) // 2

    B = interpolate_bilinear(B, B, 0, 1)

    result = np.stack((R, G, B))

    return result

raw_image = np.load('./MW-ISPNet/dataset/raw/240924_135153_336_053.npy')

# demosaic
raw_rgb = demosaic(raw_image)
# print(raw_rgb)


# RGB to GREY
def rgb_to_hdr_intensity(rgb_image):
    # rgb_image.shape = (3, height, width), where 0th axis is R, G, B channels
    R = rgb_image[0]
    G = rgb_image[1]
    B = rgb_image[2]

    # Calculate intensity using weighted sum of R, G, and B channels
    intensity = 0.299 * R + 0.587 * G + 0.114 * B

    return intensity

# Apply the function to raw RGB image
hdr_intensity_image = rgb_to_hdr_intensity(raw_rgb)
print(f'HDR intensity: {hdr_intensity_image.shape}')

# # Display the result
# plt.imshow(hdr_intensity_image, cmap='gray')
# plt.colorbar()
# plt.show()
# print(hdr_intensity_image)

# Apply log scaling (Tone mapping)
def apply_log_scale(image):
    # Add a small constant to avoid log(0)
    log_image = np.log1p(image)
    return log_image / np.max(log_image)  # Normalize after applying log scale

ldr_intensity_image = apply_log_scale(hdr_intensity_image)

# plt.imshow(img_log_scaled)
# plt.show()
# print(f'Log scale: {ldr_intensity_image.shape}')
# print(ldr_intensity_image)

# GREY2RGB
ldr_rgb_image = (raw_rgb / hdr_intensity_image)**1.0 * ldr_intensity_image

ldr_rgb_image = ldr_rgb_image.transpose(1, 2, 0)
print(ldr_rgb_image.shape)
print(ldr_rgb_image)

plt.imshow(ldr_rgb_image)
plt.show()