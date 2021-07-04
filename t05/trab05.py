# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 05. Image Descriptors

from math import inf
import numpy as np
import imageio as io
from numpy.lib import stride_tricks

def luminance_preprocessing(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    new_image = np.floor(
        (0.299 * red_channel) +
        (0.587 * green_channel) + 
        (0.114 * blue_channel)
    )

    return new_image.astype(image.dtype)

def quantize(image, bits):

    return image >> (8-bits)

def normalized_histogram_descriptor(image, bits):
    hist, _ = np.histogram(image, bins=2 ** bits)
    norm_hist = hist / np.sum(hist)
    return norm_hist / np.linalg.norm(norm_hist)

def diag_coocurrence_mat(image, b):
    mat_size = 2 ** b
    mat = np.zeros((mat_size, mat_size), dtype=int)
    co_occurrences = stride_tricks.as_strided(
        x=image,
        shape=(image.shape[0]-1, image.shape[1]-1, 2),
        strides=(*image.strides, image.strides[0] + image.strides[1]),
        writeable=False
    )


    for intensity in np.unique(image):
        counts, _ = np.histogram(co_occurrences[co_occurrences[:, :, 0] == intensity, 1], bins=mat_size)
        mat[intensity] = counts

    return mat

def haralick_texture_descriptor(image, b):
    c = diag_coocurrence_mat(image, b)
    c = c / np.sum(c)
    descriptors = []

    #Energy
    descriptors.append(
        np.sum(np.square(c))
    )

    #Entropy
    descriptors.append(
        np.multiply(
            np.sum(c * np.log10(c + 0.001)), 
            -1
        )
    )

    #Contrast
    #cálculo de (i - j)²
    ii, jj = np.indices(c.shape)
    factors = np.square(ii - jj)
    #cálculo do contraste
    descriptors.append(
        np.sum(
            c * factors
        ) / c.size
    ) #c.size == N² == número de elementos da matriz de co-ocorrencias

    #Correlation
    #TODO melhorar essa joça
    corr = 0
    for i in range(c.shape[0]):
        dir_mean_i = np.sum(i * np.sum(c, axis=1))
        std_dev_i = np.sum(np.square(i-dir_mean_i) * np.sum(c, axis=1))
        for j in range(c.shape[1]):
            dir_mean_j = np.sum(j * np.sum(c, axis=0))
            std_dev_j = np.sum(np.square(j-dir_mean_j) * np.sum(c, axis=0))
            if(std_dev_i*std_dev_j != 0):
                corr += ((i * j * c[i, j]) - (dir_mean_i*dir_mean_j)) / (std_dev_i * std_dev_j)
    descriptors.append(corr)

    #Homogeneity
    descriptors.append(
        np.sum(
            c / (1 + np.abs(ii - jj))
        )
    )

    norm = np.linalg.norm(descriptors)
    return descriptors / norm if norm != 0 else descriptors

def oriented_gradients_histogram(image):
    sobel_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    sobel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    try:
        from scipy import ndimage
        grad_x = ndimage.convolve(image.astype(np.double), sobel_x)
        grad_y = ndimage.convolve(image.astype(np.double), sobel_y)
    except ImportError:
        print("run codes not fixed")
        return

    magnitude_num = np.sqrt(np.square(grad_x) + np.square(grad_y))
    magnitude_mat = magnitude_num / np.sum(magnitude_num)
    
    np.seterr(divide='ignore', invalid='ignore')
    angles = np.arctan(grad_y / grad_x)
    
    #Fazendo as conversões
    angles = angles + (np.pi / 2)
    angles = np.degrees(angles)
    bins = np.arange(0, 180, 20)
    angle_bins = np.digitize(angles, bins, right=False)
    descriptor = np.zeros(9)
    for i in range(9):
        descriptor[i] = np.sum(magnitude_mat[angle_bins == i])

    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm != 0 else descriptor

def compute_descriptor(image, b):
    dc = normalized_histogram_descriptor(image, b)
    dt = haralick_texture_descriptor(image, b)
    dg = oriented_gradients_histogram(image)
    return np.concatenate((dc, dt, dg))

def find_object(image, b, object_descriptor):
    quantized_graylevel_image = quantize(luminance_preprocessing(image), b)
    #Separando as janelas
    window_coords = (quantized_graylevel_image.shape[0] // 16, quantized_graylevel_image.shape[1] // 16)
    window_strides = (quantized_graylevel_image.strides[0] * 16, quantized_graylevel_image.strides[1] * 16)

    windows = stride_tricks.as_strided(
        quantized_graylevel_image,
        shape=(*window_coords, 32, 32),
        strides=(*window_strides, *quantized_graylevel_image.strides),
        writeable=False
    )
    print(windows[0, 0, 0, 16], windows[0, 1, 0, 0])
    distances = np.zeros(window_coords)
    
    #TODO numpyar
    for i in range(window_coords[0]):
        for j in range(window_coords[1]):
            ld = compute_descriptor(windows[i, j], b)
            distances[i, j] = np.sqrt(
                np.sum(
                    np.square(object_descriptor - ld)
                )
            )

    coords = np.where(distances == distances.min())
    return (coords[0][0], coords[1][0])

def main():
    #Lendo entrada do programa
    f = input().rstrip()
    g = input().rstrip()
    b = int(input().rstrip())

    object_image = np.asarray(io.imread(f))
    quantized_graylevel_object = quantize(luminance_preprocessing(object_image), b)
    d = compute_descriptor(quantized_graylevel_object, b)

    large_image = np.asarray(io.imread(g))
    i, j = find_object(large_image, b, d)
    print(i, j)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots()
    ax . imshow ( large_image )
    rect = patches.Rectangle((j * 16 , i * 16 ) , 32 , 32 ,
    linewidth =1, edgecolor='r' , facecolor='none')
    ax.add_patch ( rect )
    plt.show ()

if __name__ == '__main__':
    main()