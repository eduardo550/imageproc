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

def normalized_histogram_descriptor(image):
    hist, _ = np.histogram(image, bins=image.max()+1)
    norm_hist = hist / np.sum(hist)
    return norm_hist / np.linalg.norm(norm_hist)

def diag_coocurrence_mat(image):
    mat_size = image.max() + 1
    mat = np.zeros((mat_size, mat_size), dtype=int)

    co_occurrences = stride_tricks.as_strided(
        x=image,
        shape=(image.shape[0]-1, image.shape[1]-1, 2),
        strides=(*image.strides, image.strides[0] + image.strides[1]),
        writeable=False
    )

    for intensity in np.unique(image):
        #convert to histogram
        #a, counts = np.unique(, return_counts=True)
        counts, _ = np.histogram(co_occurrences[co_occurrences[:, :, 0] == intensity, 1], bins=image.max()+1)
        mat[intensity, :] = counts

    return mat

def haralick_texture_descriptor(image):
    c = diag_coocurrence_mat(image)
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

    return np.asarray(descriptors) / np.linalg.norm(descriptors)

#Função de convolução implementada no trabalho 3, caso scipy esteja quebrado no run.codes
def convolve(image, f):
    flat_image = image.flatten()
    f = f.flatten()
    a = (f.size-1)//2   #cálculo dos intervalos do filtro, para guiar a aplicação na imagem resultante
    new_image = []

    #excluimos os primeiros e últimos 'a' pixels, que são indefinidos para a imagem final
    for i in np.arange(a, flat_image.size-a):
        #para o pixel i, o intervalo é entre [i-a, i+a]. +1 é somado para incluir o último elemento
        pixel = np.dot(flat_image[(i-a):(i+a+1)], f)
        new_image.append(pixel)

    #fazendo o padding e convertendo de volta para uma matriz
    new_image = np.pad(new_image, (a, a), "wrap")
    return new_image.reshape(image.shape)


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

    return descriptor 

def main():
    #Lendo entrada do programa
    f = input().rstrip()
    g = input().rstrip()
    b = int(input().rstrip())

    object_image = np.asarray(io.imread(g))
    quantized_graylevel_object = quantize(luminance_preprocessing(object_image), b)
    
    dc = normalized_histogram_descriptor(quantized_graylevel_object)
    dt = haralick_texture_descriptor(quantized_graylevel_object)
    dg = oriented_gradients_histogram(quantized_graylevel_object)
    print(dg)
if __name__ == '__main__':
    main()