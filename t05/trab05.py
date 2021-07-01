# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 05. Image Descriptors

import numpy as np
import imageio as io

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

def normalized_histogram(image):
    bins, hist = np.unique(image, return_counts=True)
    norm_hist = hist / np.sum(hist)
    
    return norm_hist / np.linalg.norm(norm_hist)

def quantize(image, bits):

    return image >> (8-bits)

def main():
    #Lendo entrada do programa
    f = input().rstrip()
    g = input().rstrip()
    b = int(input().rstrip())

    object_image = np.asarray(io.imread(g))
    quantized_graylevel_object = quantize(luminance_preprocessing(object_image), b)
    
    dc = normalized_histogram(quantized_graylevel_object)
    

if __name__ == '__main__':
    main()