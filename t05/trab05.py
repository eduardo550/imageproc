# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 05. Image Descriptors

import numpy as np
import imageio as io
from numpy.lib import stride_tricks
import sys

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

#Função que constrói o descritor do histograma de cores normalizado
def normalized_histogram_descriptor(image, bits):
    hist, _ = np.histogram(image, bins=2 ** bits)
    norm_hist = hist / np.sum(hist)
    return norm_hist / np.linalg.norm(norm_hist)

#Função que constrói a matriz de co-ocorrências de intensidade diagonalmente conectadas
def diag_coocurrence_mat(image, b):
    mat_size = 2 ** b #O tamanho da matriz é igual à quantidade de intensidades diferentes da imagem
    mat = np.zeros((mat_size, mat_size), dtype=int)

    #Essa será a lista de todas as co-ocorrências diagonais da imagem
    #Para cada pixel x, y, co_occurrences[x, y] será um array de 2 posições, onde 
    #co_occurrences[x, y, 0] é o valor do pixel e co_occurrences[x, y, 1] é o pixel na diagonal direita 
    co_occurrences = stride_tricks.as_strided(
        x=image,
        shape=(image.shape[0]-1, image.shape[1]-1, 2),  #Não consideramos as últimas linha e coluna
        strides=(*image.strides, image.strides[0] + image.strides[1]),  #Os primeiros passos são iguais o da imagem, o terceiro valor guia ao pixel na diagonal (+1 linha +1 coluna)
        writeable=False     #Evita escritas de memória, pois essa não é uma função segura
    )

    #Para cada valor de intensidade, contamos todas as suas co_ocorrências usando numpy fancy indexing, e preenchemos essa linha da matriz
    for intensity in np.unique(image):
        counts, _ = np.histogram(co_occurrences[co_occurrences[:, :, 0] == intensity, 1], bins=mat_size)
        mat[intensity] = counts

    return mat

#Função que constrói o descritor a partir das 5 métricas da matriz de co-ocorrências
def haralick_texture_descriptor(image, b):
    c = diag_coocurrence_mat(image, b)
    c = c / np.sum(c)   #Normalizando a matriz
    descriptors = []

    #Energy
    descriptors.append(
        np.sum(np.square(c))
    )

    #Entropy
    descriptors.append(
        np.multiply(
            np.sum(c * np.log10(c + 0.001)), 
            -1      #Multiplica por -1, conforme descrito
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
    #Para o cálculo vetorizado, computamos todos os valores separadamente para cada linha e coluna
    #O resultado final será uma matriz, em que cada elemento (i, j) é o valor da correlação para esse pixel
    #Calculando somas parciais das linhas e colunas, transformando-as em vetores linha e coluna para possibilitar
    #o broadcasting
    sum_rows = np.sum(c, axis=1)[np.newaxis, :] #Transforma em um vetor linha de 1 dimensão
    sum_cols = np.sum(c, axis=0)[:, np.newaxis] #Transforma em um vetor coluna
    #Cálculo das médias direcionais. Será um vetor em que cada valor dir_mean[x] corresponde à média direcional da linha/coluna x
    dir_mean_i = np.sum(sum_rows * ii, axis=1, keepdims=True)
    dir_mean_j = np.sum(sum_cols * jj, axis=0, keepdims=True)
    #Cálculo dos desvios padrões, equivalente ao cálculo anterior
    std_dev_i = np.sum(np.square(ii - dir_mean_i) * sum_rows, axis=1, keepdims=True)
    std_dev_j = np.sum(np.square(jj - dir_mean_j) * sum_cols, axis=0, keepdims=True)
    #Inicializamos a matriz de correlações com zeros, para os casos em que os desvios padrões são 0
    corr = np.zeros(c.shape, dtype=np.double)
    #Cálculo vetorizado da correlação. Por causa do broadcasting de numpy e as conversões anteriores para vetores linha e coluna,
    #a multiplicação de dir_mean_i e dir_mean_j resulta em uma matriz de tamanho igual ao da matriz de co-ocorrências, onde o valor [i, j]
    #é igual à multiplicação de dir_mean_i[i] * dir_mean_j[j]. A multiplicação dos desvios ocorre de maneira equivalente
    corr = np.divide(
        (ii * jj * c) - (dir_mean_i * dir_mean_j),
        (std_dev_i * std_dev_j),
        out=corr,
        where=np.logical_and(std_dev_i != 0, std_dev_j != 0)    #O cálculo é feito apenas nas posições em que os desvios são acima de 0
    )
    #Fazendo a soma dos elementos da matriz anterior, obtemos o valor de correlação geral
    descriptors.append(np.sum(corr))

    #Homogeneity
    descriptors.append(
        np.sum(
            c / (1 + np.abs(ii - jj))
        )
    )

    #Evitando divisão por 0
    norm = np.linalg.norm(descriptors)
    return descriptors / norm if norm != 0 else descriptors

#Taken from: https://gist.github.com/arifqodari/dfd734cf61b0a4d01fde48f0024d1dc9
#Caso run.codes não aceite scipy convolve
def strided_convolution(image, weight, stride):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = stride_tricks.as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))

#Função que faz o cálculo do descritor de histograma dos gradientes orientados
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

    #Try-except para o run.codes caso scipy esteja quebrada
    try:
        from scipy import ndimage
        grad_x = ndimage.convolve(image.astype(np.double), sobel_x)
        grad_y = ndimage.convolve(image.astype(np.double), sobel_y)
    except ImportError:
        grad_x = strided_convolution(image.astype(np.double), sobel_x, 1)
        grad_y = strided_convolution(image.astype(np.double), sobel_x, 1)

    #Cálculo da matriz de magnitude
    magnitude_num = np.sqrt(np.square(grad_x) + np.square(grad_y))
    magnitude_mat = magnitude_num / np.sum(magnitude_num)
    
    #Ignora os erros de divisão por 0
    np.seterr(divide='ignore', invalid='ignore')
    #Algumas divisões de 0/0 resultam em NaN, mas isso é tratado automaticamente pela função np.digitize
    angles = np.arctan(grad_y / grad_x)
    
    #Fazendo as conversões
    angles = angles + (np.pi / 2)
    angles = np.degrees(angles)
    #Construindo as bins
    bins = np.arange(0, 180, 20)
    angle_bins = np.digitize(angles, bins, right=False)

    #Preenchendo as bins
    descriptor = [np.sum(magnitude_mat[angle_bins == i]) for i in range(9)]

    #Evitando divisão por 0
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm != 0 else descriptor

#Função auxiliar que calcula cada descritor e os retorna já concatenados
def compute_descriptor(image, b):
    dc = normalized_histogram_descriptor(image, b)
    dt = haralick_texture_descriptor(image, b)
    dg = oriented_gradients_histogram(image)
    return np.concatenate((dc, dt, dg))

#Função que faz a procura de um objeto na imagem à partir dos seus descritores
def find_object(image, b, object_descriptor):
    #Fazendo o pré-processamento
    quantized_graylevel_image = quantize(luminance_preprocessing(image), b)

    #Calculando a quantidade de janelas 32x32 que cabem na imagem
    window_coords = ((quantized_graylevel_image.shape[0] // 16)- 1, (quantized_graylevel_image.shape[1] // 16 )- 1) #Nessa conta, a última janela 32x32 seria metade fora do vetor, portanto é ignorada
    #Cada passo pula 16 posições, para a próxima janela 32x32
    window_strides = (quantized_graylevel_image.strides[0] * 16, quantized_graylevel_image.strides[1] * 16)

    #Esse vetor, de 4 dimensões, computa automaticamente todas as janelas 32x32 para a imagem
    #windows[x, y] é a matriz 32x32 correspondente à janela y da linha x
    windows = stride_tricks.as_strided(
        quantized_graylevel_image,
        shape=(*window_coords, 32, 32),
        #strides[0, 1] faz o cálculo dos pulos para cada janela. strides[2, 3] faz o cálculo do pixel dentro da janela
        strides=(*window_strides, *quantized_graylevel_image.strides),  
        writeable=False
    )
    
    distances = np.zeros(window_coords)
    
    for i in range(window_coords[0]):
        for j in range(window_coords[1]):
            ld = compute_descriptor(windows[i, j], b)
            distances[i, j] = np.sqrt(
                np.sum(
                    np.square(object_descriptor - ld)
                )
            )

    coords = np.where(distances == distances.min()) #Distância mínima é onde assumimos que o programa encontrou o objeto
    return (coords[0][0], coords[1][0])

def main(opt):
    #Lendo entrada do programa
    f = input().rstrip()
    g = input().rstrip()
    b = int(input().rstrip())

    #Computando descritor do objeto d
    object_image = np.asarray(io.imread(f))
    quantized_graylevel_object = quantize(luminance_preprocessing(object_image), b)
    d = compute_descriptor(quantized_graylevel_object, b)
    
    #Fazendo a busca baseado no descritor d obtido
    large_image = np.asarray(io.imread(g))
    i, j = find_object(large_image, b, d)
    print(i, j) #Impressão na tela dos índices da janela

    #Passando arg 1 na linha de comando, faz a impressão da imagem e o resultado da busca
    if opt:
       import matplotlib.pyplot as plt
       import matplotlib.patches as patches
       fig, ax = plt.subplots()
       ax . imshow ( large_image )
       rect = patches.Rectangle((j * 16 , i * 16 ) , 32 , 32 ,
       linewidth =1, edgecolor='r' , facecolor='none')
       ax.add_patch ( rect )
       plt.show ()

if __name__ == '__main__':
    #Conveniência para correção.
    opt = False
    if len(sys.argv) == 2:
        opt = True
    main(opt)
