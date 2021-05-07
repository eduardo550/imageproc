# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 02. Realce e Superresolução

import numpy as np
import imageio as io
import math

#construir o histograma cumulativo
def buildcumhistogram(image, colors=256):
    hist, edges = np.histogram(image, bins=colors)
    return np.cumsum(hist)

#equalizar o histograma
def histogram_equalization(image, histogram, L=256):
    new_image = np.empty(image.shape)
    norm = np.prod(image.shape)

    #funcao de equalizacao
    new_image = ((L-1)/norm)*histogram[image]

    return new_image

#funcao para identificador 0 da funcao de realce
def do_nothing(images, gamma):
    return images

#construir o histograma cumulativo individual
def individual_cumulative_histogram(images, gamma):
    new_images = []

    #para cada imagem, constroi o histograma cumulativo
    for img in images:
        h = buildcumhistogram(img)
        new_images.append(histogram_equalization(img, h))

    return np.asarray(new_images)

#construir o histograma cumulativo conjunto
def joint_cumulative_histogram(images, gamma):
    histogram = buildcumhistogram(images.flatten())
    
    #equalizando 
    new_images = histogram_equalization(images.flatten(), histogram)

    return new_images.reshape(images.shape)

#realizar a correcao gama
def gamma_correction(images, gamma):
    new_images = []

    #para cada imagem, aplica a correcao
    for img in images:
        new_images.append(np.floor(255*np.float_power(img/255.0, 1/gamma)))

    return np.asarray(new_images)

#realiza a superresolucao
def superresolution(images):
    #assumindo resolução segundo proposta do trabalho
    h_res = (images.shape[1]*2, images.shape[2]*2)
    h_image = np.empty(h_res)
    step = 2    #cada sub-imagem tem metade da resolucao da final

    #construindo a cena a partir das sub-imagens
    for i in np.arange(images.shape[1]):
        for j in np.arange(images.shape[2]):
            hi, hj = (i*step, j*step)
            h_image[hi][hj] = images[0][i][j]
            h_image[hi][hj+1] = images[1][i][j]
            h_image[hi+1][hj] = images[2][i][j]
            h_image[hi+1][hj+1] = images[3][i][j]

    return h_image

#Root Mean Squared Error
def rmse(image1, image2):
    assert image1.shape == image2.shape

    return math.sqrt(np.square(np.subtract(image1, image2)).mean())

#Começo do programa principal

#recebendo a entrada
imglowbase = input().rstrip()
imghigh = input().rstrip()
idfunc = input().rstrip()
gamma = np.single(input().rstrip())

#Lendo imagens e salvando em um numpy array
imglownames = [imglowbase + str(x) + ".png" for x in range(4)]
imglows = np.asarray([io.imread(name) for name in imglownames])

#identificadores da funcao de realce
enhancetype = {
    "0": do_nothing,
    "1": individual_cumulative_histogram,
    "2": joint_cumulative_histogram,
    "3": gamma_correction
}
enhanced_images = enhancetype[idfunc](imglows, gamma)

#criando a superresolucao
superimage = superresolution(enhanced_images)

#carregando imagem de referencia
ref_image = io.imread(imghigh)

error = rmse(superimage, ref_image)
print("%.4f" % error)
