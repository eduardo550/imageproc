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
def identity(images, gamma):
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
    #histograma de todas as imagens na forma de um numpy array de 1D
    all_images = images.flatten()
    joint_histogram = buildcumhistogram(all_images)
    
    #equalizando 
    new_images = histogram_equalization(all_images, joint_histogram)

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
    n_images = images.shape[0]  #qtde de imagens
    h_res = (images.shape[1]*math.sqrt(n_images), images.shape[2]*math.sqrt(n_images)) #resolucao da imagem final
    h_image = np.empty(np.int32(h_res))
    step = np.int32(math.sqrt(n_images))
    x_offset = 0
    y_offset = 0
    id_imgs = 0

    #construindo a cena a partir das sub-imagens
    for img in images:
        n_lines, n_cols = img.shape
        for i in np.arange(n_lines):
            for j in np.arange(n_cols):
                x,y = (i*step + x_offset, j*step + y_offset)
                h_image[x][y] = img[i][j]

        id_imgs += 1
        y_offset = (id_imgs) % step
        x_offset = (id_imgs) // step #divisao inteira

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
    "0": identity,
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
