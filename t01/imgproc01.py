# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 01. Geração de Imagens

import numpy as np
import math
import random

# definindo funcoes de geracao
# todas requerem como parametro o tamanho da imagem e o parametro Q
# todas retornam uma matriz de pixels quadrada, em um numpy array, sem normalizacao
# sao padronizadas para uso conveniente no codigo

#gerador f(x, y) = xy + 2y
def generate_image1(size, q):
    return np.fromfunction(lambda x, y: x*y + 2*y, shape=(size, size))

#gerador f(x, y) = |cos(x/Q) + 2sin(y/Q)|
def generate_image2(size, q):
    image = np.empty((size, size))

    for i in np.arange(size):
        for j in np.arange(size):
            image[i][j] = math.fabs(math.cos(i/q) + 2*math.sin(j/q))

    return image

#gerador f(x, y) = |3(x/Q)-cuberoot(y/Q)|
def generate_image3(size, q):
    image = np.empty((size, size))

    for i in np.arange(size):
        for j in np.arange(size):
            image[i][j] = math.fabs((3*(i/q)) - math.pow(j/q, 1/3))     #usando math.pow para garantir acuracia

    return image

#gerador f(x, y) = rand(0,1,S), preenchendo coluna por coluna
def generate_image4(size, q):
    image = np.empty((size, size))

    for j in np.arange(size):
        for i in np.arange(size):
            image[i][j] = random.random()

    return image

#randomwalk
def generate_image5(size, q):
    image = np.zeros((size, size))
    image[0][0] = 1
    idx, idy = (0, 0)
    for step in np.arange(1+(size**2)):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        idx = (idx+dx) % size
        idy = (idy+dy) % size
        image[idx][idy] = 1

    return image

# funcao de normalizacao
# formula copiada de https://en.wikipedia.org/wiki/Normalization_(image_processing)
# recebe a imagen original e o intervalo em que os valores serao normalizados como um tupla(max, min)
def normalize(image, values):
    new_image = np.empty(shape=image.shape, dtype=image.dtype)
    new_max, new_min = values                               #intervalo dos novos valores
    old_max, old_min = (np.amax(image), np.amin(image))     #intervalo dos valores originais

    #calculando imagem normalizada
    #aqui e feito o arredondamento para 4 casas decimais, para corrigir erros de precisao e garantir o funcionamento das funcoes 
    #de amostragem e quantizacao
    #escolhemos 4 casas decimais segundo a sugestao da proposta do trabalho, em que o calculo do erro futuramente e para ser impresso
    #com 4 casas decimais, e julgamos que nao havera perda de precisao significativa
    new_image = np.round(((image - old_min) * (new_max-new_min))/(old_max-old_min) + new_min, 4)

            
    return new_image

# operador para diminuicao da resolucao da imagem
# recebe a imagem original como parametro
# retorna uma nova imagem com as dimensoes desejadas
def downsampling(image, new_size):
    new_image = np.empty((new_size, new_size), dtype=image.dtype)
    old_size = image.shape[0]   #usando o fato que as imagens sao quadradas

    #calculando os passos. passo horizontal e vertical sao iguais dado que a matriz e quadrada
    step = math.floor(old_size/new_size)

    # fazendo o agrupamento
    for i in np.arange(new_size):
        for j in np.arange(new_size):
            new_image[i][j] = image[i*step][j*step]

    return new_image

def quantizing(image, bits):
    #normalizacao e conversao dos valores
    new_image = normalize(image, (255, 0)).astype(np.uint8)
    #para preservar os B bits mais significativos: (8-B) shifts
    shift_amnt = 8-bits

    #aplicando bitshifts
    new_image = new_image >> shift_amnt
    new_image = new_image << shift_amnt

    return new_image

def root_squared_error(image1, image2):
    #garantindo que as imagens sao de dimensoes iguais
    assert image1.shape == image2.shape
    error = 0.0

    # acumulando o quadrado da diferenca entre pixels
    for i in np.arange(image1.shape[0]):
        for j in np.arange(image1.shape[1]):
            error += (float(image1[i][j]) - float(image2[i][j])) ** 2   #convertendo valores para float para evitar erros de overflow

    # retornando a raiz quadrada do valor calculado
    return math.sqrt(error)

#Comeco do programa principal

# inicializando um seletor de funcoes geradoras de imagem
image_generators = {
    "1": generate_image1,
    "2": generate_image2,
    "3": generate_image3,
    "4": generate_image4,
    "5": generate_image5
}
# lendo entrada da stdin, como descrito
filename = input().rstrip()
synthimgsize = int(input().rstrip())
genfunctionselector = input().rstrip()
q = float(input().rstrip())
digitizedimgsize = int(input().rstrip())
bitsperpixel = int(input().rstrip())
seed = int(input().rstrip())
random.seed(seed)

# gerando a imagem sintetica com normalizacao
generated_image = normalize(image_generators[genfunctionselector](synthimgsize, q), (65535, 0))
# amostragem e quantizacao
digitized_image = quantizing(downsampling(generated_image, digitizedimgsize), bitsperpixel)

#carregando imagem de referencia
reference_image = np.load(filename)

#calculando e imprimindo o erro
error = root_squared_error(digitized_image, reference_image)
print("%.4f" % error)
