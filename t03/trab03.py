# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 03. Filtragem

import numpy as np
import imageio as io

#Filtragem 1D
def oned_filtering(image, f):
    flat_image = image.flatten()
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

#Filtragem 2D
def twod_filtering(image, f):
    a, b = ((f.shape[0]-1)//2, (f.shape[1]-1)//2)     #cálculo dos intervalos do filtro, dessa vez em 2D
    new_image = []

    #aplicamos o filtro somente nos intervalos onde ele é definido
    for i in np.arange(a, image.shape[0]-a):
        line = []
        for j in np.arange(b, image.shape[1]-b):
            #aqui é feita a multiplicação element-wise para a vizinhança do pixel [i, j] entre a e b
            pixel = np.sum(np.multiply(image[(i-a):(i+a+1), (j-b):(j+b+1)], f))
            line.append(pixel)
        new_image.append(line)

    #fazendo o padding de reflexão, com os intervalos a e b ditando quantos pixels devem ser preenchidos
    return np.pad(new_image, ((a, a), (b, b)), "reflect", reflect_type="even")

#Filtragem 2D com filtro mediana
def median_2dfiltering(image, filter_size):
    a = (filter_size-1) // 2    #calculado apenas uma vez, visto que é quadrado

    padded_image = np.pad(image, ((a, a), (a, a)), "constant", constant_values=0)
    new_image = np.empty(image.shape)

    for i in np.arange(image.shape[0]):
        for j in np.arange(image.shape[1]):
            #cálculo de índices:
            #para o índice i, j da matriz final, precisamos da mediana da vizinhança da imagem preenchida centrada no pixel [i+a, j+a]
            #intervalo de índices da matriz preenchida: [i+a-a, i+a+a] para linhas | [j+a-a, j+a+a] para colunas
            #resultando em: [i, i+2a] | [j, j+2a]
            i_final, j_final = (i + (2*a), j + (2*a))
            new_image[i, j] = np.median(padded_image[i:i_final+1, j:j_final+1]) #novamente é adicionado 1 para incluir o último valor

    return new_image

# funcao de normalizacao, implementada no trabalho 1
# recebe a imagen original e o intervalo em que os valores serao normalizados como um tupla(max, min)
def normalize(image, values):
    new_image = np.empty(shape=image.shape, dtype=image.dtype)
    new_max, new_min = values                               #intervalo dos novos valores
    old_max, old_min = (np.amax(image), np.amin(image))     #intervalo dos valores originais

    new_image = ((image - old_min) * (new_max-new_min))/(old_max-old_min) + new_min

    return new_image

#Root Mean Squared Error
def rmse(image1, image2):
    assert image1.shape == image2.shape

    return np.sqrt(np.square(np.subtract(image1, image2)).mean())
    
#Começo do programa principal

#Lendo entrada do programa
imgref_name = input().rstrip()
F = input().rstrip()

reference_image = io.imread(imgref_name)

if (F == '1'):
    #Lendo tamanho do filtro
    filter_size = int(input().rstrip())

    #Lendo os pesos e salvando em um vetor
    filter1d = np.array((input().rstrip().split(" ")), dtype=float)
    #Garantindo que a leitura foi correta
    assert filter_size == filter1d.size

    #Aplicando filtro sobre a imagem de referencia
    filtered_image = oned_filtering(reference_image, filter1d)

elif (F == '2'):
    lateral_size = int(input().rstrip())
    filter2d = []
    l = 0

    #Lendo cada linha da matriz de pesos
    for n in np.arange(lateral_size):
        aux = input().rstrip().split(" ")
        filter2d.append(aux)
        l += len(aux)

    assert l == lateral_size ** 2

    filtered_image = twod_filtering(reference_image, np.asarray(filter2d, dtype=float))

elif (F == '3'):
    lateral_size = int(input().rstrip())

    filtered_image = median_2dfiltering(reference_image, lateral_size)
else:
    print("Method not recognized. Aborting.")
    exit()

#Normalizando a imagem filtrada e convertendo para bytes
normalized_image = normalize(filtered_image, (255, 0)).astype(np.uint8)
#Para o cálculo do erro, convertemos os valores para float para o cálculo correto
error = rmse(reference_image.astype(np.int32), normalized_image.astype(np.int32))
print("%.4f" % error)