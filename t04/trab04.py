# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 04. Image Restoration

import numpy as np
import imageio as io
from numpy.lib.stride_tricks import as_strided
from scipy.fft import rfft2, irfft2, fftshift

#Função que define o método usado para o cálculo de centralidade
#Retorna um function object de acordo com as opções "average" e "robust"
def get_centrality(mode):
    if(mode == "average"):
        return np.mean
    else:
        return np.median

#Função que define o método usado para o cálculo de dispersão
#Retorna um function object de acordo com as opções "average" e "robust"
def get_dispersion(mode):
    if(mode == "average"):
        return np.std
    else:
        return lambda img: np.subtract(*np.percentile(img, [75, 25]))

# Função que aplica o filtro de denoising de acordo com
#   mode: "average" ou "robust"
#   filter_size, gamma: definidos na entrada do programa
#   noise_disp: estimated noise dispersion, calculado de acordo com o mode e quadrante fornecidos na entrada
def denoising(img, mode, filter_size, gamma, noise_disp):
    #Doing the thing so it doesn't go wrong
    noise_disp = 1 if noise_disp == 0 else noise_disp
    #Padding antes da aplicação do filtro
    padded_image = np.pad(img, (filter_size-1)//2, "symmetric")

    #Criando um vetor que itera sobre todos os quadrantes relevantes da imagem, para paralelização da convolução
    #filter será um ndarray de 4 eixos, onde filter[i, j] é uma matriz de tamanho nxn, onde n é o tamanho do filtro
    #que representa a localidade do pixel (i, j) na imagem original onde é aplicado a convolução
    filter = as_strided(padded_image, (*img.shape, filter_size, filter_size), (*padded_image.strides, *padded_image.strides))
    
    #Funções de cálculo de centralidade e dispersão locais
    centrality = get_centrality(mode)
    dispersion = get_dispersion(mode)

    #Reduzindo um eixo do filtro, para aplicação das funções de centrality e dispersion
    filter = np.reshape(filter, (*filter.shape[:-2], filter_size**2))

    #Calculando centralidades e dispersões locais para cada pixel da imagem original
    local_centrality = np.apply_along_axis(centrality, -1, filter).reshape(img.shape) #Após apply_alog_axis, o vetor possui shape (n_rows, n_cols, 1). 
    local_dispersion = np.apply_along_axis(dispersion, -1, filter).reshape(img.shape) #Esse reshape é aplicado para converter para o tamanho da imagem original
    
    #Doing the thing so it doesn't go wrong
    local_dispersion[local_dispersion == 0] = noise_disp

    #Aplicando a convolução de forma paralela, com os valores calculados anteriormente
    new_image = img - ((gamma * noise_disp/local_dispersion) * (img - local_centrality))

    return new_image

#Dado na descrição do trabalho
def gaussian_filter(k=5, sigma=1.0):
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)

#Aplica o filtro de constrained least squares
#Recebe a imagem convolucionada, o filtro gaussiano usado e o parâmetro gamma fornecido na entrada
#Retorna a imagem com o filtro aplicado, no domínio espacial 
def constrained_least_squares(img, gamma, gaussian):
    inverse_laplacian = np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    #Padding para o tamanho da imagem
    inverse_laplacian = np.pad(inverse_laplacian, (gaussian.shape[0] - 3)//2, "constant", constant_values=0)
    
    row_diff, col_diff = (img.shape[0] - gaussian.shape[0], img.shape[1] - gaussian.shape[1])
    row_pad, col_pad = (row_diff // 2, col_diff//2)
    padded_laplacian = np.pad(inverse_laplacian, 
                                ((row_pad + (row_diff%2), row_pad), 
                                (col_pad + (col_diff%2), col_pad)),
                                "constant", constant_values=0
                            )
    padded_gaussian = np.pad(gaussian, 
                                ((row_pad + (row_diff%2), row_pad), 
                                (col_pad + (col_diff%2), col_pad)),
                                "constant", constant_values=0
                            )

    #Convertendo para o domínio de frequencias
    fimg = rfft2(img)
    flaplacian = rfft2(padded_laplacian)
    fgaussian = rfft2(padded_gaussian)

    #Conjugado do filtro
    fconjugate = np.conj(fgaussian)

    #Fazendo a operação
    denom = (np.abs(fgaussian)**2) + (gamma*(np.abs(flaplacian)**2))
    aux = fconjugate / denom
    new_img = irfft2(aux * fimg)

    #Aplicando o fftshift para devolver os valores para suas posições originais
    return fftshift(new_img)

#Root Mean Squared Error
def rmse(image1, image2):
    assert image1.shape == image2.shape

    return np.sqrt(np.square(np.subtract(image1, image2)).mean())
    
#Começo do programa principal

#Lendo entrada do programa
imgref_name = input().rstrip()
imgdegr_name = input().rstrip()
F = input().rstrip()
gamma = float(input().rstrip())

degraded_image = io.imread(imgdegr_name)

if (F == '1'):
    #Lendo entradas especificas para filtro 1
    line = input().rstrip().split(" ")
    i1, i2, j1, j2 = map(int, line)
    filter_size = int(input().rstrip())
    mode = input().rstrip()

    assert filter_size in [3, 5, 7, 9, 11]

    #Estimando dispersão de noise na vizinhança fornecida
    estimated_noise_dispersion = get_dispersion(mode)(degraded_image[i1:i2+1, j1:j2+1])
    #Aplicando o filtro
    restored_image = denoising(degraded_image, mode, filter_size, gamma, estimated_noise_dispersion)

elif (F == '2'):
    #Lendo entradas especificas para filtro 2
    filter_size = int(input().rstrip())
    alpha = float(input().rstrip())
    
    assert filter_size in [3, 5, 7, 9, 11]

    #Calculando o filtro gaussiano, para uso na função constrained_least_squares
    g = gaussian_filter(filter_size, alpha)

    #Aplicando o método de deconvolução
    restored_image = constrained_least_squares(degraded_image, gamma, g)
else:
    print("Method not recognized. Aborting.")
    exit()

#Cortando os valores para o intervalo especificado
clipped_image = np.clip(restored_image, 0, 255)

#Carregando imagem de referência
reference_image = io.imread(imgref_name)

#Para o cálculo do erro, convertemos os valores para float para o cálculo correto
error = rmse(reference_image.astype(np.int32), clipped_image.astype(np.int32))
print("%.4f" % error)
