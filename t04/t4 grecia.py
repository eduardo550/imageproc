# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 04. Image Restoration

import numpy as np
import imageio as io
from numpy.lib.stride_tricks import as_strided
from numpy.fft import rfft2, irfft2
#import scipy.fftpack as fp
# import matplotlib.pyplot as plt

def get_centrality(mode):
    if(mode == "average"):
        return np.mean
    else:
        return np.median

def get_dispersion(mode):
    if(mode == "average"):
        return np.std
    else:
        return lambda img: np.subtract(*np.percentile(img, [75, 25]))

def denoising(img, mode, filter_size, gamma, noise_disp):
    #Doing the thing so it doesn't go wrong
    noise_disp = 1 if noise_disp == 0 else noise_disp
    #Padding antes da aplicação do filtro
    padded_image = np.pad(img, (filter_size-1)//2, "symmetric")

    filter = as_strided(padded_image, (*img.shape, filter_size, filter_size), (*padded_image.strides, *padded_image.strides))
    centrality = get_centrality(mode)
    dispersion = get_dispersion(mode)

    filter = np.reshape(filter, (*filter.shape[:-2], filter_size**2))

    local_centrality = np.apply_along_axis(centrality, -1, filter).reshape(img.shape)
    local_dispersion = np.apply_along_axis(dispersion, -1, filter).reshape(img.shape)
    #Doing the thing so it doesn't go wrong
    local_dispersion[local_dispersion == 0] = noise_disp

    new_image = img - ((gamma * noise_disp/local_dispersion) * (img - local_centrality))

    return new_image

#Dado na descrição do trabalho
def gaussian_filter (k, sigma):
    arx = np.arange(-(k // 2) + 1.0 , (k//2) + 1.0)
    x, y = np.meshgrid(arx , arx)
    filt = np.exp( -(1/2) * (np.square(x) + np.square(y)) / np.square( sigma ))
    return filt / np.sum(filt)

def cls_filter(img,gaussian,laplacian,gamma):
    Hf = fp.fft2(fp.ifftshift(gaussian))
    Cf = fp.fft2(fp.ifftshift(laplacian))
    Hf = np.conj(Hf) / (Hf*np.conj(Hf) + gamma*Cf*np.conj(Cf))
    Yf = fp.fft2(img)
    I = Yf*Hf 
    im = np.abs(fp.ifft2(I))
    return (im, Hf) 

def constrained_least_squares(img, gaussian):
    fgaussian = rfft2(gaussian)
    laplacian = np.matrix([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    print("Laplacian")
    print(laplacian)
    inverse_laplacian = rfft2(laplacian)
    print("Inverse Laplacian")
    print(inverse_laplacian)
    inv_inv_lap = irfft2(inverse_laplacian)
    print("Laplacian Again")
    print(inv_inv_lap)
    fimg = rfft2(img)
    #do the math
    #
    #complex conjugate gaussian
    #----------------------------------------
    #abs(gaussian)^2 + gamma*abs(laplacian)^2
    #
    #above * input image


    #mat 4x3
    rconj=np.conj(fgaussian)
    print("rconj")
    print(rconj)
    rabsgaus=np.abs(fgaussian)
    print("rabsgaus")
    print(rabsgaus)
    rpowgaus=np.power(rabsgaus,2)
    print("rpowgaus")
    print(rpowgaus)
    #mat 3x2
    rabslap=np.abs(inverse_laplacian)
    print("rabslap")
    print(rabslap)
    rpowlap=np.power(rabslap,2)
    print("rpowlap")
    print(rpowlap)


    newmat1=np.matmul(rpowgaus,gamma*rpowlap)
    newmat2=np.divide(rconj,newmat1)

    rimg=newmat2*img
    #rimg=result*fimg
    return irfft2(rimg)
    #return irfft2(fimg)

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

    estimated_noise_dispersion = get_dispersion(mode)(degraded_image[i1:i2+1, j1:j2+1])
    restored_image = denoising(degraded_image, mode, filter_size, gamma, estimated_noise_dispersion)

elif (F == '2'):
    filter_size = int(input().rstrip())
    alpha = float(input().rstrip())
    
    assert filter_size in [3, 5, 7, 9, 11]
    g = gaussian_filter(filter_size, alpha)


    restored_image = constrained_least_squares(degraded_image, g)

else:
    print("Method not recognized. Aborting.")
    exit()

clipped_image = np.clip(restored_image, 0, 255)

reference_image = io.imread(imgref_name)

# fig = plt.figure()
# fig.add_subplot(1, 4, 1)
# plt.imshow(degraded_image, cmap='gray')
# plt.title("Degraded")
# fig.add_subplot(1, 4, 2)
# plt.imshow(restored_image, cmap='gray')
# plt.title("Restored")
# fig.add_subplot(1, 4, 3)
# plt.imshow(clipped_image, cmap='gray')
# plt.title("Clipped")
# fig.add_subplot(1, 4, 4)
# plt.imshow(reference_image, cmap='gray')
# plt.title("Reference")
# plt.show()


#Para o cálculo do erro, convertemos os valores para float para o cálculo correto
error = rmse(reference_image.astype(np.int32), clipped_image.astype(np.int32))
print("%.4f" % error)
