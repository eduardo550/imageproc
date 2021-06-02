# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 04. Image Restoration

import numpy as np
import imageio as io
from numpy.lib.stride_tricks import as_strided#sliding_window_view
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

    #filter = sliding_window_view(padded_image, (filter_size, filter_size))
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

    estimated_noise_dispersion = get_dispersion(mode)(degraded_image[i1:i2+1, j1:j2+1])
    restored_image = denoising(degraded_image, mode, filter_size, gamma, estimated_noise_dispersion)

elif (F == '2'):
    filter_size = int(input().rstrip())
    assert filter_size in [3, 5, 7, 9, 11]
    alpha = float(input().rstrip())
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
