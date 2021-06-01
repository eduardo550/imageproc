# Nomes:    Eduardo de Sousa Siqueira       nUSP: 9278299
#           Igor Barbosa Grécia Lúcio       nUSP: 9778821
# SCC0251 - Image Processing          1 semestre 2021
# Trabalho 04. Image Restoration

import numpy as np
import imageio as io



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
imgdegr_name = input().rstrip()
F = input().rstrip()
gamma = float(input().rstrip())

reference_image = io.imread(imgref_name)
degraded_image = io.imread(imgdegr_name)

if (F == '1'):
    line = input().rstrip().split(" ")
    i1, i2, j1, j2 = list(map(int, line))
    print(i2)
    filter_size = int(input().rstrip())
    mode = input().rstrip()
elif (F == '2'):
    filter_size = int(input().rstrip())
    assert filter_size in [3, 5, 7, 9, 11]
    alpha = float(input().rstrip())
else:
    print("Method not recognized. Aborting.")
    exit()

#Para o cálculo do erro, convertemos os valores para float para o cálculo correto
"""
error = rmse(reference_image.astype(np.int32), normalized_image.astype(np.int32))
print("%.4f" % error)
"""