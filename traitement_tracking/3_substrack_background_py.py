import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend 'Agg' pour la génération de fichiers sans affichage
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import util
import trackpy as tp
import functions_analyze as lib
import imageio
import os
import gc


path = '/Users/souchaud/Desktop/A_analyser/CytoOne_HL5_10x/'

# Liste tous les sous-dossiers dans le chemin spécifié
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

print(directories)

for directorie in directories :
    frame = lib.import_img_sequences(path='/Users/souchaud/Desktop/A_analyser/CytoOne_HL5_10x/'+directorie + '/mosaic/',
                                first_frame=0, last_frame=340, file_extension='.tif')
    
    # Conversion des images en tableau NumPy
    # Assurez-vous que toutes vos images ont les mêmes dimensions
    images_array = np.stack([np.array(img) for img in frame])

    # Calculer la médiane pixel par pixel
    median_image = np.median(images_array, axis=0)
    del images_array
    # Afficher l'image médiane
    plt.figure(figsize=(10, 10))
    plt.imshow(median_image, cmap='gray')  # Utilisez cmap='gray' pour les images en niveaux de gris
    plt.title("Image Moyenne")
    plt.axis('off')  # Cacher les axes
    plt.show()
    # Soustraire l'image médiane
    subtracted_image = frame[1] - median_image
    # Afficher l'image médiane
    plt.figure(figsize=(10, 10))
    plt.imshow(subtracted_image, cmap='gray')  # Utilisez cmap='gray' pour les images en niveaux de gris
    plt.title("substracted image")
    plt.axis('off')  # Cacher les axes
    plt.show()
    # Chemin où sauvegarder les images soustraites
    save_path = '/Users/souchaud/Desktop/A_analyser/CytoOne_HL5_10x/' +  directorie + '/mosaic_substract/'

    # Vérifiez si le dossier existe. Sinon, créez-le.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Le dossier {save_path} a été créé.")
    else:
        print(f"Le dossier {save_path} existe déjà.")

    # Supposons que 'frame' est une liste ou un itérable des images que vous voulez traiter
    # et que 'median_image' est l'image médiane calculée précédemment

    for i, img in enumerate(frame):
        # Convertir l'image en tableau NumPy si ce n'est pas déjà le cas
        img_array = np.array(img)
        
        # Soustraire l'image médiane de chaque image
        subtracted_image = img_array - median_image

        # Ajuster les valeurs négatives et NaN
        subtracted_image[subtracted_image < 0] = 0  # Remplacer les valeurs négatives par 0
        subtracted_image = np.nan_to_num(subtracted_image, nan=0, posinf=0, neginf=0)  # Remplacer NaN et inf par 0

        # Construire le nom de fichier pour chaque image soustraite
        filename = f'mosaic_total_{i}.tif'
        
        # Sauvegarder l'image ajustée
        # Note : Assurez-vous que la conversion en uint32 n'introduit pas de distorsions
        imageio.imwrite(os.path.join(save_path, filename), subtracted_image.astype(np.uint8))
        print(filename, " saved ")
    
    # À la fin de chaque itération où les variables ne sont plus nécessaires
    del frame, median_image, subtracted_image
    gc.collect()  # Encourage le ramasse-miettes à libérer la mémoire