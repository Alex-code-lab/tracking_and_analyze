#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:59:50 2023

@author: souchaud
"""

import os
import re
import shutil
import numpy as np
from PIL import Image


def create_mosaic_images(positions_dir, output_dir):
    # Triez d'abord par colonne, puis par ligne en ordre décroissant
    position_folders = sorted([f for f in os.listdir(positions_dir) if
                               os.path.isdir(os.path.join(positions_dir, f))],
                              key=lambda x: (int(x.split("_")[2]), -int(x.split("_")[1])))

    for time_code in range(0, 240):
        mosaic_col_all = []

        col_images = []

        for i in range(0, len(position_folders)):
            position = position_folders[i].split("_")
            row = int(position[1])
            col = int(position[2])

            position_path = os.path.join(positions_dir, position_folders[i])
            image_file = sorted([f for f in
                                 os.listdir(position_path) if f.endswith(".tif")])[time_code]
            image_path = os.path.join(position_path, image_file)

            image = Image.open(image_path)
            col_images.append(image)

            # Si c'est la dernière image de la colonne, concaténez-les toutes
            # et ajoutez-les à mosaic_col_all
            if row == 0:  # Since the rows are in descending order,row 0 will be the last in the col
                mosaic_col = np.concatenate(col_images, axis=0)
                mosaic_col_all.append(mosaic_col)
                col_images = []

        # Créez la mosaïque totale
        mosaic_total = np.concatenate(mosaic_col_all, axis=1)
        mosaic_image_tot_pil = Image.fromarray(mosaic_total)

        output_path = os.path.join(output_dir, f"mosaic_total_{time_code}.tif")
        mosaic_image_tot_pil.save(output_path)

        print(f"Mosaic total image for time code {time_code} saved at {output_path}")

# Maintenant, appelez votre fonction
# create_mosaic_images(path_to_positions, path_to_output)


def create_mosaic_images2(positions_dir, output_dir):
    """
    Make mosaics for an experiment.

    Parameters
    ----------
    positions_dir : TYPE
        DESCRIPTION.
    output_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Récupérer la liste des dossiers de positions triés
    position_folders = sorted([f for f in os.listdir(positions_dir)
                               if os.path.isdir(os.path.join(positions_dir, f))])

    # Initialiser des listes pour stocker les informations des lignes et des colonnes
    rows = []
    lines = []
    mosaic_row_all = []

    # Parcourir les dossiers de positions et extraire les informations des lignes et des colonnes
    for position_folder in position_folders:
        position = position_folder.split("_")
        # lines.append(int(position[2]))
        # rows.append(int(position[1]))
        lines.append(int(position[2]))
        rows.append(int(position[1]))

    # Grouper les positions par code de temps (time code)
    for time_code in range(0,
                           len(os.listdir(
                               os.path.join(positions_dir, position_folders[0])))):
        images = []

        # Parcourir les dossiers de positions
        for position_folder in position_folders:
            position = position_folder.split("_")
            current_line = int(position[2])

            # Obtenir le chemin du dossier de position et le chemin de l'image
            # correspondante pour le code de temps actuel
            position_path = os.path.join(positions_dir, position_folder)
            image_file = sorted([f for f in os.listdir(position_path)
                                 if f.endswith(".tif")])[time_code]
            image_path = os.path.join(position_path, image_file)

            # Ouvrir l'image à l'aide de PIL (Pillow) et l'ajouter à la liste des images
            image = Image.open(image_path)
            images.append(image)

            # Si la ligne actuelle est la plus grande parmi toutes les lignes,
            # cela signifie qu'il est temps de créer une mosaïque de la ligne
            if int(current_line) == max(lines):
                # Concaténer les images de la ligne verticalement
                mosaic_row = np.concatenate(images, axis=1)
                # Ajouter la mosaïque de ligne à la liste de toutes les mosaïques de lignes
                mosaic_row_all.append(mosaic_row)

                # Réinitialiser la liste des images pour la prochaine ligne
                images = []

        # Concaténer toutes les mosaïques de lignes horizontalement pour créer une mosaïque totale
        mosaic_total = np.concatenate(mosaic_row_all[::-1], axis=0)

        # Convertir la mosaïque totale en une image PIL (Pillow)
        mosaic_image_tot_pil = Image.fromarray(mosaic_total)

        # Définir le chemin de sortie de l'image de mosaïque totale
        output_path = os.path.join(output_dir, f"mosaic_total_{time_code}.tif")

        # Enregistrer l'image de mosaïque totale sur le disque
        mosaic_image_tot_pil.save(output_path)

        # Réinitialiser les listes et les variables pour le prochain code de temps
        images = []
        mosaic_row_all = []
        mosaic_total = []

        # Afficher un message pour indiquer que l'image de mosaïque
        # totale a été enregistrée avec succès
        print(f"Mosaic total image for time code {time_code} saved at {output_path}")


# Example usage:
# create_mosaic_images2("path_to_positions_dir", "path_to_output_dir")


# Example usage:
# create_mosaic_images3('/path/to/positions_dir', '/path/to/output_dir', (2048, 2048))


# Example usage:
# create_mosaic_images3('/path/to/positions_dir', '/path/to/output_dir')


def create_mosaic_images_agregat(positions_dir, output_dir):
    """
    Make mosaics for an experiment.

    Parameters
    ----------
    positions_dir : TYPE
        DESCRIPTION.
    output_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Récupérer la liste des dossiers de positions triés
    position_folders = sorted([f for f in os.listdir(positions_dir)
                               if os.path.isdir(os.path.join(positions_dir, f))])

    # Initialiser des listes pour stocker les informations des lignes et des colonnes
    rows = []
    lines = []
    mosaic_row_all = []

    # Parcourir les dossiers de positions et extraire les informations des lignes et des colonnes
    for position_folder in position_folders:
        position = position_folder.split("_")
        rows.append(int(position[2]))
        lines.append(int(position[1]))

    # Grouper les positions par code de temps (time code)
    for time_code in range(0, 240):
        images = []

        # Parcourir les dossiers de positions
        for position_folder in position_folders:
            position = position_folder.split("_")
            current_line = int(position[2])

            # Obtenir le chemin du dossier de position et le chemin de l'image
            # correspondante pour le code de temps actuel
            position_path = os.path.join(positions_dir, position_folder)
            image_file = sorted([f for f in os.listdir(position_path)
                                 if f.endswith(".tif")])[time_code]
            image_path = os.path.join(position_path, image_file)

            # Ouvrir l'image à l'aide de PIL (Pillow) et l'ajouter à la liste des images
            image = Image.open(image_path)
            images.append(image)

            # Si la ligne actuelle est la plus grande parmi toutes les lignes,
            # cela signifie qu'il est temps de créer une mosaïque de la ligne
            if int(current_line) == max(rows):
                # Concaténer les images de la ligne verticalement
                mosaic_row = np.concatenate(images, axis=1)
                # Ajouter la mosaïque de ligne à la liste de toutes les mosaïques de lignes
                mosaic_row_all.append(mosaic_row)

                # Réinitialiser la liste des images pour la prochaine ligne
                images = []

        # Concaténer toutes les mosaïques de lignes horizontalement pour créer une mosaïque totale
        # mosaic_total = np.concatenate(mosaic_row_all, axis=0)
        mosaic_total = np.concatenate(mosaic_row_all[::-1], axis=0)

        # Convertir la mosaïque totale en une image PIL (Pillow)
        mosaic_image_tot_pil = Image.fromarray(mosaic_total)

        # Définir le chemin de sortie de l'image de mosaïque totale
        output_path = os.path.join(output_dir, f"mosaic_total_{time_code}.tif")

        # Enregistrer l'image de mosaïque totale sur le disque
        mosaic_image_tot_pil.save(output_path)

        # Réinitialiser les listes et les variables pour le prochain code de temps
        images = []
        mosaic_row_all = []
        mosaic_total = []

        # Afficher un message pour indiquer que l'image de mosaïque
        # totale a été enregistrée avec succès
        print(f"Mosaic total image for time code {time_code} saved at {output_path}")


# # Specify the directories
# # positions_dir =
# '/Users/souchaud/Desktop/2022_12_09_ASMOT035_BoiteNonT_SorC_15s_5x_P6_AX3Chi2_t90/'
# # output_dir = '/Users/souchaud/Desktop/coucou/'

# # Specify the directories
# positions_dir = '/Users/souchaud/Desktop/2023_02_06_EmAg001_4.3-10-6-AX3_Chi1-p1/'
# name = re.search('Ag[0-9]{3}', positions_dir).group() + "_MOSAIC"
# output_dir = f'/Users/souchaud/Desktop/{name}/'
# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)


# # Create mosaic images for each time code
# create_mosaic_images_agregat(positions_dir, output_dir)


# Specify the directories


manip_names = []

# position_folders = '/Volumes/Labo_Alex_Mac/A_analyser/CytoOne_HL5/to_convert/'
condition = 'CytoOne_HL5_10x'
position_folders = f'/Users/souchaud/Desktop/A_analyser/{condition}/'


manip_names = [f for f in os.listdir(position_folders) if
               os.path.isdir(os.path.join(position_folders, f))]

# '/Volumes/Labo-Dicty/Alex/A_analyser/CytoOne_SorC/'
for manip_name in manip_names:
    positions_dir = position_folders + manip_name + '/8bits/'
    output_dir = position_folders + manip_name + '/mosaic/'
    if not os.path.exists(positions_dir):
        print(positions_dir)
        continue
    if os.path.exists(output_dir):
        continue
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create mosaic images for each time code
    create_mosaic_images2(positions_dir, output_dir)
