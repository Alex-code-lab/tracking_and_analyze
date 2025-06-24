#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:59:50 2023

@author: souchaud

Ce code permet d'aller créer les mosaics d'images des expériences. Le schéma des dossiers est 
défini pour un seul exemple de dossier. Cela doit être revu au cas par cas. 
"""

import os
import numpy as np
from PIL import Image
import shutil


def create_mosaic_images(positions_dir, output_dir):
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
    # lines = []
    mosaic_line_all = []

    # Parcourir les dossiers de positions et extraire les informations des lignes et des colonnes
    for position_folder in position_folders:
        position = position_folder.split("_")
        rows.append(int(position[2]))
        # lines.append(int(position[1]))

    time = len(os.listdir(positions_dir + position_folders[0]))
    # Grouper les positions par code de temps (time code)
    for time_code in range(0, time):
        images = []

        # Parcourir les dossiers de positions
        for position_folder in position_folders:
            position = position_folder.split("_")
            current_row = int(position[2])

            # Obtenir le chemin du dossier de position et le chemin de l'image
            # corr espondante pour le code de temps actuel
            position_path = os.path.join(positions_dir, position_folder)
            image_file = sorted([f for f in os.listdir(position_path)
                                 if f.endswith(".tif")])[time_code]
            image_path = os.path.join(position_path, image_file)

            # Ouvrir l'image à l'aide de PIL (Pillow) et l'ajouter à la liste des images
            image = Image.open(image_path)
            images.append(image)

            # Si la ligne actuelle est la plus grande parmi toutes les lignes,
            # cela signifie qu'il est temps de créer une mosaïque de la ligne
            if int(current_row) == max(rows):
                # Concaténer les images de la ligne horizontalement
                mosaic_line = np.concatenate(images, axis=1)
                # Ajouter la mosaïque de ligne à la liste de toutes les mosaïques de lignes
                mosaic_line_all.append(mosaic_line)

                # Réinitialiser la liste des images pour la prochaine ligne
                images = []

        # Concaténer toutes les mosaïques de lignes horizontalement pour créer une mosaïque totale
        # mosaic_total = np.concatenate(mosaic_row_all, axis=0)
        mosaic_total = np.concatenate(mosaic_line_all[::-1], axis=0)

        # Convertir la mosaïque totale en une image PIL (Pillow)
        mosaic_image_tot_pil = Image.fromarray(mosaic_total)

        # Définir le chemin de sortie de l'image de mosaïque totale
        output_path = os.path.join(output_dir, f"mosaic_total_{time_code}.tif")

        # Enregistrer l'image de mosaïque totale sur le disque
        mosaic_image_tot_pil.save(output_path)

        # Réinitialiser les listes et les variables pour le prochain code de temps
        images = []
        mosaic_line_all = []
        mosaic_total = []

        # Afficher un message pour indiquer que l'image de mosaïque
        # totale a été enregistrée avec succès
        print(f"Mosaic total image for time code {time_code} saved at {output_path}")


def create_mosaic_images_too_heavy(positions_dir, output_dir):
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

    # Parcourir les dossiers de positions et extraire les informations des lignes et des colonnes
    for position_folder in position_folders:
        position = position_folder.split("_")
        rows.append(int(position[2]))
        lines.append(int(position[1]))

    time = len(os.listdir(os.path.join(positions_dir, position_folders[0])))
    # Grouper les positions par code de temps (time code)
    for time_code in range(0, time):
        images = []
        row_num = 0
        # Parcourir les dossiers de positions
        for position_folder in position_folders:
            position = position_folder.split("_")
            current_row = int(position[2])
            current_line = int(position[1])

            # Obtenir le chemin du dossier de position et le chemin de l'image
            # correspondante pour le code de temps actuel
            position_path = os.path.join(positions_dir, position_folder)
            image_file = sorted([f for f in os.listdir(position_path)
                                 if f.endswith(".tif")])[time_code]
            image_path = os.path.join(position_path, image_file)

            # Ouvrir l'image à l'aide de PIL (Pillow) et l'ajouter à la liste des images
            image = Image.open(image_path)
            print(image_path)
            image = np.array(image)
            images.append(np.array(image))

            # Si la ligne actuelle est la plus grande parmi toutes les lignes,
            # cela signifie qu'il est temps de créer une mosaïque de la ligne
            if int(current_row) == max(rows):
                row_num += 1
                # Concaténer les images de la ligne horizontalement
                mosaic_line = np.concatenate(images, axis=1)

                # Enregistrer la mosaïque de ligne dans un dossier temporaire
                temp_output_dir = os.path.join(output_dir, "temp_mosaic_line")
                os.makedirs(temp_output_dir, exist_ok=True)
                mosaic_line_path = os.path.join(temp_output_dir,
                                                f"mosaic_line_{time_code}_{row_num}.tif")
                mosaic_line_pil = Image.fromarray(mosaic_line)
                mosaic_line_pil.save(mosaic_line_path)

                # Afficher le chemin de la mosaïque de ligne enregistrée
                print(f"Mosaic line {current_line} image for\
                      time code {time_code} saved at {mosaic_line_path}")
                # Réinitialiser la liste des images pour la prochaine ligne
                images = []

                if int(current_line) == max(lines):

                    # Ajouter le chemin de la mosaïque de ligne à la
                    # liste de toutes les mosaïques de lignes

                    fichiers = os.listdir(temp_output_dir)
                    mosaic_line_all = []  # Liste pour stocker les chemins des mosaïques de ligne

                    # Parcourez les fichiers et supprimez-les un par un
                    for fichier in fichiers:
                        mosaic_line = Image.open(mosaic_line_path)
                        mosaic_line_all.append(np.array(mosaic_line))
                        chemin_fichier = os.path.join(temp_output_dir, fichier)
                        if os.path.isfile(chemin_fichier):
                            os.remove(chemin_fichier)

                    mosaic_total = np.concatenate(mosaic_line_all, axis=0)

                    # Convertir la mosaïque totale en une image PIL (Pillow)
                    mosaic_image_tot_pil = Image.fromarray(mosaic_total)

                    # Définir le chemin de sortie de l'image de mosaïque totale
                    output_path = os.path.join(output_dir, f"mosaic_total_{time_code}.tif")

                    # Enregistrer l'image de mosaïque totale sur le disque
                    mosaic_image_tot_pil.save(output_path)

                    # Supprimer le dossier temporaire et ses fichiers intermédiaires
                    shutil.rmtree(temp_output_dir)
                    # Réinitialiser les listes et les variables pour le prochain code de temps
                    del mosaic_total, mosaic_line, mosaic_line_all, mosaic_image_tot_pil,
                    del mosaic_line_pil, images, image

            # # Concaténer toutes les mosaïques de lignes horizontalement pour créer une mosaïque totale
            # for mosaic_line_path in mosaic_line_all[::-1]:
            #     mosaic_line = Image.open(mosaic_line_path)
            #     mosaic_total.append(np.array(mosaic_line))

            # mosaic_total = np.concatenate(mosaic_total, axis=0)

            # # Convertir la mosaïque totale en une image PIL (Pillow)
            # mosaic_image_tot_pil = Image.fromarray(mosaic_total)

            # # Définir le chemin de sortie de l'image de mosaïque totale
            # output_path = os.path.join(output_dir, f"mosaic_total_{time_code}.tif")

            # # Enregistrer l'image de mosaïque totale sur le disque
            # mosaic_image_tot_pil.save(output_path)

            # # Supprimer le dossier temporaire et ses fichiers intermédiaires
            # shutil.rmtree(temp_output_dir)

            # # Afficher un message pour indiquer que l'image de mosaïque
            # # totale a été enregistrée avec succès
            # print(f"Mosaic total image for time code {time_code} saved at {output_path}")


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
    mosaic_line_all = []

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
            current_row = int(position[2])
            print(current_row)

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
            if int(current_row) == max(rows):
                # Concaténer les images de la ligne horizontalement
                mosaic_line = np.concatenate(images, axis=1)
                # Ajouter la mosaïque de ligne à la liste de toutes les mosaïques de lignes
                mosaic_line_all.append(mosaic_line)

                # Réinitialiser la liste des images pour la prochaine ligne
                images = []

        # Concaténer toutes les mosaïques de lignes horizontalement pour créer une mosaïque totale
        # mosaic_total = np.concatenate(mosaic_row_all, axis=0)
        mosaic_total = np.concatenate(mosaic_line_all[::-1], axis=0)

        # Convertir la mosaïque totale en une image PIL (Pillow)
        mosaic_image_tot_pil = Image.fromarray(mosaic_total)

        # Définir le chemin de sortie de l'image de mosaïque totale
        output_path = os.path.join(output_dir, f"mosaic_total_{time_code}.tif")

        # Enregistrer l'image de mosaïque totale sur le disque
        mosaic_image_tot_pil.save(output_path)

        # Réinitialiser les listes et les variables pour le prochain code de temps
        images = []
        mosaic_line_all = []
        mosaic_total = []

        # Afficher un message pour indiquer que l'image de mosaïque
        # totale a été enregistrée avec succès
        print(f"Mosaic total image for time code {time_code} saved at {output_path}")


# # Specify the directories for specific names
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
    create_mosaic_images(positions_dir, output_dir)
    # create_mosaic_images_too_heavy(positions_dir, output_dir)
