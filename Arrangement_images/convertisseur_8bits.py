#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 08:27:24 2022

@author: AlexSouch

This program aim to convert every type of picture in 8 bits.
Works for image sequences and an alternative has been added for stack frames.
In this case, the part that could be change by the user is mostly the
"folders to treat" as this program has been written for a specific config of
datas. But the fonctions to import the frames / convert / and save should work
until you can find your frames.
"""
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.util import img_as_ubyte
import os
import numpy as np
import shutil


def folders_to_treat(path_folder):
    """
    Read all the folders we have to treat for the convertion.

    Parameters
    ----------
    path_folder : STRING : folder path of the experiment

    Returns : list of folders to treat
    -------
    None.
    """
    folders_name_list = [f for f in os.listdir(path_folder)
                         if os.path.isdir(path_folder + str(f))]
    # Pour chaque élément dans la liste, on réécrit le chemin
    # grace à os.path.join.
    folders_list = [os.path.join(path_folder, f) for f in folders_name_list]

    return folders_list, folders_name_list


def import_tiff_sequence(path, first: int = None, last: int = None):
    """
    Import a sequence of TIFF images from a folder.

    path: the path to the folder containing the TIFF images.

    Returns a 3D numpy array with shape (num_images, height, width)
    representing the sequence of TIFF images.
    """
    # Get the list of TIFF image filenames in the folder.
    file_names = [f for f in os.listdir(path) if f.endswith('.tif')]

    # Sort the filenames in the order that the images should be imported.
    file_names.sort()
    file_names = file_names[first:last]

    # Load the TIFF images from the filenames using the skimage library.
    images = [io.imread(os.path.join(path, f)) for f in file_names]

    # Concatenate the images into a single 3D numpy array.
    image_sequence = np.stack(images)

    return image_sequence, file_names


def convert_to_8bit(image_stack):
    """
    Convert stack of 16 in 8bits.

    Function that converts a stack of 16-bit images to 8-bit images.
    image_stack: a 3D numpy array with shape (num_images, height, width)
    representing a stack of 16-bit images.
    Returns a 3D numpy array with shape (num_images, height, width)
    representing a stack of 8-bit images.

    Parameters
    ----------
    image_stack : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    image_stack_8bit : TYPE
        DESCRIPTION.

    """
    # Convert the image stack to 8-bit images using the skimage library.
    # image_stack_8bit = img_as_ubyte(image_stack_rgb, 8)
    if len(image_stack.shape) not in (3, 4):
        raise ValueError("image_stack has an invalid number of channels")
    if len(image_stack.shape) == 3:
        image_stack_8bit = img_as_ubyte(image_stack, 8)
    else:
        # Check if image_stack has 3 or 4 channels
        if image_stack.shape[3] == 4:
            print("RGBA picture")
            image = rgba2rgb(image_stack)
            image_stack_8bit = img_as_ubyte(rgb2gray(image), 8)
        elif image_stack.shape[3] == 3:
            print("RGB picture")
            # image = rgb2gray(image_stack)
            # image_stack_8bit = img_as_ubyte(image, 8)
            num_images, height, width, depth = image_stack.shape
            image_stack_8bit = np.empty((num_images, height, width), dtype=np.uint8)

            for i in range(num_images):
                # Convert each 16-bit image to 8-bit using OpenCV
                # image = rgb2gray(image_stack)
                image_stack_8bit[i, :, :] = img_as_ubyte(rgb2gray(image_stack[i, :, :]), 8)
    return image_stack_8bit


def save_image_sequence(image_stack, filenames, saving_folder):
    """
    Save a stack of images as a sequence of images in a folder.

    image_stack: a 3D numpy array with shape (num_images, height, width)
    representing the stack of images to save.
    folder: the path to the folder where the images should be saved.

    Returns nothing.
    """
    # Create the folder if it does not exist.
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Iterate over the images in the stack and save them to the folder.
    for i, image in enumerate(image_stack):
        # Generate the filename for the image.
        # filename = f'image_{i:05d}.tiff'
        filename = filenames[i]
        # Save the image to the file.
        io.imsave(os.path.join(saving_folder, filename), image)


def save_stack_image(image_stack, filenames, saving_folder):
    """
    # Function that saves a stack of images as a stack image in 8bits in a new folder.

    # image_stack: a 3D numpy array with shape (num_images, height, width)
    # representing the stack of images to save.
    # folder: the path to the folder where the images should be saved.

    # Returns nothing.
    """
    # Create the folder if it does not exist.
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Iterate over the images in the stack and save them to the folder.
    # Save the image to the file.
    io.imsave(os.path.join(saving_folder, filenames), image_stack)


def create_and_move_images(base_dir):
    """
    Separate folders in 2.

    Parameters
    ----------
    base_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Créer le nouveau dossier avec 'bis' à la fin du nom
    new_dir = base_dir + "_bis"
    os.makedirs(new_dir, exist_ok=True)
    separate = int(len(os.listdir(base_dir))/2)
    # Déplacer les images numérotées de 0 à 900 vers le nouveau dossier
    for i in range(separate, len(os.listdir(base_dir)), 1):  # 0 to 900 (inclusive)
        image_name = f"img_{i:09}_PHASE_000.tif"  # Format du nom de l'image avec padding
        # image_name = f"img_channel000_position007_time{i:09}_z000.tif"
        image_path = os.path.join(base_dir, image_name)
        if os.path.isfile(image_path):  # Vérifier si l'image existe
            new_image_path = os.path.join(new_dir, image_name)
            shutil.move(image_path, new_image_path)
            print(f"Image {image_name} moved to {new_dir}")
# In[Main]
# =============================================================================
# =============================================================================
#                           Main : Coeur du programme :
# =============================================================================
# =============================================================================


# Define the path to the folder containing the TIFF images.
general_path = '/Users/souchaud/Desktop/A_analyser/CytoOne_HL5_10x/'
# general_path = '/Users/souchaud/Desktop/Jean-Paul/'´
# general_path = '/Volumes/Labo_Alex_Mac/A_analyser/CytoOne_HL5/to_convert/'

experiment_names = [f for f in os.listdir(general_path) if
                    os.path.isdir(general_path + f)]

# experiment_names = [
#                     '2022_12_09_ASMOT035_BoiteNonT_SorC_15s_5x_P6_AX3Chi2_t90/',
#                     '2022_12_09_ASMOT036_BoiteNonT_SorC_15s_5x_P6_AX3Chi2_t0/',
#                     '2022_12_15_ASMOT042_BoiteNonT_SorC_15s_5x_P8_AX3Chi2_t0_21c/',
#                     '2022_12_15_ASMOT043_BoiteNonT_SorC_15s_5x_P8_AX3Chi2_t90_21c/'
#                     ]

for experiment_name in experiment_names:
    name_manip = [f for f in os.listdir(os.path.join(general_path, experiment_name)) if
                  os.path.isdir(os.path.join(general_path, experiment_name, f))]
    if len(name_manip) == 1:
        if name_manip[0] == '8bits':
            continue
        if name_manip[0] == '8_bits':
            continue
        if name_manip[0] == 'mosaic':
            print('coucou')
            experiment_to_convert = general_path + experiment_name + '/' + name_manip[0] + '/'
            size = len(os.listdir(experiment_to_convert))
            image_sequence, filenames = import_tiff_sequence(experiment_to_convert)
            a = 0
            while a < size:
                # Convert the image stack to 8-bit images.
                image_stack_8bit = convert_to_8bit(image_sequence[0+a:a+1])
                # if you put "folder" as saving folder, you will erase orignal stack for
                # the new one. Or you can give a new name to create a new folder and
                # keep orignial frames.
                name = 'mosaic_8bits'
                save_image_sequence(image_stack_8bit, filenames[0+a:a+1],
                                    general_path + experiment_name + '/8bits/' + name)
                a += 1
                if a > size and a - 1 != size:  # Si le prochain a est sup à size et =! à size
                    a = size  # Réglez a à size
            continue

        experiment_to_convert = general_path + experiment_name + '/' + name_manip[0] + '/'
    else:
        experiment_to_convert = general_path + experiment_name

    # def of the folders to treat
    folders, names = folders_to_treat(experiment_to_convert)

    # In[If need to suppr some files :]
    # suppr = True
    # if suppr:
    #     for folder in folders:
    #         for numero in range(1900, len(os.listdir(folder)), 1):
    #             nom_fichier = f"img_{numero:09d}_PHASE_000.tif"
    #             chemin_fichier = os.path.join(folder, nom_fichier)
    #             if os.path.exists(chemin_fichier):
    #                 os.remove(chemin_fichier)
    #                 print(f"fichier supprimé : {nom_fichier}")

    # In[Utilisation de la fonction avec le dossier de base comme argument]
    # for folder in folders:
    #     create_and_move_images(folder)

    # In[conversion in 8bits]
    for folder, name in zip(folders, names):
        # Import the sequence of TIFF images from the folder.
        image_sequence, filenames = import_tiff_sequence(folder)
        # Convert the image stack to 8-bit images.
        image_stack_8bit = convert_to_8bit(image_sequence)
        # if you put "folder" as saving folder, you will erase orignal stack for
        # the new one. Or you can give a new name to create a new folder and
        # keep orignial frames.
        save_image_sequence(image_stack_8bit, filenames,
                            general_path + experiment_name + '/8bits/' + name)

# %%
# #If you don't have image sequences but a stack image, you can use :
# ############################ MAIN ##################################
# folder = '/Users/souchaud/Desktop/test/_1/'
# folders, names = folders_to_treat(folder)
# for folder, name in zip(folders, names):
#     image_stack = io.imread(folder+'*.tif')
#     image_stack_8bit = convert_to_8bit(image_stack)
#     # if you put "folder" as saving folder, you will erase orignal stack for
#     # the new one. Or you can give a new name to create a new folder and
#     # keep orignial frames.
#     save_stack_image(image_stack_8bit, name, folder)
