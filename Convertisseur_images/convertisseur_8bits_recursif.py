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


def folders_to_treat(path_folder): 
    """
    Function to read all the folders we have to treat for the convertion

    Parameters
    ----------
    path_folder : STRING : folder path of the experiment

    Returns : list of folders to treat
    -------
    None.
    """
    folders_name_list = [f for f in os.listdir(path_folder)\
                         if os.path.isdir(path_folder + str(f))]
    # Pour chaque élément dans la liste, on réécrit le chemin
    # grace à os.path.join. 
    folders_list = [ os.path.join(path_folder,f) for f in folders_name_list]
    
    return folders_list, folders_name_list


def import_tiff_sequence(path):
    """
    Function that imports a sequence of TIFF images from a folder.
    
    path: the path to the folder containing the TIFF images.
    
    Returns a 3D numpy array with shape (num_images, height, width) 
    representing the sequence of TIFF images.
    """
    # Get the list of TIFF image filenames in the folder.
    file_names = [f for f in os.listdir(path) if f.endswith('.tif')]
    
    # Sort the filenames in the order that the images should be imported.
    file_names.sort()
    
    # Load the TIFF images from the filenames using the skimage library.
    images = [io.imread(os.path.join(path, f)) for f in file_names]
    
    # Concatenate the images into a single 3D numpy array.
    image_sequence = np.stack(images)
    
    return image_sequence, file_names 

def convert_to_8bit(image_stack):
    """
    Function that converts a stack of 16-bit images to 8-bit images.    
    image_stack: a 3D numpy array with shape (num_images, height, width) 
    representing a stack of 16-bit images. 
    Returns a 3D numpy array with shape (num_images, height, width) 
    representing a stack of 8-bit images.
    """
    # Convert the image stack to 8-bit images using the skimage library.
    # image_stack_8bit = img_as_ubyte(image_stack_rgb, 8)
    if len(image_stack.shape) not in (3,4) : 
        raise ValueError("image_stack has an invalid number of channels")
    if len(image_stack.shape) == 3 :     
        image_stack_8bit = img_as_ubyte(image_stack,8)
    else: 
        # Check if image_stack has 3 or 4 channels
        if image_stack.shape[3]==4:
            print("RGBA picture")
            image_stack_8bit = img_as_ubyte(rgb2gray(rgba2rgb(image_stack)), 8)
        elif image_stack.shape[3] == 3:
            print("RGB picture")
            image_stack_8bit = img_as_ubyte(rgb2gray(image_stack), 8)         
    return image_stack_8bit


def save_image_sequence(image_stack, filenames, saving_folder):
    """
    Function that saves a stack of images as a sequence of images in a folder.
    
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
    # Function that saves a stack of images as a stack image in
    # 8bits in a new folder.
    
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
#%%   
# =============================================================================
# =============================================================================
#                           Main : Coeur du programme :
# =============================================================================
# =============================================================================
# Define the path to the folder containing the TIFF images.
folder = '/Users/souchaud/Desktop/A_analyser/MOT46-270122/_1/'
folders_manip = os.listdir(folder)

folders, names = folders_to_treat(folder)
for folder, name in zip(folders, names): 
    # Import the sequence of TIFF images from the folder.
    image_sequence, filenames = import_tiff_sequence(folder)
    # Convert the image stack to 8-bit images.
    image_stack_8bit = convert_to_8bit(image_sequence)
    # if you put "folder" as saving folder, you will erase orignal stack for
    # the new one. Or you can give a new name to create a new folder and 
    # keep orignial frames.
    save_image_sequence(image_stack_8bit, filenames, '/Users/souchaud/Desktop/A_analyser/MOT46-270122_8bits/'+ name)

#%%
##If you don't have image sequences but a stack image, you can use :
############################# MAIN ##################################
# folder = '/Users/souchaud/Desktop/test/_1/'
# folders, names = folders_to_treat(folder)
# for folder, name in zip(folders, names): 
#     image_stack = io.imread(folder+'*.tif')
#     image_stack_8bit = convert_to_8bit(image_stack)
#     # if you put "folder" as saving folder, you will erase orignal stack for
#     # the new one. Or you can give a new name to create a new folder and 
#     # keep orignial frames.
#     save_stack_image(image_stack_8bit, name, folder)
    