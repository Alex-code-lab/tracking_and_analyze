#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:49:44 2022

@author: souchaud
"""
import sys
print("python version is : " , sys.version)
from matplotlib import pyplot as plt
import pims
from skimage import io
import os
import glob
from skimage import exposure,  img_as_ubyte
import skimage
import re
def ImportImagesSequences(path) :
    # Pims permettra d'oubrir des séquence d'images. Pour le moment, on va déjà
    # repérer pour une seule image ce qu'il se passe exactement
    Frames = pims.ImageSequence(path)#, as_grey=True)
    plt.figure(figsize=(10, 10))
    io.imshow(Frames[0])
    return Frames

def positions_fields (pathway) :
    # Location of the files.
    os.chdir(pathway)
    dossier_positions = os.getcwd() # dossier de base ou se trouve le programme
    list_field_positions = glob.glob(dossier_positions + '/*')
    # formation de la liste des images à traiter
    return (list_field_positions)

#%%
# 2022_11_28_ASMOT023_BoiteNonT_HL5_15s_5x_P1_AX3Chi2_t0/_1/
# 2022_11_28_ASMOT024_BoiteNonT_HL5_15s_5x_P1_AX3Chi2_t90/_1/
pathway_positions = '/Users/souchaud/Desktop/A_analyser/images_vit_4_6_a_convertir/'
enregistrement =    '/Users/souchaud/Desktop/A_analyser/images_vit_4_6_a_convertir/_8bits/'
list_fields_positions = positions_fields(pathway_positions)
os.mkdir(enregistrement)
#%%
for positions in list_fields_positions : 
    frames= ImportImagesSequences(path = positions + '/*.tif')
    position_name = re.sub(pathway_positions,"",positions)    #%%
    os.mkdir(enregistrement+'/{}'.format(position_name))
    for i in range(0,len(frames),1) : 
        frame_rgb_skimage = skimage.color.rgba2rgb(frames[i])
        frame_32bit_skimage = skimage.color.rgb2gray(frame_rgb_skimage)
        frame_8bit_skimage = img_as_ubyte(exposure.rescale_intensity(frame_32bit_skimage))
        io.imsave(enregistrement+'/{}/{}_{}.tif'.format(position_name,position_name,i) , frame_8bit_skimage)
    

