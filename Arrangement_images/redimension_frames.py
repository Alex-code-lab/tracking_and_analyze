#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:47:23 2023

@author: souchaud
"""
import os
import cv2
path_gen = '/Users/souchaud/Desktop/'

exp = [# '2023_01_25_ASMOT052_BoiteCytoOne_HL5_15s_5x_P3_AX3Chi1_t0_21c8_LDMOT01/8bits/',
    #    '2023_01_25_ASMOT053_BoiteCytoOne_HL5_15s_5x_P3_AX3Chi1_t90_21c8_LDMOT01',
        '2024_02_20_ASMOT127_AX3_MAT_P4_10x_CytoOne_HL5_1902-11h-2002-14h42_1/8bits/',
       # '2023_01_25_ASMOT054_BoiteCytoOne_HL5_15s_5x_P3_AX3Chi2_t0_21c_DGMOT01/8bits/',
       # '2023_01_27_ASMOT055_BoiteCytoOne_HL5_15s_5x_P4_AX3Chi2_t0_21c/8bits/',
       # '2023_02_15_ASMOT056_BoiteCytoOne_HL5_Chi2_P4_5x_15s_21c_t0_Em/8bits/',
       # '2023_02_15_ASMOT057_BoiteCytoOne_HL5_Chi2_P4_5x_15s_21c_t90_Em/8bits/'
       ]


for path in [os.path.join(path_gen, nom) for nom in exp]:
    for folder in [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]:
        for image in [f for f in os.listdir(os.path.join(path, folder))]:

            image_path = os.path.join(path, folder, image)

            image_initiale = cv2.imread(image_path)

            if image_initiale is not None:
                if image_initiale.dtype != 'uint8':
                    print("here is a prob on image : ", image,
                          'which type is:', image_initiale.dtype)
                else:
                    # Convert to grayscale if the image has more than one channel
                    # (i.e., is not already grayscale)
                    if len(image_initiale.shape) > 2:
                        image_grayscale = cv2.cvtColor(image_initiale, cv2.COLOR_BGR2GRAY)
                        # print(len(image_initiale.shape))
                    else:
                        image_grayscale = image_initiale

                    nouvelle_dim = (2048, 2048)
                    image_redim = cv2.resize(image_grayscale, nouvelle_dim,
                                             interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(image_path, image_redim)

            # nouvelle_dim = (2048, 2048)
            # image_redim = cv2.resize(image_initiale, nouvelle_dim, interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(os.path.join(path, folder, image), image_redim)
