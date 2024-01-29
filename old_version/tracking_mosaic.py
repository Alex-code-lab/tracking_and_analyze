#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:51:23 2023

@author: souchaud
"""

import os
import glob
import imageio
import numpy as np
import pandas as pd
import trackpy as tp
from tqdm import tqdm
# import functions_track_and_analyze as lib
import matplotlib.pyplot as plt
import cv2
from skimage import util
# from joblib import Parallel, delayed


# Consolidated parameters
PARAMS = {
    'diameter': 25,
    'minmass': 50,
    'max_size': 35,
    'separation': 15,
    'noise_size': 0,
    'smoothing_size': None,
    'invert': True,
    'percentile': 10,
    'topn': None,
    'preprocess': True,
    'max_iterations': 25,
    'filter_before': None,
    'filter_after': None,
    'characterize': True,
    'engine': 'auto',
    'threshold': 40,  # 90,
    'min_frames': 1,
    'max_displacement': 15,
    'frame_interval': 15,
    'pixel_size': 1.2773,
    'remove_exts': ['.jpg', '.svg', 'hdf5', '.png'],
    'long_time': True,
    'max_frame': 1000,
    # 'data_dir': '/Users/souchaud/Desktop/A_Analyser/CytoOne_HL5/',
    'data_dir': '/Users/souchaud/Desktop/A_Analyser/NonT_SorC/',
    # 'output_dir': '/Users/souchaud/Desktop/Analyses/CytoOne_HL5_longtime/'
    # 'data_dir': '/Volumes/Labo_Alex_Mac/A_analyser/CytoOne_HL5/',´
    # 'output_dir': '/Users/souchaud/Desktop/Analyses/CytoOne_HL5_longtime/'
    'output_dir': '/Users/souchaud/Desktop/Analyses/NonT_SorC_longtime_New/'
}

# Define experiment names
# EXPERIMENT_NAMES = [
#       '2023_05_31_ASMOT064_BoiteCytoOne_SorC_Chi2_P3_5x_15s_21c_t0/mosaic/',
#       '2023_06_12_ASMOT065_BoiteCytoOne_SorC_Chi2_P2_5x_15s_21c_t0/mosaic/',
#       '2023_06_12_ASMOT066_BoiteCytoOne_SorC_Chi2_P2_5x_15s_21c_t08t2h/mosaic/',
#       '2023_06_12_ASMOT067_BoiteCytoOne_SorC_Chi2_P2_5x_15s_21c_t08t5h/mosaic/',
# #      '2023_06_12_ASMOT068_BoiteCytoOne_SorC_Chi2_P3_5x_15s_21c_t0_tooconfluent/mosaic/',
#       '2023_06_12_ASMOT070_BoiteCytoOne_SorC_Chi2_P4_5x_15s_21c_t0_ok/mosaic/',
# #      '2023_06_12_ASMOT071_BoiteCytoOne_SorC_Chi2_P5_5x_15s_21c_t0_ilyadestas/mosaic/',
#       '2023_06_12_ASMOT072_BoiteCytoOne_SorC_Chi2_P5_5x_15s_21c_t2h/mosaic/',
#       '2023_06_21_ASMOT073_BoiteCytoOne_SorC_Chi2_P5_5x_15s_21c_t2h_overnight/mosaic/',
#       '2023_06_22_ASMOT074_BoiteCytoOne_SorC_Chi2_P6_5x_15s_21c_t0/mosaic/',
#       '2023_06_28_ASMOT075_BoiteCytoOne_SorC_Chi2_P6_5x_15s_21c_tovvernight/mosaic/',
#                       ]
# EXPERIMENT_NAMES = [
# '2022_12_19_ASMOT050_BoiteCytoOne_HL5_15s_5x_P9_AX3Chi2_t0_21c/mosaic/',
# '2022_12_19_ASMOT051_BoiteCytoOne_HL5_15s_5x_P9_AX3Chi2_t90_21c/mosaic/',
# '2023_01_25_ASMOT052_BoiteCytoOne_HL5_15s_5x_P3_AX3Chi1_t0_21c8_LDMOT01/mosaic/',
# '2023_01_25_ASMOT053_BoiteCytoOne_HL5_15s_5x_P3_AX3Chi1_t90_21c8_LDMOT01/mosaic/',
# '2023_01_25_ASMOT054_BoiteCytoOne_HL5_15s_5x_P3_AX3Chi2_t0_21c_DGMOT01/mosaic/',
# '2023_01_27_ASMOT055_BoiteCytoOne_HL5_15s_5x_P4_AX3Chi2_t0_21c/mosaic/',
# '2023_02_15_ASMOT056_BoiteCytoOne_HL5_Chi2_P4_5x_15s_21c_t0_Em/mosaic/',
# '2023_02_15_ASMOT057_BoiteCytoOne_HL5_Chi2_P4_5x_15s_21c_t90_Em/mosaic/',
# '2023_02_16_ASMOT058_BoiteCytoOne_HL5_Ch1_p1_5x_15s_21c_t90_Em/mosaic/',
# '2023_02_16_ASMOT059_BoiteCytoOne_HL5_Chi1_p1_5x_15s_21c_t0_Em/mosaic/',
# '2023_02_16_ASMOT060_BoiteCytoOne_HL5_Chi2_p5_5x_15s_21c_t0_Em/mosaic/',
# '2023_02_23_ASMOT062_BoiteCytoOne_HL5_Chi2_p6_5x_15s_21c_t90/mosaic/',
# '2023_02_23_ASMOT063_BoiteCytoOne_HL5_Chi2_p6_5x_15s_21c_t0/mosaic/',
# '2023_06_29_ASMOT076_BoiteCytoOne_HL5_Chi2Em_P7_5x_15s_21c_t0/mosaic/',
# '2023_06_29_ASMOT077_BoiteCytoOne_HL5_Chi2Em_P7_5x_15s_21c_t2h-pour5h/mosaic/',
# '2023_06_29_ASMOT078_BoiteCytoOne_HL5_Chi2Em_P7_5x_15s_21c_t0_OVERNIGHT/mosaic/',
# '2023_07_05_ASMOT080_BoiteCytoOne_HL5_Chi2Em_P2_5x_15s_21c_t0h-pour5h/mosaic/',
# '2023_07_05_ASMOT081_BoiteCytoOne_HL5_Chi2Em_P2_5x_15s_21c_t0h-pournuit/mosaic/',
# '2023_09_06_ASMOT089_AX3_P2_CytoOnne_HL5_5x_15s_t0_4h/mosaic/',
# '2023_09_07_ASMOT091_AX3_P2_CytoOnne_HL5_5x_15s_t0/mosaic/',
# '2023_09_07_ASMOT092_AX3_P2_CytoOnne_HL5_5x_15s_t5h/mosaic/',
# '2023_09_08_ASMOT093_AX3_P2_CytoOnne_HL5_5x_15s_t0h/mosaic/',
# ]

EXPERIMENT_NAMES = [f + '/mosaic/' for f in os.listdir(PARAMS['data_dir'])
                    if os.path.isdir(os.path.join(PARAMS['data_dir'], f))]

# %%


def compute_mean_speed(filtered):
    """
    Compute mean speed.

    Parameters
    ----------
    - filtered: DataFrame with tracked cells
    Returns
    - mean_speed: Mean speed of all particles
    - mean_speed_part: Mean speed per particle
    """
    dx = filtered.groupby('particle')['x'].diff()
    dy = filtered.groupby('particle')['y'].diff()
    displacement = np.sqrt(dx**2 + dy**2)
    duration = filtered.groupby('particle')['frame'].diff() * PARAMS['frame_interval']
    mean_speed = (displacement.sum() / duration.sum()) * PARAMS['pixel_size'] * 60
    instant_speed = displacement / duration
    mean_speed_part = instant_speed.groupby(filtered['particle']).mean() * PARAMS['pixel_size'] * 60
    return mean_speed, mean_speed_part


def clean_directory(dir_path):
    """Remove all files with the specified extensions in the directory."""
    for file in os.listdir(dir_path):
        if file.endswith(tuple(PARAMS['remove_exts'])):
            os.remove(os.path.join(dir_path, file))


def process_experiment(exp_name, PARAMS):
    """Process a single experiment."""
    output_path = os.path.join(PARAMS['output_dir'], exp_name)
    os.makedirs(output_path, exist_ok=True)

    clean_directory(output_path)

    experiment_data_dir = os.path.join(PARAMS['data_dir'], exp_name)

    def extract_number(filename):
        # Extrait le numéro à partir du nom de fichier
        base_name = os.path.basename(filename)
        # Supprime l'extension et extrait le numéro
        number = int(base_name.split('_')[-1].split('.')[0])
        return number

    tiff_files = sorted(glob.glob(os.path.join(experiment_data_dir, "*.tif")), key=extract_number)

    # Use PARAMS dictionary to get the parameters
    frame_data = []
    frame_counter = 0
    boucle = []
    if PARAMS['long_time'] is False:
        nbr_frame_study_total = PARAMS['max_frame']
    else:
        nbr_frame_study_total = len(os.listdir(experiment_data_dir))

    if nbr_frame_study_total > 150:
        number = 150
        while number < nbr_frame_study_total:
            boucle.append(150)
            number += 150
            if number > nbr_frame_study_total:
                boucle.append(nbr_frame_study_total - len(boucle) * 150)
        nbr_frame_study = 150
    else:
        nbr_frame_study = nbr_frame_study_total
        boucle.append(nbr_frame_study)

    # Process each batch of frames
    import time
    for i in tqdm(boucle, desc="processing batches"):
        batch_frames = tiff_files[frame_counter:frame_counter + i]
        batch_data = [np.array(imageio.imread(tiff_file)) for tiff_file in batch_frames]
        time_count = time.time()
        for num, frame in enumerate(batch_data):
            frame = util.invert(frame)
            # Create a CLAHE object (Arguments are optional)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
            # Apply CLAHE
            frame = clahe.apply(frame)

            # # Seuillage
            # _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # # Opérations morphologiques
            # kernel = np.ones((3, 3), np.uint8)
            # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)

            # Seuillage
            # _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

        # for num, frame in enumerate(batch_data):
        #     frame = cv2.bitwise_not(frame)

        #     # Créer et appliquer CLAHE
        #     frame = clahe.apply(frame)

        #     # Appliquer un filtre médian pour réduire le bruit
        #     frame = cv2.medianBlur(frame, ksize=3)

        #     # Seuillage
        #     _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #     # Opérations morphologiques
        #     kernel = np.ones((3, 3), np.uint8)
        #     frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)

            batch_data[num] = frame

            if num == 0:
                plt.figure(figsize=(12, 6))
                plt.imshow(frame, cmap='gray')
                plt.show()

                f = tp.locate(frame,
                              diameter=PARAMS['diameter'],
                              minmass=PARAMS['minmass'],
                              maxsize=PARAMS['max_size'],
                              separation=PARAMS['separation'],
                              noise_size=PARAMS['noise_size'],
                              smoothing_size=PARAMS['smoothing_size'],
                              threshold=PARAMS['threshold'],
                              invert=PARAMS['invert'],
                              percentile=PARAMS['percentile'],
                              topn=PARAMS['topn'],
                              preprocess=PARAMS['preprocess'],
                              max_iterations=PARAMS['max_iterations'],
                              filter_before=PARAMS['filter_before'],
                              filter_after=PARAMS['filter_after'],
                              characterize=PARAMS['characterize'],
                              engine=PARAMS['engine'])
                print(len(f))
                #             diameter=7,
                #               minmass=50, maxsize=15,
                #               separation=10, invert=PARAMS['invert'],
                #               characterize=True, engine='auto')
                plt.figure(figsize=(12, 6))
                tp.annotate(f, frame)
                plt.show()
        print("temps de travail sur les images : ", (time.time() - time_count)/60, "min")
        # frame[frame > 90] = 0
        # plt.figure(figsize=(12, 6))
        # plt.imshow(cv2.GaussianBlur(frame, (251, 251), 0), cmap='gray')
        # plt.show()
        # if nbr_frame_study_total <= 260:
        cells_loc = tp.batch(batch_data,
                             diameter=PARAMS['diameter'],
                             minmass=PARAMS['minmass'],
                             maxsize=PARAMS['max_size'],
                             separation=PARAMS['separation'],
                             noise_size=PARAMS['noise_size'],
                             smoothing_size=PARAMS['smoothing_size'],
                             threshold=PARAMS['threshold'],
                             invert=PARAMS['invert'],
                             percentile=PARAMS['percentile'],
                             topn=PARAMS['topn'],
                             preprocess=PARAMS['preprocess'],
                             max_iterations=PARAMS['max_iterations'],
                             filter_before=PARAMS['filter_before'],
                             filter_after=PARAMS['filter_after'],
                             characterize=PARAMS['characterize'],
                             engine=PARAMS['engine'])
        cells_loc['frame'] += frame_counter
        frame_counter += i
        frame_data.append(cells_loc)

    all_features = pd.concat(frame_data)

    try:
        trajectories = tp.link_df(all_features,
                                  search_range=15,  # PARAMS['max_displacement'],
                                  memory=5,
                                  neighbor_strategy='KDTree',
                                  link_strategy='auto',  # 'hybrid',
                                  adaptive_stop=3,
                                  # verify_integritxy=True,
                                  )
        trajectories.to_hdf(os.path.join(output_path, 'filtered.hdf5'), 'table')
        # verify_intetegrity=True)
        # neighbor_strategy='KDTree',
    except tp.SubnetOversizeException:
        print("Issue with this one")

    filtered = tp.filter_stubs(trajectories, PARAMS['min_frames'])
    # filtered = filtered[~filtered.particle.isin(
    #     tp.filter_clusters(filtered, quantile=0.1,
    #                        threshold=filtered['size'].mean() * 1).index)]
    all_features.to_hdf(os.path.join(output_path, 'features.hdf5'), 'table')
    filtered.to_hdf(os.path.join(output_path, 'filtered.hdf5'), 'table')

    fig, axis = plt.subplots(figsize=(10, 10))
    plt.title(f'Trajectories after suspicious particles {exp_name}')
    tp.plot_traj(filtered, superimpose=batch_data[0], label=(False))
    plt.show()
    return filtered


# def main():
#     """Process all experiments."""
#     # Use all available CPUs but one. This helps in preventing system freeze in some scenarios.
#     n_jobs = max(1, os.cpu_count() - 1)

#     # Use joblib's Parallel and delayed to parallelize the processing
#     Parallel(n_jobs=n_jobs)(delayed(process_experiment)(exp_name)
#     for exp_name in EXPERIMENT_NAMES)

#     # 'results' will now be a list of the results from each call to process_experiment
#     # You can process these results further if required


def main():
    """Process all experiments."""
    for exp_name in EXPERIMENT_NAMES:
        print(exp_name)
        process_experiment(exp_name, PARAMS)


if __name__ == '__main__':
    main()
