#!/usr/bin/env python3.
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:58:17 2023

@author: souchaud
"""
import os
import sys
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import warnings
import numpy as np
from collections import defaultdict
from colorama import init, Fore, Style
sys.path.append('/Users/souchaud/Desktop/Dictyostelium/Motility/tracking_and_analyze')
import functions_analyze_bis as lib
init(autoreset=True)
# from matplotlib.cm import ScalarMappable
# import pdb; pdb.set_trace()
# warnings.simplefilter("always")  # This will always display warnings
# warnings.simplefilter('error', RuntimeWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)


# In[VARIABLE DEFINITION]
# set initial time
INITIAL_TIME = time.time()

# experiment parameters
TIME_FRAME = 15  # 75
SIZE_PIX = 1.2773  # 1.634  # 4.902
FPS = 1/TIME_FRAME

# number of frame kept
N_FRAME = 1

# nber hours of stydy:
LONG_TIME = False

# Study parameters
ROLLING_MEAN = False
PIXELISATION = False
TIME_FRAME_STUDY = False
DRIFT = True

# plot parameters
IMG_TYPE = 'jpg'
ALPHA = 0.5
LINEWIDTH = 0.1
COLOR_SUP = 'blue'
COLOR_INF = 'red'
color_sup_inf = (COLOR_SUP, COLOR_INF)

# % de présences de la particules sur le total de frame étudiées

FRAME_PARTICULE = 1

# ##########
# % de présences des courbes dans les frames

FRAME_PARTICULE = 0.8

# ##########################   GENERAL PATH   #################################
GENERAL_PATH = '/Users/souchaud/Desktop/Analyses/'
# # Pathway to the experiments
PATHWAY_EXPERIMENT = []

# ##########################  EXPERIMENT PATH  ################################

# # # # ######### CONDITION: ################
# CONDITION = 'CytoOne_SorC_longtime'
# CONDITION = 'NonT_SorC_longtime'
CONDITION = 'CytoOne_HL5_longtime'
# CONDITION = 'ASMOT035_fiji'
# # # #####################################


if len(PATHWAY_EXPERIMENT) == 0:
    PATHWAY_EXPERIMENT = [f for f in os.listdir(GENERAL_PATH + CONDITION) if
                          os.path.isdir(os.path.join(GENERAL_PATH + CONDITION, f))]
# ##########################   Path Exp final  ###############################

PATHWAY_EXPERIMENT = [f'{GENERAL_PATH}{CONDITION}/' +
                      elem + '/mosaic/' for elem in PATHWAY_EXPERIMENT]

# ##########################   Path to Save pic  ##############################

path_save_pic = f'{GENERAL_PATH}résultats_{CONDITION}_ALL_OK_x5_15s/'

# création d'un dossier spécific d'enregistrement.
if not os.path.exists(path_save_pic):
    os.mkdir(path_save_pic)
os.chdir(path_save_pic)
# In[Lecture des données expérimentales]
DATA = lib.read_hdf5_all_features(pathway_experiment=PATHWAY_EXPERIMENT,
                                  condition=CONDITION,  nbr_frame_min=200, memory=10,
                                  search_range=15, adaptive_stop=3,
                                  link_strategy='auto', neighbor_strategy='KDTree',
                                  drift=DRIFT)
# DATA = DATA[DATA['frame'] < 240]
# In[Calcul des courbes à 80% des frames]
print("before all procedure, we have: ")
print(DATA.groupby('experiment')['particle'].nunique())


def select_data(DATA, nbr_frame_min):
    """
    Select the datas we want to study.

    Parameters
    ----------
    DATA : DataFrame
        Trajectories.
    nbr_frame_min : int

    Returns
    -------
    DATA : DataFrame
        trajectories filtred.

    """
    # On compte le nombre de frames pour chaque particule dans chaque expérience
    counts = DATA.groupby(['experiment', 'particle']).size()
    # On fait la liste des particles que l'on souhaite garder, cad qui sont suivies
    # sur au moins "nbr_frame_min" images.
    particles_to_keep = counts[counts >= nbr_frame_min].reset_index()
    # On récupères les données de ces particules uniquement.
    DATA = DATA.merge(particles_to_keep, on=['experiment', 'particle'])

    # On compte le nombre de frames pour chaque expérience
    total_frames_per_experiment = DATA.groupby('experiment')['frame'].nunique()

    # On calcule le seuil pour chaque expérience (60% du nombre total de frames)
    threshold = total_frames_per_experiment * 0.6
    threshold = threshold.astype(int)

    # On garde uniquement les particules qui ont un nombre de frames supérieur
    # au seuil pour chaque expérience : c'est un deuxième seuil.
    filtered_particles = counts[
        counts > threshold.reindex(counts.index, level=0)].reset_index()

    # On filtre le DataFrame original pour garder uniquement les particules sélectionnées
    DATA = DATA.merge(filtered_particles, on=['experiment', 'particle'])

    # # ## JE DECIDE DE GARDER UNIQUEMENT LES EXPERIENCES AVEC PLUS DE X FRAMES ####
    # # Step 1: Identify experiments with max frame > 500
    # valid_experiments = DATA.groupby('experiment')['frame'].max()
    # valid_experiments = valid_experiments[valid_experiments > nbr_frame_min].index

    # # Step 2: Filter the DataFrame to keep rows associated with those experiments
    # DATA = DATA[DATA['experiment'].isin(valid_experiments)]

    return DATA


# let's select the datas we want to study. For long time, we want to keep
# experiment with at least 350 frames. This could be change easely.


# DATA = select_data(DATA=DATA, nbr_frame_min=200)
# print('Nbr of particles after selecting datas function : ', DATA['particle'].nunique())


def find_swaps_with_return(data_frame):
    """
    Find and delete suspicious movements.

    Identify and return the indices of rows in a DataFrame that represent suspicious
    movements in particle trajectories, indicative of swaps. These suspicious movements
    are those where a particle makes a large jump, significantly more than the average
    movement in the experiment, and then returns to a location near its original position.

    Parameters
    ----------
    data_frame (pd.DataFrame): A DataFrame containing the particle tracking data.
        It must include the columns 'particle', 'frame', 'x', and 'y', representing
        the particle identifier, the frame number, and the x and y coordinates
        of the particle, respectively

    Returns
    -------
    list: Indices of the rows in the original DataFrame where suspicious movements,
        potentially indicative of tracking swaps, occur.

    Note:
    - The function detects swaps by identifying large jumps that return close to
      the original location.
    - The definition of 'close' should be adjusted based on the context of the experiment and the
      expected behavior of the particles.
    """
    # Placeholder for the indices of suspicious movements
    swap_indices = []

    # Calculate movement distances and keep original index to reference later
    data_frame['dx'] = data_frame.groupby('particle')['x'].diff()
    data_frame['dy'] = data_frame.groupby('particle')['y'].diff()
    data_frame['dist'] = np.sqrt(data_frame['dx']**2 + data_frame['dy']**2)

    # Calculate the mean and standard deviation for the entire experiment
    mean_dist_experiment = data_frame['dist'].mean()
    std_dist_experiment = data_frame['dist'].std()

    # Group by particle to analyze each trajectory
    grouped = data_frame.groupby('particle')

    for particle, group in grouped:
        group = group.sort_values(by='frame')  # ensure the group is sorted by frame

        # Check each point in the trajectory
        for i in range(1, len(group) - 1):
            # We skip the first and last points, as they cannot form a "return" trajectory

            prev_point = group.iloc[i - 1]
            current_point = group.iloc[i]
            next_point = group.iloc[i + 1]

            # Calculate the distances for the potential out-and-back
            dist_out = np.sqrt((current_point['x'] - prev_point['x'])**2 +
                               (current_point['y'] - prev_point['y'])**2)
            dist_back = np.sqrt((next_point['x'] - current_point['x'])**2 +
                                (next_point['y'] - current_point['y'])**2)

            # Check if the "out" distance is large enough to be suspicious
            if dist_out > mean_dist_experiment + 2 * std_dist_experiment:
                # Now check if the "back" distance is small enough to indicate a return
                # The threshold for "returning close" can be adjusted.
                # Here we use the mean distance for the experiment.
                if dist_back < mean_dist_experiment:
                    # This trajectory is suspicious. Save the index of the "out" point.
                    swap_indices.append(current_point.name)  # 'name' of the row is its original idx

    # Return the list of indices corresponding to suspicious movements
    return swap_indices


# # Get the indices of rows to drop
# to_drop = find_swaps_with_return(DATA)

# print(len(to_drop), " movements to delete there while the run is too much.")

# # # Drop the rows from DATA
# DATA = DATA.drop(to_drop)
DATA.reset_index(inplace=True)
# In[Filter on mass]
if 'mass' in DATA.columns:
    mask = DATA.groupby('particle')['mass'].transform('mean') >= 1000
    DATA = DATA[mask]
if 'level_0' in DATA.columns:
    DATA = DATA.drop('level_0', axis=1)
DATA.reset_index(inplace=True)
# In[Compute some datas as instant displacement /speed / centering trajectories ...]

DATA = lib.vit_instant_new(traj=DATA, lag_time=TIME_FRAME, pix_size=SIZE_PIX, triage=1)


# 1. Calculer la différence entre les valeurs consécutives de frame
DATA['frame_diff'] = DATA['frame'].diff().fillna(0)

# 2. Calculer la limite maximale pour displacement [pix]
DATA['max_displacement'] = DATA['frame_diff'] * 5

# 3. Créer un masque pour les lignes où displacement [pix] dépasse max_displacement
mask = DATA['displacement [pix]'] > DATA['max_displacement']

# 4. Supprimer les lignes identifiées par ce masque
DATA = DATA[~mask]


mask = DATA['displacement [pix]'] > float(6)
DATA.drop(DATA[mask].index, inplace=True)
DATA.dropna(inplace=True)
DATA = lib.vit_instant_new(traj=DATA, lag_time=TIME_FRAME, pix_size=SIZE_PIX, triage=1)

# DATA = DATA[DATA['displacement [pix]'] < 5]
DATA = lib.center(traj=DATA)

print("\n"*2)
print(f"Le temps de lecture et de préparation des données pour la condition {CONDITION} est : ",
      (time.time() - INITIAL_TIME)/60, 'min')
print("\n"*2)

# In[Compute the DATAS according to some parameters]
if ROLLING_MEAN:
    DATA = lib.rolling_mean(datas=DATA, roll=3)
if PIXELISATION:
    DATA = lib.pixelisation(datas=DATA, size_pix=SIZE_PIX)
if TIME_FRAME_STUDY:
    DATA, TIME_FRAME = lib.keep_nth_image(traj=DATA, n=N_FRAME, time_frame=TIME_FRAME)

# #############################################################################
# In[Calculation of total and cumulative displacement]
# #############################################################################

DATA, start_end = lib.length_displacement(traj=DATA, size_pix=SIZE_PIX)

# #############################################################################
# In[Recalcul du max displacement]
# ###################Erasing the suspicious displacements #####################
grouped_data = DATA.groupby('particle')
# Obtenir la valeur maximale de 'displacement' pour chaque groupe
max_displacements = SIZE_PIX*grouped_data['displacement [pix]'].max()
# Sélectionner les groupes dont la valeur maximale de 'displacement' est supérieure à 10
selected_particles = max_displacements.loc[max_displacements > 50].index.tolist()
bool_mask = DATA['particle'].isin(selected_particles)
DATA_HIGH_DISP = DATA[bool_mask]
if len(DATA_HIGH_DISP) > 0:
    lib.plot_msd(msd=tp.imsd(traj=DATA_HIGH_DISP, mpp=SIZE_PIX, fps=FPS),
                 fps=FPS, name='MSD with HIGHT DISP (sup at 10)', color_plot=COLOR_SUP,
                 save=True, pathway_saving=path_save_pic, alpha=ALPHA, linewidth=LINEWIDTH)
# Erasing the spurious traectories with too high displacement
DATA = DATA[~bool_mask]

# In[MSD computation]
IMSD = tp.imsd(traj=DATA[DATA['frame'] < 240],
               mpp=SIZE_PIX, fps=FPS,
               max_lagtime=200, statistic='msd',
               pos_columns=None)

# In[Exclusion des particules de MSD constante]
print("\n"*2)
print("Nombre de particules étudiées avant tri sur MSD : ", DATA['particle'].nunique())

DATA, data_exclues, IMSD = lib.exclusion_on_msd(datas=DATA, imsd=IMSD,
                                                level=2)

DATA.reset_index(inplace=True, drop=True)
print("\n"*2)
print("Nombre de particules étudiées au final : ", DATA['particle'].nunique())

# #############################################################################
# #############################################################################
# We now Consider having all the good particles and all good datas.
# #############################################################################
# #############################################################################

# In[Plot all the trajectories]
fig, axis = plt.subplots(figsize=(10, 10))
# Assurer une échelle égale pour les axes
axis.set_aspect('equal', 'box')
plt.title('Trajectories after suspicious particles')
tp.plot_traj(DATA, label=(False))
plt.show()
fig.savefig(path_save_pic +
            'Trajectories after removing suspicious particles.jpg', format='jpg')
# In[Plot all the trajectories]
plot_exp = DATA.groupby('experiment')
for exp_name, exp_data in plot_exp:
    fig, axis = plt.subplots(figsize=(10, 10))
    # Assurer une échelle égale pour les axes
    axis.set_aspect('equal', 'box')
    plt.title(f'Trajectories after suspicious particles for {exp_name}')
    tp.plot_traj(exp_data, label=(False))
    plt.show()
# In[traj clustering with fit and defining a cutoff]
LAG_TIME_FIT = 5
# Compute et plot the director factor of the imsd

COEF_INF, COEF_SUP, PART_COEF_INF, PART_COEF_SUP, CUTOFF =\
    lib.traj_clustering_with_fit_cutoff(DATA[DATA['frame'] < 240], imsd=IMSD, hist=True,
                                        lag_time_fit=LAG_TIME_FIT,
                                        micronperpixel=SIZE_PIX,
                                        fps=FPS, binsize=600,
                                        peak_height=50, peak_width=1,
                                        save=True, pathway_fig=path_save_pic,
                                        name='all the experiment', img_type="jpg",
                                        plot=True, color_sup_inf=color_sup_inf,
                                        cutoff_default=0.5
                                        )

# DATA_INF, DATA_SUP, IMSD_INF, IMSD_SUP,
DATA_INF = DATA[DATA['particle'].isin(PART_COEF_INF)]
DATA_SUP = DATA[DATA['particle'].isin(PART_COEF_SUP)]
IMSD_INF = IMSD.loc[:, IMSD.columns.isin(PART_COEF_INF)]
IMSD_SUP = IMSD.loc[:, IMSD.columns.isin(PART_COEF_SUP)]

# In[Print datas]
print("\n"*2)
if (len(DATA_SUP) != 0) & (len(DATA_INF) != 0):
    print("Résultats après premier tri cellulaire")
    print("proportion de hight/low : ",
          DATA_SUP['particle'].nunique()
          / (DATA_INF['particle'].nunique() + DATA_SUP['particle'].nunique()))
    print("Cutoff : ", round(CUTOFF, 2))

    print("Mean speed for high slopes : ",
          round(DATA_SUP.groupby('particle')['VitInst [um/min]'].mean().mean(), 2), " +/- ",
          round(DATA_SUP.groupby('particle')['VitInst [um/min]'].mean().std(), 2))

    print("Mean speed for low slopes : ",
          round(DATA_INF.groupby('particle')['VitInst [um/min]'].mean().mean(), 2), " +/- ",
          round(DATA_INF.groupby('particle')['VitInst [um/min]'].mean().std(), 2))

else:
    print("Only hight value particles")
print("\n"*2)
# %%
# print('Number of particle followed for each experiment : ')
# print(DATA.groupby('experiment')['particle'].nunique())

series_y = DATA.groupby('experiment')['y'].max().div(2048).apply(math.ceil).astype(int)
series_x = DATA.groupby('experiment')['x'].max().div(2048).apply(math.ceil).astype(int)

# Multiplication élément par élément entre les deux séries
result = series_y * series_x

for exp_name, group in DATA.groupby('experiment'):
    message = ""
    if int(result[exp_name]*50*(1 + 0.333)) < group['particle'].nunique():
        message = "TOO DENSE?"
        COLOR = Fore.RED
    elif int(result[exp_name]*50*(1 - 0.333)) > group['particle'].nunique():
        message = "ENOUGH?"
        COLOR = Fore.RED
    else:
        COLOR = Fore.GREEN
    output = "{}: {} +/- {} um/min: {} < {}{}{}{} < {} {} {} {}x{}".format(
        exp_name,
        round(group.groupby('particle')['VitInst [um/min]'].mean().mean(), 1),
        round(group.groupby('particle')['VitInst [um/min]'].mean().std(), 1),
        int(result[exp_name]*50*(1-0.333)),
        COLOR, Style.BRIGHT,
        group['particle'].nunique(),
        Style.RESET_ALL,
        int(result[exp_name]*50*(1+0.333)),
        message,
        "size:",
        series_x[exp_name],
        series_y[exp_name],
        )
    print(output)

    if message == "soooo":
        # Supprimer toutes les données de l'expériment
        DATA = DATA[DATA['experiment'] != exp_name]

        print("\n"*2)
        print("Donc on a :", DATA['particle'].nunique(), "particle etudiees après tout ça mdrrr")
        # In[MSD computation re-computation]
        IMSD = tp.imsd(traj=DATA[DATA['frame'] < 240],
                       mpp=SIZE_PIX, fps=FPS,
                       max_lagtime=200, statistic='msd',
                       pos_columns=None)
        # In[traj clustering with fit and defining a cutoff]
        LAG_TIME_FIT = 5
        # Compute et plot the director factor of the imsd

        COEF_INF, COEF_SUP, PART_COEF_INF, PART_COEF_SUP, CUTOFF =\
            lib.traj_clustering_with_fit_cutoff(DATA, imsd=IMSD, hist=True,
                                                lag_time_fit=LAG_TIME_FIT,
                                                micronperpixel=SIZE_PIX,
                                                fps=FPS, binsize=600,
                                                peak_height=15, peak_width=1,
                                                save=True, pathway_fig=path_save_pic,
                                                name='all the experiment', img_type="jpg",
                                                plot=True, color_sup_inf=color_sup_inf,
                                                cutoff_default=0.75
                                                )

        # DATA_INF, DATA_SUP, IMSD_INF, IMSD_SUP,
        DATA_INF = DATA[DATA['particle'].isin(PART_COEF_INF)]
        DATA_SUP = DATA[DATA['particle'].isin(PART_COEF_SUP)]
        IMSD_INF = IMSD.loc[:, IMSD.columns.isin(PART_COEF_INF)]
        IMSD_SUP = IMSD.loc[:, IMSD.columns.isin(PART_COEF_SUP)]

        print("\n"*2)
        if (len(DATA_SUP) != 0) & (len(DATA_INF) != 0):
            print("proportion de hight/low : ",
                  DATA_SUP['particle'].nunique()
                  / (DATA_INF['particle'].nunique() + DATA_SUP['particle'].nunique()))
            print("Cutoff : ", round(CUTOFF, 2))

            print("Mean speed for high slopes : ",
                  round(DATA_SUP.groupby('particle')['VitInst [um/min]'].mean().mean(), 2), " +/- ",
                  round(DATA_SUP.groupby('particle')['VitInst [um/min]'].mean().std(), 2))

            print("Mean speed for low slopes : ",
                  round(DATA_INF.groupby('particle')['VitInst [um/min]'].mean().mean(), 2), " +/- ",
                  round(DATA_INF.groupby('particle')['VitInst [um/min]'].mean().std(), 2))

        else:
            print("Only hight value particles")
        print("\n"*2)

# In[clustering for each experiment to be sure]
for exp_name, group in DATA.groupby('experiment'):

    IMSD_temp = tp.imsd(traj=group[group['frame'] < 240],
                        mpp=SIZE_PIX, fps=FPS,
                        max_lagtime=200, statistic='msd',
                        pos_columns=None)

    LAG_TIME_FIT = 5
    # Compute et plot the director factor of the imsd

    COEF_INF_temp, COEF_SUP_temp, PART_COEF_INF_temp, PART_COEF_SUP_temp, CUTOFF_temp =\
        lib.traj_clustering_with_fit_cutoff(group, imsd=IMSD_temp, hist=True,
                                            lag_time_fit=LAG_TIME_FIT,
                                            micronperpixel=SIZE_PIX,
                                            fps=FPS, binsize=100,
                                            peak_height=15, peak_width=1,
                                            save=True, pathway_fig=path_save_pic,
                                            name=f'{exp_name}', img_type="jpg",
                                            plot=True, color_sup_inf=color_sup_inf,
                                            cutoff_default=0.75
                                            )

    # DATA_INF, DATA_SUP, IMSD_INF, IMSD_SUP,
    DATA_INF_temp = group[group['particle'].isin(PART_COEF_INF_temp)]
    DATA_SUP_temp = group[group['particle'].isin(PART_COEF_SUP_temp)]
    IMSD_INF_temp = IMSD_temp.loc[:, IMSD_temp.columns.isin(PART_COEF_INF_temp)]
    IMSD_SUP_temp = IMSD_temp.loc[:, IMSD_temp.columns.isin(PART_COEF_SUP_temp)]

    # Grouper les données par 'frame' et calculer la moyenne de 'VitInst [um/min]'
    mean_VitInst_per_frame_manip = group.groupby('frame')['VitInst [um/min]'].mean()
    mean_VitInst_per_frame_manip = mean_VitInst_per_frame_manip.rolling(10).mean().dropna()

    lib.plot_datas(x_values=mean_VitInst_per_frame_manip.index,
                   y_values=mean_VitInst_per_frame_manip.values,
                   title=f'Mean VitInst [um/min] per Frame for {exp_name}',
                   x_label='Frame', y_label='Mean VitInst [um/min]',
                   x_lim=[0, 1000], y_lim=[0, 10], save=True,
                   path_save_pic=path_save_pic, img_type="jpg")

    proportion = 1
    if (len(DATA_SUP_temp) != 0) & (len(DATA_INF_temp) != 0):
        somme = (DATA_INF_temp['particle'].nunique() + DATA_SUP_temp['particle'].nunique())
        proportion = DATA_SUP_temp['particle'].nunique() / somme

    message = ""
    output = "{}-> MeanSpeed: {} +/- {} um/min with {}% of hight particles".format(
        exp_name,
        round(group['VitInst [um/min]'].mean(), 1),
        round(group['VitInst [um/min]'].std(), 1),
        round(proportion*100, 3)
        )
    print(output)
# In[Mean speed]
# Grouper les données par 'frame' et calculer la moyenne de 'VitInst [um/min]'
mean_VitInst_per_frame = DATA.groupby('frame')['VitInst [um/min]'].mean()
mean_VitInst_per_frame = mean_VitInst_per_frame.rolling(10).mean().dropna()

lib.plot_datas(x_values=mean_VitInst_per_frame.index,
               y_values=mean_VitInst_per_frame.values,
               title='Mean VitInst [um/min] per Frame',
               x_label='Frame', y_label='Mean VitInst [um/min]',
               x_lim=[0, 1000], y_lim=[0, 10], save=True,
               path_save_pic=path_save_pic, img_type="jpg")

# Mean speed for hight particles
# Grouper les données par 'frame' et calculer la moyenne de 'VitInst [um/min]'
mean_VitInst_per_frame_SUP = DATA_SUP.groupby('frame')['VitInst [um/min]'].mean()
mean_VitInst_per_frame_SUP = mean_VitInst_per_frame_SUP.rolling(10).mean().dropna()

lib.plot_datas(x_values=mean_VitInst_per_frame_SUP.index,
               y_values=mean_VitInst_per_frame_SUP.values,
               title='Mean VitInst [um/min] per Frame for particle sup',
               x_label='Frame', y_label='Mean VitInst [um/min]',
               x_lim=[0, 1000], y_lim=[0, 10], save=True,
               parameters_plot={'color': COLOR_SUP,
                                'linewidth': 0.5,
                                'linestyle': 'solid',
                                },
               path_save_pic=path_save_pic, img_type="jpg")

# Mean speed for hight particles
# Grouper les données par 'frame' et calculer la moyenne de 'VitInst [um/min]'
mean_VitInst_per_frame_INF = DATA_INF.groupby('frame')['VitInst [um/min]'].mean()
mean_VitInst_per_frame_INF = mean_VitInst_per_frame_INF.rolling(10).mean().dropna()
lib.plot_datas(x_values=mean_VitInst_per_frame_INF.index,
               y_values=mean_VitInst_per_frame_INF.values,
               title='Mean VitInst [um/min] per Frame for particle inf',
               x_label='Frame', y_label='Mean VitInst [um/min]',
               x_lim=[0, 1000], y_lim=[0, 10], save=True,
               parameters_plot={'color': COLOR_INF,
                                'linewidth': 0.5,
                                'linestyle': 'solid',
                                },
               path_save_pic=path_save_pic, img_type="jpg")

# In[Mean speed for each particle]

lib.mean_speed(traj=DATA, start_end=start_end,
               part_coef_inf=PART_COEF_INF, part_coef_sup=PART_COEF_SUP,
               pathway_saving=path_save_pic, color_sup_inf=color_sup_inf)

# In[Number of particle on each frame]

# Grouper les données par 'frame' et calculer la moyenne de 'VitInst [um/min]'
nbr_part_per_frame = DATA.groupby('frame')['particle'].nunique()

lib.plot_datas(x_values=nbr_part_per_frame.index, y_values=nbr_part_per_frame.values,
               title='Nbr particle per Frame',
               x_label='Frame', y_label='Number of particle',
               x_lim=[0, 1000], y_lim=[0, 20000], save=True,
               path_save_pic=path_save_pic, img_type="jpg")

# In[plot the trajectories centered]
lib.plot_centered_traj_inf_sup(traj=DATA, size_pix=SIZE_PIX,
                               PART_INF=PART_COEF_INF, PART_SUP=PART_COEF_SUP,
                               name='Trajectories recentered one color',
                               xlim=[-500, 500], ylim=[-500, 500],
                               color_sup_inf=(COLOR_SUP, COLOR_INF), save=True,
                               pathway_fig=path_save_pic, img_type="jpg")

lib.plot_centered_traj(traj=DATA.loc[DATA['frame'] <= 240],
                       size_pix=SIZE_PIX, save=True,
                       xlim=[-500, 500], ylim=[-500, 500],
                       pathway_fig=path_save_pic, name="all trajectories recentered")

if len(DATA_INF) > 0:
    lib.plot_centered_traj(traj=DATA_INF.loc[DATA_INF['frame'] <= 240],
                           size_pix=SIZE_PIX, save=True,
                           xlim=[-500, 500], ylim=[-500, 500],
                           linewidth=0.1, alpha=0.5,
                           pathway_fig=path_save_pic, name="inf trajectories recentered",
                           color=COLOR_INF)

if len(DATA_SUP) > 0:
    lib.plot_centered_traj(traj=DATA_SUP.loc[DATA_SUP['frame'] <= 240],
                           size_pix=SIZE_PIX, save=True,
                           xlim=[-500, 500], ylim=[-500, 500],
                           linewidth=0.1, alpha=0.5,
                           pathway_fig=path_save_pic, name="sup trajectories recentered",
                           color=COLOR_SUP,
                           )

# In[gif for centered traj]


def gif_centered_traj(data: pd.DataFrame(), path_save_pic: str,
                      linewidth: float = 0.2, alpha: float = 0.5,
                      last_frame: int = 240, gap_frame: int = 10):
    """
    Generate of a gif for the evolution of the traj.

    Parameters
    ----------
    data : pd.DataFrame()
        DESCRIPTION.
    path_save_pic : str
        DESCRIPTION.
    linewidth : float, optional
        DESCRIPTION. The default is 0.2.
    alpha : float, optional
        DESCRIPTION. The default is 0.5.
    last_frame : int, optional
        DESCRIPTION. The default is 240.
    gap_frame : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    path = path_save_pic + '/gif_centered/'
    if not os.path.exists(path):
        os.mkdir(path)
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.set(xlabel='Xc (µm)', ylabel='Yc(µm)')
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.tick_params(axis='both', which='major', labelsize=20)

    for i in range(1, last_frame, gap_frame):
        # DATA_centered = DATA.loc[DATA['frame'] <= i]
        DATA_centered = data.loc[(data['frame'] >= i) & (DATA['frame'] < i + 10)]
        plt.title(f'Trajectories recentered {i}', fontsize=40,
                  fontweight="bold", fontstyle='italic', fontname="Arial")

        # Ajout des points de données de la frame courante
        tp.plot_traj(DATA_centered, pos_columns=['Xc [pix]', 'Yc [pix]'],
                     ax=ax, color=None, linewidth=linewidth, alpha=alpha)

        # Sauvegardez la figure
        fig.savefig(path + f"Trajectories recentered frame {i}.png", format='png')


gif_centered_traj(data=DATA, path_save_pic=path_save_pic, linewidth=0.2,
                  alpha=0.5, last_frame=240, gap_frame=10)
# In[plot MSD graph]
if len(DATA_INF) > 0 and len(DATA_SUP) > 0:
    lib.plot_msd_inf_sup(msd_inf=IMSD.loc[:, PART_COEF_INF], msd_sup=IMSD.loc[:, PART_COEF_SUP],
                         fps=FPS, name="hight and of all frames in function of lag time (s)",
                         save=True, pathway_saving=path_save_pic, alpha=ALPHA,
                         linewidth=LINEWIDTH, img_type='jpg', color_sup_inf=color_sup_inf)

if len(DATA_SUP) > 0:
    lib.plot_msd(msd=IMSD.loc[:, PART_COEF_SUP],
                 fps=FPS, name='MSD of particles with high slope', color_plot=color_sup_inf[0],
                 save=True, pathway_saving=path_save_pic, alpha=ALPHA, linewidth=LINEWIDTH)
if len(DATA_INF) > 0:
    lib.plot_msd(msd=IMSD.loc[:, PART_COEF_INF],
                 fps=FPS, name='MSD of particles with low slope', color_plot=color_sup_inf[1],
                 save=True, pathway_saving=path_save_pic, alpha=ALPHA, linewidth=LINEWIDTH)

# In[Compute and plot the EMSD]

if len(DATA_INF) == 0:
    DATA_INF = pd.DataFrame()
# lib.plot_emsd(traj_inf=DATA_INF, traj_sup=DATA_SUP, size_pix=SIZE_PIX, color_sup_inf=color_sup_inf,
#               fps=FPS, max_lagtime=LAG_TIME_FIT, pathway_saving=path_save_pic)

if len(DATA_INF) > 0:
    bool_mask = IMSD_INF.iloc[0] > 3

    # Récupérer les noms de particules vérifiant la condition
    names = IMSD_INF.columns[bool_mask].tolist()

    bool_mask = DATA_INF['particle'].isin(names)

    # Récupérer les lignes correspondantes dans DATA_INF
    result = DATA_INF[bool_mask]
    if len(result) < 0:
        lib.plot_msd(msd=tp.imsd(traj=result, mpp=SIZE_PIX, fps=FPS), color_plot=color_sup_inf[1],
                     fps=FPS, name='MSD with low slope and sup at 10 at t0',
                     save=False, pathway_saving=path_save_pic, alpha=1, linewidth=0.5)

# In[GIF]
# #############################################################################
# ################################## GIF ######################################
# ############################################################################
gif = False
if gif:
    import glob
    experiment = 'ASMOT035'
    condition = 'NonT_SorC'
    picture_path = f'/Users/souchaud/Desktop/{condition}/'
    # Utilisez glob pour rechercher les chemins correspondants
    matching_paths = glob.glob(f"{picture_path}*{experiment}*")

    # Affichez les chemins trouvés
    for path in matching_paths:
        print("Chemin trouvé :", path)

    lib.gif_and_traj(data=DATA, size_pix=SIZE_PIX, condition=condition,
                     experiment=experiment,
                     pathway_experiment=matching_paths[0] + '/mosaic/',
                     pathway_saving=f'/Users/souchaud/Desktop/Analyses/gif_{experiment}/',
                     dot_size=50)

# In[Plot displacements graphs]
lib.plot_displacement_low_and_high(traj_sup=DATA_SUP, traj_inf=DATA_INF,
                                   start_end=start_end, color_sup_inf=color_sup_inf,
                                   name='all datas',
                                   save=True, pathway_saving=path_save_pic)
if len(DATA_INF) > 0:
    lib.plot_displacement(traj=DATA_INF, start_end=start_end.loc[PART_COEF_INF],
                          color_plot=color_sup_inf[1],
                          name='datas low slopes', save=True, pathway_saving=path_save_pic)

lib.plot_displacement(traj=DATA_SUP, start_end=start_end.loc[PART_COEF_SUP],
                      color_plot=color_sup_inf[0],
                      name='datas high slopes', save=True, pathway_saving=path_save_pic)

# In[Creating GIF]
# NUM_PART_DISP_MAX = DATA_INF.loc[DATA_INF['displacement'].idxmax(), 'particle']
# NUM_PART_TO_DRAW = 11032
# lib.creating_gif(datas=DATA[DATA['particle'] == NUM_PART_TO_DRAW],
#                  particle_infos=dict(DATA[DATA['particle'] == NUM_PART_TO_DRAW].iloc[0]),
#                  pathway_experiment='f/Users/souchaud/Desktop/A_analyser/',
#                  condition='CytoOne_HL5',
#                  first_frame=DATA[DATA['particle'] == NUM_PART_TO_DRAW]['frame'].iloc[0],
#                  last_frame=DATA[DATA['particle'] == NUM_PART_TO_DRAW]['frame'].iloc[-1],
#                  pathway_saving=path_save_pic + f"GIF_particle_{NUM_PART_TO_DRAW}/")
# df = DATA
# In[plot histogram displacement abs / displacement integerd
lib.histogram_directionality(traj=DATA, start_end=start_end,
                             part_coef_inf=PART_COEF_INF, part_coef_sup=PART_COEF_SUP,
                             pathway_saving=path_save_pic)
# In[plot histogram displacement abs / displacement integerd
for exp_name, exp_data in DATA.groupby('experiment'):
    lib.histogram_directionality(traj=exp_data, start_end=start_end,
                                 name=exp_name,
                                 )
# ############### WORK ON EACH HOUR ####################
# In[Analyse progrossive des msd en fonction du temps]
# #############################################################################
if LONG_TIME:
    print('go')
    from tqdm import tqdm
    particle_count = DATA.groupby('particle').size()
    particles_to_keep = particle_count[particle_count >= 300].index
    DATA_STUDY = DATA[DATA['particle'].isin(particles_to_keep)]

    coef_time_parts = {}  # defaultdict()
    for hour in tqdm(range(0, 850, 10), desc="hours"):
        # Trouver les particules qui satisfont la condition
        frame_max_by_particle = DATA_STUDY.groupby('particle')['frame'].max()
        valid_particles = frame_max_by_particle[frame_max_by_particle >= hour + 100].index.tolist()

        # Filtrer le DataFrame original en utilisant `query`
        query_string = f'particle in @valid_particles and {hour} <= frame <= {hour + 100}'
        DATA_TO_ANALYSE = DATA.query(query_string)

        # DATA_TO_ANALYSE = DATA.query(f'{hour} <= frame <= {50 + hour}')
        # testos = DATA_TO_ANALYSE['particle'].nunique()*0.05
        IMSD_TO_ANALYSE = tp.imsd(traj=DATA_TO_ANALYSE, mpp=SIZE_PIX, fps=FPS, max_lagtime=10,
                                  statistic='msd', pos_columns=None)

        COEF, PARTS = lib.traj_clustering(df=DATA_TO_ANALYSE,
                                          imsd=IMSD_TO_ANALYSE,
                                          lag_time_fit=LAG_TIME_FIT,
                                          micronperpixel=SIZE_PIX,
                                          fps=FPS,
                                          )
        coef_time_parts[hour] = {'COEF': COEF, 'PARTS': PARTS}

    # In[just save some stuff quickly]
    import pandas as pd

    data_list = []

    # Assuming coef_time_parts is your dictionary
    for hour, data in coef_time_parts.items():
        COEF = data['COEF']
        PARTS = data['PARTS']
        for part, coef in zip(PARTS, COEF):
            data_list.append({'frame': hour, 'particle': part, 'coef': coef})

    # Create DataFrame from the list of dictionaries
    DATA_COEF = pd.DataFrame(data_list)

# In[Calculate the derivate]
    HOUR = defaultdict(list)
    COEFF = defaultdict(list)
    PART = set()
    derivees = defaultdict(list)
    part_hight_derivate = []
    maxima_indices_part = defaultdict(list)
    # Pour chaque particule unique, trace une ligne avec les coefficients directeurs
    for part in DATA_COEF['particle'].unique():
        # Supprimer le dernier élément de la liste des coefficients
        coeff_list = DATA_COEF[DATA_COEF['particle'] == part]['coef'][1:-1]
        # Supprimer le dernier élément de la liste des heures
        time = DATA_COEF[DATA_COEF['particle'] == part]['frame'][1:-1]

        delta_coef = np.diff(coeff_list)
        delta_time = np.diff(time)

        HOUR[part].append(time)
        COEFF[part].append(coeff_list)

        # La dérivée discrète
        derivative = delta_coef / delta_time
        derivees[part].append(derivative)
        # Trouver les indices des maxima locaux
        maxima_indices = np.where((derivative[:-1] > derivative[1:]) &
                                  (derivative[:-1] > 0))[0] + 1
        maxima_indices_part[part].append(maxima_indices)
        # maxima_indices = maxima_indices -1
        # Supposons que seuil soit le seuil de variation
        seuil = 0.1
        high_variation_indices = np.where(np.abs(derivative) > seuil)[0] + 1
        if len(high_variation_indices) != 0:
            part_hight_derivate.append(part)
        high_variation_indices = []

    # In[Plot the deriative result with the threshold]

    plt.figure(figsize=(10, 6))
    for part in part_hight_derivate:
        temps = np.array(HOUR[part])[0]
        derivative = derivees[part][0]
        plt.plot(temps[1:], derivative, alpha=0.5, linewidth=0.5, label='Dérivée Discrète')
        plt.axhline(y=seuil, color='r', linestyle='--',  label=f'Seuil: {seuil}')
        plt.axhline(y=-seuil, color='r', linestyle='--', label=f'Seuil: {seuil}')
    plt.xlabel('Temps')
    plt.ylim([-0.3, 0.3])
    plt.ylabel('Taux de Variation')
    plt.show()

    # In[plot the coef vs time of the chosen particle]
    plt.figure(figsize=(10, 6))
    for part in part_hight_derivate:
        temps = np.array(HOUR[part])[0]
        coef = np.array(COEFF[part])[0]
        maxima_indices = list(maxima_indices_part[part][0])
        plt.plot(temps, coef, alpha=0.5, linewidth=0.5, label='Dérivée Discrète')
    plt.xlabel('Time (hours)')
    plt.xlim([0, 1000])
    plt.ylim([0, 3])
    plt.ylabel('Coefficient directeur')
    plt.title('Coefficient directeur en fonction du temps pour chaque particule')
    plt.show()

    # In[Graph mean speed vs msd]

    MEAN_SPEED = DATA.groupby('particle')['VitInst [um/min]'].median()
    MS_LIST_SUP = []
    for particle in PART_COEF_SUP:
        MS_LIST_SUP.append(MEAN_SPEED.loc[particle])
    MS_LIST_INF = []
    for particle in PART_COEF_INF:
        MS_LIST_INF.append(MEAN_SPEED.loc[particle])
    # Création du graphique
    fig, axis = plt.subplots(figsize=(20, 20))
    # axis.set_xlim([1, 15*len(COEFF)])
    # axis.set_ylim(0, 2)
    plt.scatter(MS_LIST_SUP, COEF_SUP, alpha=0.5, linewidth=1, marker='+', color='red')
    plt.scatter(MS_LIST_INF, COEF_INF, alpha=0.5, linewidth=1, marker='+', color='blue')

    plt.xlabel("Mean_speed")
    plt.ylabel("Coefficient directeur")

    plt.title("Mean speed vs coeff director")
    # Affichage du graphique
    plt.show()

    # In[plot the trajectories with color map based on a new col like instant speed of coeff]
    for part in part_hight_derivate:
        ax = lib.modified_plot_traj(DATA[DATA['particle'].isin([part])],
                                    pos_columns=['Xc [pix]', 'Yc [pix]'],
                                    colorby=True, cmap='plasma',
                                    color_column='VitInst [um/min]',
                                    v_min=0, v_max=5,
                                    mpp=SIZE_PIX, label=True,
                                    save=True, path_save_pic=path_save_pic + 'traj_speed/',
                                    img_type=IMG_TYPE)
        plt.show()

    # In[plot the trajectories with color map based on a new col like instant speed of coeff]
    DATA_COEF.rename(columns={'hour': 'frame'}, inplace=True)
    DATA_COEF = pd.merge(DATA_COEF, DATA[['frame', 'particle', 'Xc [pix]', 'Yc [pix]']],
                         how='left', on=['frame', 'particle'])
    # %%
    for part in part_hight_derivate:
        ax = lib.modified_plot_traj(DATA_COEF[DATA_COEF['particle'].isin([part])],
                                    pos_columns=['Xc [pix]', 'Yc [pix]'],
                                    colorby=True, cmap="viridis",
                                    color_column='coef',
                                    v_min=0, v_max=2,
                                    mpp=SIZE_PIX, label=True,
                                    save=True, path_save_pic=path_save_pic + 'traj_coeff_direct/',
                                    img_type=IMG_TYPE)
        plt.show()
