#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour le prétraitement des images, la détection des particules avec trackpy,
et le suivi des particules en parallélisant le traitement des images.
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
import imageio.v2 as imageio
import trackpy as tp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc  # Garbage Collector interface
import matplotlib.pyplot as plt

# Paramètres consolidés pour le traitement d'images et la détection des particules
PARAMS = {
    # Paramètres de préparation d'images
    'GaussianBlur': (5, 5),
    'sigmaX': 10,
    'sigmaY': 10,
    'threshold': 1,
    'percentile': 20,
    'lenght_study': 50,  # Pour découper la manipulation en nombre de frames pour l'étude
    'smoothing_size': None,
    'invert': True,
    'preprocess': True,
    'characterize': True,
    'filter_before': None,
    'filter_after': None,
    # Paramètres de manipulation
    'pixel_size': 0.637,
    'frame_interval': 15,
    'long_time': False,
    'max_frame': 340,
    'min_frames': 150,
    'topn': 500,
    # Paramètres de détection de particules
    'diameter': 15,
    'max_displacement': 30,
    'search_range': 30,
    'minmass': 500,
    'max_size': 30,
    'separation': 20,
    'noise_size': 3,
    'max_iterations': 15,
    'memory': 5,
    'engine': 'auto',
    # Chemins et formats
    'remove_exts': ['.jpg', '.svg', '.hdf5', '.png'],
    'data_dir': '/Users/souchaud/Desktop/A_Analyser/CytoOne_HL5_10x/',
    'output_dir': '/Users/souchaud/Desktop/Analyses/CytoOne_HL5_10x_new_param/'
}

# Fonction pour le prétraitement et la localisation des particules dans une seule image
def preprocess_and_locate(tiff_file, params):
    """
    Applique le prétraitement sur une image et utilise trackpy pour localiser les particules.
    
    Args:
        tiff_file (str): Chemin vers le fichier TIFF à traiter.
        params (dict): Dictionnaire des paramètres pour le prétraitement et la localisation.
    
    Returns:
        pandas.DataFrame: DataFrame contenant les résultats de la localisation des particules.
    """
    # Charger et prétraiter l'image
    frame = imageio.imread(tiff_file)
    blurred = ndimage.median_filter(frame, size=8)
    blurred = cv2.GaussianBlur(blurred, params['GaussianBlur'], 0)
    
    # Utiliser trackpy pour localiser les particules
    particles = tp.locate(blurred, diameter=params['diameter'], minmass=params['minmass'])
    
    return particles

# Fonction pour nettoyer le répertoire de sortie
def clean_directory(dir_path):
    """
    Supprime tous les fichiers avec les extensions spécifiées dans le répertoire donné.
    
    Args:
        dir_path (str): Chemin du répertoire à nettoyer.
    """
    for file in os.listdir(dir_path):
        if file.endswith(tuple(PARAMS['remove_exts'])):
            os.remove(os.path.join(dir_path, file))

# Fonction principale pour traiter une expérience
def process_experiment(exp_name, params):
    """
    Traite une seule expérience en prétraitant les images, en détectant les particules,
    et en effectuant le suivi des particules.
    
    Args:
        exp_name (str): Nom de l'expérience à traiter.
        params (dict): Dictionnaire des paramètres pour le traitement.
    """
    output_path = os.path.join(params['output_dir'], exp_name)
    print(f"output_path : {output_path}")
    exp_name_solo = exp_name.split('/', 1)[0]
    print(f"exp name : {exp_name}")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)
    clean_directory(output_path)
    
    experiment_data_dir = os.path.join(params['data_dir'], exp_name)
    tiff_files = sorted(
        glob.glob(os.path.join(experiment_data_dir, "*.tif")),
        key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
    )

    # Traiter chaque image en parallèle
    particles_results = []

    with ProcessPoolExecutor() as executor:
        # Soumettre toutes les tâches
        futures = {executor.submit(preprocess_and_locate, tiff_file, params): tiff_file for tiff_file in tiff_files}

        # Utiliser tqdm pour afficher la progression du traitement des futures
        for future in tqdm(as_completed(futures), total=len(futures), desc="Traitement des images"):
            tiff_file = futures[future]
            try:
                particles = future.result()
                # Extraire le numéro de la frame directement à partir de l'ordre de tiff_file dans la liste
                frame_number = tiff_files.index(tiff_file)
                particles['frame'] = frame_number
                particles_results.append(particles)
            except Exception as exc:
                print(f"Erreur lors du traitement de {tiff_file}: {exc}")

                
    # with ProcessPoolExecutor() as executor:
    #     future_to_particles = {
    #         executor.submit(preprocess_and_locate, tiff_file, params): tiff_file for tiff_file in tiff_files
    #     }
    #     for future in as_completed(future_to_particles):
    #         tiff_file = future_to_particles[future]
    #         try:
    #             particles = future.result()
    #             # Ajouter le numéro de la frame basé sur l'ordre du fichier dans tiff_files
    #             frame_number = tiff_files.index(tiff_file)
    #             particles['frame'] = frame_number
    #             particles_results.append(particles)
    #         except Exception as exc:
    #             print(f"Erreur lors du traitement de {tiff_file}: {exc}")

    if particles_results:
        all_features = pd.concat(particles_results, ignore_index=True)
        all_features.sort_values(by='frame', inplace=True)
        
        # Effectuer le suivi des particules
        trajectories = tp.link_df(
            all_features,
            search_range=params['search_range'],
            memory=params['memory']
        )
        
        filtered_trajectories = tp.filter_stubs(trajectories, params['min_frames'])
        filtered_trajectories.to_hdf(os.path.join(output_path, 'filtered.hdf5'), 'table')
        
        # Générer et sauvegarder des figures si nécessaire
        # ...
        
    else:
        print("Aucun résultat à concaténer. Vérifiez les erreurs précédentes.")
    
    gc.collect()

def main():
    """
    Fonction principale pour traiter toutes les expériences.
    """
    EXPERIMENT_NAMES = [f + '/mosaic/' for f in os.listdir(PARAMS['data_dir'])
                    if os.path.isdir(os.path.join(PARAMS['data_dir'], f))]
    print(EXPERIMENT_NAMES)
    for exp_name in EXPERIMENT_NAMES:
        process_experiment(exp_name, PARAMS)

if __name__ == '__main__':
    main()