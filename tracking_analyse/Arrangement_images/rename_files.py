#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:15:36 2023

@author: souchaud
"""
import os


def sort_filenames(filenames):
    """
    Sort filenames based on their sequential number.
    """
    return sorted(filenames, key=lambda x: int(x.split("_")[-1].split(".")[0]))


def rename_files_in_directory(directory_path):
    """
    Rename all files in the given directory from the format '1-Pos_000_000_X'
    to 'img_00000000X_Default_000'.xs

    Parameters:
    - directory_path: str, path to the directory containing the files to be renamed.

    Returns:
    - None. The files in the directory are renamed.
    """

    # List all files in the directory
    files = sort_filenames([f for f in os.listdir(directory_path) if
                            os.path.isfile(os.path.join(directory_path, f)) and f.endswith(".tif")])

    for filename in files:
        # Check if the file follows the expected format
        if filename.startswith("1-Pos_00"):
            # Extract the sequence number from the filename
            sequence_number = int(filename.split("_")[-1].split(".")[0])

            # Construct the new filename
            new_filename = f"img_{sequence_number:09}_Default_000.tif"

            # Rename the file
            os.rename(os.path.join(directory_path, filename),
                      os.path.join(directory_path, new_filename))

            print(f"Renamed {filename} to {new_filename}")
    print("Renaming process completed.")


def add_tiff_suffix(directory_path):
    """
    Rename all files in the given directory from the format '1-Pos_000_000_X'
    to 'img_00000000X_Default_000'.xs

    Parameters:
    - directory_path: str, path to the directory containing the files to be renamed.

    Returns:
    - None. The files in the directory are renamed.
    """

    # List all files in the directory
    files = sort_filenames([f for f in os.listdir(directory_path) if
                            os.path.isfile(os.path.join(directory_path, f))])

    for filename in files:
        # Check if the file follows the expected format
        if not filename.endswith(".tif"):

            # Construct the new filename
            new_filename = filename + '.tif'

            # Rename the file
            os.rename(os.path.join(directory_path, filename),
                      os.path.join(directory_path, new_filename))

            print(f"Renamed {filename} to {new_filename}")
    print("Renaming process completed.")

# Example usage:
# rename_files_in_directory("/path/to/your/directory")


# condition = 'CytoOne_SorC'
# position_folders = f'/Users/souchaud/Desktop/A_analyser/{condition}/'

position_folders = '/Users/souchaud/Desktop/CytoOne_SorC/2022_12_19_ASMOT049_BoiteCytoOne_SorC_15s_5x_P9_AX3Chi2_t90_21c/8bits/'

directory_paths = [f for f in os.listdir(position_folders)
                   if os.path.isdir(os.path.join(position_folders, f))]

for directory_path in directory_paths:
    rename_files_in_directory(position_folders+directory_path)

    # add_tiff_suffix(position_folders + directory_path)
