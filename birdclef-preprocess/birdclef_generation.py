import os
import sys
import csv
import shutil
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
import matplotlib.pyplot as plt

import torch
import random
import numpy as np

from birdclef_generation_utils import copy_folder_structure, mix_audio_files, gain, gaussian_noise, time_split, perform_all_aug, generate_pt_files, generate_csv_files

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



print("\n\n#####################################################")
print("BirdCLEF-2023             (Preprocessing the Dataset)")
print("#####################################################\n\n")

dataset_path = "birdclef-preprocess/"
print(f"Dataset Path: {dataset_path}")

if not os.path.exists(dataset_path + "birdclef-2023.zip"):
    print(f"File 'birdclef-2023.zip' could not be found in the dataset directory: '{dataset_path}/birdclef-2023.zip'")
    print(f"Please download the file from the following link and place it in in the dataset directory: '{dataset_path}/birdclef-2023.zip'")
    print("BirdCLEF-2023 Dataset Link =  https://www.kaggle.com/c/birdclef-2023/data")
    exit(0)
else:
    print(f"Dataset 'birdclef-2023.zip' found!") 
    
os.system(f"mkdir {os.path.join(dataset_path, 'birdclef2023-dataset')}")
os.system(f"mkdir {os.path.join(dataset_path, 'birdclef2023-dataset', 'extracted_birdclef_dataset')}")
os.system(f"mkdir {os.path.join(dataset_path, 'birdclef2023-dataset', 'audio_files')}")
os.system(f"mkdir {os.path.join(dataset_path, 'birdclef2023-dataset', 'pt_files')}")
os.system(f"unzip {os.path.join(dataset_path, 'birdclef-2023.zip')} -d {os.path.join(dataset_path, 'birdclef2023-dataset', 'extracted_birdclef_dataset')}")

dataset_path = os.path.join(dataset_path, 'birdclef2023-dataset')
print(f"Dataset extracted to: '{dataset_path}'")

extracted_dataset_path = os.path.join(dataset_path, "extracted_birdclef_dataset")

source_directory = os.path.join(extracted_dataset_path, 'train_audio')
os.system(f"cp -r {source_directory} {os.path.join(dataset_path, 'audio_files', 'original')}")

destination_directory = os.path.join(dataset_path, 'audio_files', 'augmented')
copy_folder_structure(source_directory, destination_directory)

destination_directory = os.path.join(dataset_path, 'pt_files', 'original')
copy_folder_structure(source_directory, destination_directory)

destination_directory = os.path.join(dataset_path, 'pt_files', 'augmented')
copy_folder_structure(source_directory, destination_directory)


source_directory = os.path.join(extracted_dataset_path, 'train_audio')
destination_directory = os.path.join(dataset_path, 'pt_files', 'augmented')
copy_folder_structure(source_directory, destination_directory)


root_directory = os.path.join(dataset_path, 'audio_files', 'original')


# dst_root = "./dataset/augmented_2/"
dst_root = os.path.join(dataset_path, 'audio_files', 'augmented')

df = pd.read_csv( os.path.join(extracted_dataset_path, 'train_metadata.csv'))
bird_dist_dict = {}

for i in range(len(df)):
    # print(df.loc[i, "primary_label"])
    temp_label = df.loc[i, "primary_label"]
    if temp_label in bird_dist_dict:
      bird_dist_dict[temp_label] = bird_dist_dict[temp_label] + 1
    else:
      bird_dist_dict[temp_label] = 1


folders_list = os.listdir(root_directory)
sorted_folders_list = sorted(folders_list)

for label, folder in enumerate(sorted_folders_list):
    files_list = os.listdir(os.path.join(root_directory, folder))
    for file in tqdm(files_list, desc=f"Processing {folder}", unit="file"):
        file_path = os.path.join(root_directory, folder, file)
        dst_path = os.path.join(dst_root, folder)
        if bird_dist_dict[folder] < 100:
            time_split(file_path, dst_path)
        else:
            shutil.copy(file_path, dst_path)
            
            
dst_root = os.path.join(dataset_path, 'audio_files', 'augmented')


folders_list = os.listdir(dst_root)
sorted_folders_list = sorted(folders_list)

for label, folder in enumerate(sorted_folders_list):
    files_list = os.listdir(os.path.join(dst_root, folder))
    num_samples = len(files_list)

    if num_samples >= 100:
        continue
    else:
        for file in tqdm(files_list, desc=f"Processing {folder}", unit="file"):
            file_path = os.path.join(dst_root, folder, file)
            dst_path = os.path.join(dst_root, folder)

            if (num_samples * 8) > 100:
                if random.random() <= (100 / (num_samples * 8)):
                    perform_all_aug(file_path, dst_path)
            else:
                perform_all_aug(file_path, dst_path)
                
                
root_directory = os.path.join(dataset_path, 'audio_files', 'original')
dst_root = os.path.join(dataset_path, 'pt_files', 'original')

generate_pt_files(root_directory, dst_root)


root_directory = os.path.join(dataset_path, 'audio_files', 'augmented')
dst_root = os.path.join(dataset_path, 'pt_files', 'augmented')

generate_pt_files(root_directory, dst_root)


root_directory = os.path.join(dataset_path, 'pt_files', 'original')
output_csv = os.path.join(dataset_path, 'pt_files', 'original.csv')

generate_csv_files(root_directory=root_directory, output_csv=output_csv)


root_directory = os.path.join(dataset_path, 'pt_files', 'augmented')
output_csv = os.path.join(dataset_path, 'pt_files', 'augmented.csv')

generate_csv_files(root_directory=root_directory, output_csv=output_csv)