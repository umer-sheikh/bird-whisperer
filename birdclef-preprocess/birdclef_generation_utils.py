import os
import sys
import csv
import shutil
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment


import torch
import random
import numpy as np
import whisper

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



def copy_folder_structure(src_dir, dest_dir):
    # Iterate through the directory tree
    for root, dirs, files in os.walk(src_dir):
        # Get the relative path from the source directory to the current directory
        relative_path = os.path.relpath(root, src_dir)
        # Construct the corresponding destination directory
        dest_path = os.path.join(dest_dir, relative_path)
        # Create the directory if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
    

def mix_audio_files(file1_path, file2_path, destination_folder, aug_name, file2_volume=-20):
    """
    Mixes two audio files, looping file 2 over file 1 and adjusting file 2's volume.
    Outputs the mixed audio in .ogg format.
    
    Parameters:
    - file1_path: Path to the first audio file (in .ogg format).
    - file2_path: Path to the second audio file (background noise) in .wav or .mp3 format.
    - destination_folder: Path to the folder where the mixed audio will be saved.
    - file2_volume: The volume adjustment for file 2 in dB. Default is -20dB.
    - aug_name: The type of augmentation being performed
    """
    # Load the audio files
    sound1 = AudioSegment.from_file(file1_path, format="ogg")
    sound2_format = "wav" if file2_path.endswith(".wav") else "mp3"
    sound2 = AudioSegment.from_file(file2_path, format=sound2_format)
    
    # Adjust volume of file 2
    sound2 = sound2 + file2_volume  # Decrease or increase the volume
    
    # Calculate how many times to loop file 2 based on the duration of file 1
    loop_count = len(sound1) // len(sound2) + 1
    
    # Loop file 2
    sound2_looped = sound2 * loop_count
    
    # Trim file 2 to the length of file 1
    sound2_looped = sound2_looped[:len(sound1)]
    
    # Overlay file 2 on file 1
    mixed = sound1.overlay(sound2_looped)
    
    # Generate the output file name and path
    file1_name = os.path.splitext(os.path.basename(file1_path))[0]
    destination_path = os.path.join(destination_folder, f"{file1_name}_{aug_name}.ogg")
    
    # Export the mixed audio in .ogg format to the destination folder
    mixed.export(destination_path, format="ogg")
    
def gain(src, dest_folder):
    audio = AudioSegment.from_ogg(src)
    gain_db = random.uniform(13, 17)  # Gain in decibels, corresponding to a factor between 1.3 and 1.7
    modified_audio = audio + gain_db  # Adjusting volume in decibels

    # primary_label = os.path.basename(dest_folder)
    file1_name = os.path.splitext(os.path.basename(src))[0]
    new_filename = f"{file1_name}_aug_gain.ogg"

    dest_file = os.path.join(dest_folder, new_filename)
    modified_audio.export(dest_file, format="ogg")
    
def gaussian_noise(src, dest_folder):
    audio = AudioSegment.from_ogg(src)
    duration = min(len(audio), 30 * 1000)  # Limit duration to 30 seconds

    # Generate Gaussian noise
    noise = np.random.normal(0, 1, duration)
    noise_audio = AudioSegment(
        noise.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

    # Combine the original audio with the noise
    noisy_audio = audio.overlay(noise_audio)

    # primary_label = os.path.basename(dest_folder)
    # new_filename = f"{primary_label}_gaussiannoise_{os.path.basename(src)}"

    file1_name = os.path.splitext(os.path.basename(src))[0]
    new_filename = f"{file1_name}_aug_gaussian.ogg"

    dest_file = os.path.join(dest_folder, new_filename)
    noisy_audio.export(dest_file, format="ogg")
    
def time_split(src, dest_folder):
    audio = AudioSegment.from_ogg(src)
    duration = len(audio)  # Duration in milliseconds
    chunk_length = 30 * 1000  # 30 seconds in milliseconds

    # primary_label = os.path.basename(dest_folder)  # Extract the label name
    filename_base = os.path.splitext(os.path.basename(src))[0]

    # file_path = "./dataset/dataset.csv"

    # data_dict = {'Index': 'value1', 'OG_Filename': og_filename, 'Segment_Filename': 'value3', 'Augment_Filename': '', 'Label_Name': label_name, 'Label_Integer': label_int}

    for i in range(0, duration, chunk_length):
        if duration - i < 10 * 1000:
            if i == 0:
                chunk_name = f"{filename_base}_seg_{1}.ogg"
                temp_dest_folder = os.path.join(dest_folder, chunk_name)
                shutil.copy(src, temp_dest_folder)

                # data_dict['Segment_Filename'] = chunk_name

                # update_csv(file_path, data_dict)
                
                break
            else: 
                break  # Skip the last chunk if it's less than 10 seconds

        chunk = audio[i:i + chunk_length]
        segment_number = i // chunk_length
        # chunk_name = f"{filename_base}_chunk{i//1000}.ogg"
        chunk_name = f"{filename_base}_seg_{segment_number + 1}.ogg"
        chunk_path = os.path.join(dest_folder, chunk_name)
        chunk.export(chunk_path, format="ogg")

def perform_all_aug(src, dest_folder):

    gain(src, dest_folder)
    # data_dict['Augment_Filename'] = aug_filename + "_aug_gain.ogg"
    # update_csv(csv_path, data_dict)

    gaussian_noise(src, dest_folder)
    # data_dict['Augment_Filename'] = aug_filename + "_aug_gaussian.ogg"
    # update_csv(csv_path, data_dict)

    BG_1_filepath = "birdclef-preprocess/bg_noise/breeze_through_the_trees.wav"
    BG_2_filepath = "birdclef-preprocess/bg_noise/calm_thunderstorm.wav"
    BG_3_filepath = "birdclef-preprocess/bg_noise/crickets_singing.wav"
    BG_4_filepath = "birdclef-preprocess/bg_noise/leaves_rustling.mp3"
    BG_5_filepath = "birdclef-preprocess/bg_noise/river_in_the_forest.mp3"

    mix_audio_files(src, BG_1_filepath, dest_folder, "aug_bg_1", file2_volume=-20)
    mix_audio_files(src, BG_2_filepath, dest_folder, "aug_bg_2", file2_volume=-20)
    mix_audio_files(src, BG_3_filepath, dest_folder, "aug_bg_3", file2_volume=-20)
    mix_audio_files(src, BG_4_filepath, dest_folder, "aug_bg_4", file2_volume=-20)
    mix_audio_files(src, BG_5_filepath, dest_folder, "aug_bg_5", file2_volume=-20)
    
def generate_pt_files(root_directory, dst_root):
    
    folders_list = os.listdir(root_directory)
    sorted_folders_list = sorted(folders_list)

    for label, folder in enumerate(sorted_folders_list):
        files_list = os.listdir(os.path.join(root_directory, folder))

        for file in tqdm(files_list, desc=f"Processing {folder}", unit="file"):

            file_path_src = os.path.join(root_directory, folder, file)

            # Load audio waveform
            audio_waveform = whisper.load_audio(file_path_src)

            # Pad or trim audio waveform
            audio_waveform = torch.from_numpy(whisper.pad_or_trim(audio_waveform))

            # Compute mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_waveform)

            file_name = file.split('.')[0]
            file_name = file_name + '.pt'
            file_path_dest = os.path.join(dst_root, folder, file_name)

            # Save mel spectrogram as .pt file
            torch.save(mel, file_path_dest)
            
def generate_csv_files(root_directory, output_csv):
    # List to store all the entries
    entries = []

    # Walk through the directory
    for subdir, _, files in os.walk(root_directory):
        # Get the sub-directory name
        label_name = os.path.basename(subdir)

        # Process each file
        for file in files:
            if file.endswith('.pt'):
                filename_pt = file
                entries.append((label_name, label_name + "/" + filename_pt))

    # Sort entries by Label_Name and filename_pt
    entries.sort(key=lambda x: (x[0], x[1]))

    # Write sorted entries to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header row
        csv_writer.writerow(['Label_Name', 'filename_pt'])
        
        # Write the sorted entries
        csv_writer.writerows(entries)

    print(f"CSV file '{output_csv}' created successfully.")
    
    
