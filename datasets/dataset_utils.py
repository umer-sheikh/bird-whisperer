import os
import torch
from torch.utils.data import DataLoader

from datasets.dataset import BirdClefDataset
from birdclef_preprocess.data_preprocessing import get_dicts, get_train_test_data


def get_dataloaders(dataset_root, augmented_run, spec_augment, seed, batch_size=16, num_workers=4):

	if augmented_run:
		csv_path = os.path.join(dataset_root, 'pt_files/augmented.csv')
		audio_folder_path = os.path.join(dataset_root, 'audio_files/augmented_audio/')
		main_folder_path = os.path.join(dataset_root, 'pt_files/augmented/')
	else:
		csv_path = os.path.join(dataset_root, 'pt_files/original.csv')
		audio_folder_path = os.path.join(dataset_root, 'audio_files/original/')
		main_folder_path = os.path.join(dataset_root, 'pt_files/original/')


	df, bird2label_dict, label2bird_dict = get_dicts(csv_path)
	audio_files_paths_train, labels_train, audio_files_paths_test, labels_test, labels_unique = get_train_test_data(df, bird2label_dict, seed=seed)
	
	train_dataset = BirdClefDataset(audio_files_paths_train, labels_train, main_folder_path, bird2label_dict=bird2label_dict, label2bird_dict=label2bird_dict, spec_augment=spec_augment)
	test_dataset = BirdClefDataset(audio_files_paths_test, labels_test, main_folder_path, bird2label_dict=bird2label_dict, label2bird_dict=label2bird_dict, spec_augment=False)


	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	return train_dataloader, test_dataloader, labels_unique


