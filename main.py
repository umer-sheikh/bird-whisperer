import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

import utils.trainer as trainer
from models.model_utils import get_model
from utils.trainer import load_model, print_test_scores
from datasets.dataset_utils import get_dataloaders
from utils.utils import print_total_time, print_dataset_info, get_optimizer, get_args, setup_logging


def main(args):

	print("\n\n#####################################################")
	print("BirdCLEF-2023 (Fine-grained Bird Call Classification)")
	print("#####################################################\n\n")
	

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device
	
	print(f"Arguments:\n{args}\n")


	# To ensure reproducibility
	seed = args.seed
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)

	
	dataset_root = args.dataset_root
	model_name = args.model_name
	training_mode = args.training_mode
	augmented_run = args.augmented_run
	spec_aug = args.spec_aug
	n_epochs = args.n_epochs
	start_epoch = 0
	do_logging = args.do_logging
	logs_dir  = args.log_dir
	batch_size = args.batch_size
	num_workers = args.num_workers
	lr = args.lr


	if args.training_mode=='random-init':
		pre_trained = False
	elif args.training_mode=='linear-probing' or  args.training_mode=='fine-tuning':
		pre_trained = True
	else:
		raise ValueError("Unrecognized 'training mode'. Choose from ['random-init', 'linear-probing', 'fine-tuning']")
	


	save_model_path = os.path.join(args.save_model_path, model_name, args.config_name)
	if not args.eval_only: print(f"\nModel will be saved at '{save_model_path}'\n")
	if not os.path.exists(save_model_path): os.makedirs(save_model_path)


	train_dataloader, test_dataloader, labels_unique = get_dataloaders(dataset_root, augmented_run, spec_augment=spec_aug, seed=seed, batch_size=batch_size, num_workers=num_workers)
	print_dataset_info(train_dataloader, test_dataloader)


	model = get_model(model_name=model_name, pre_trained=pre_trained)
	model = model.to(device)


	criterion = torch.nn.CrossEntropyLoss() # loss function
	optimizer = get_optimizer(model_name, model, training_mode, lr=lr)


	if args.eval_only:
		print("\n\n###################################################")
		print("Starting Model Evaluation on BirdCLEF Test Dataset")
		print(f"Loading Model from Path='{args.model_path}'")
		
		now_start = datetime.now()
		load_model(model, optimizer, args.model_path)
		
		print("\nEvaluating ...")
		class_report, avg_test_loss = trainer.test(device, model, model_name, test_dataloader, criterion, labels_unique)
		print_test_scores(class_report, avg_test_loss)
		
		now_end = datetime.now()
		print_total_time(now_start,now_end)
		print(f"\nTesting Completed Successfully!\n\n")
		
	else:
		now_start = datetime.now()
		print("\n\n###################################################")
		print(f"\nStarting Training the Model on BirdCLEF Dataset ...\n")
		print(f'\nStart Time & Date = {now_start.strftime("%I:%M %p")} , {now_start.strftime("%d_%b_%Y")}\n')
		print(f"Training Mode = {training_mode.upper()}\n")

		trainer.train(device, model, model_name, train_dataloader, test_dataloader, criterion, optimizer, labels_unique, start_epoch, n_epochs, training_mode, save_model_path, logs_dir, do_logging=do_logging, args=args)

		now_end = datetime.now()
		print_total_time(now_start, now_end)

		print(f"\nTraining Completed Successfully!\n\n")




if __name__ == "__main__":

	args = get_args()

	setup_logging(args)

	main(args)

