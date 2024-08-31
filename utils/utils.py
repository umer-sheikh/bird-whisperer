import os, sys, argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='BirdCLEF-2023 (Fine-grained Bird Call Classification)')
    parser.add_argument('--model_name', type=str, default='', help='Model Name (default: None)', required=True)
    parser.add_argument('--save_model_path', type=str, default='', help='Path to save the trained model (default: None)', required=True)
    parser.add_argument('--model_path', type=str, default='', help='Path to the pre-trained model (default: None)')
    parser.add_argument('--dataset_root', type=str, default='', help='Path to the dataset root directory (default: None)', required=True)
    parser.add_argument('--training_mode', type=str, default='fine-tuning', help='Training Mode (default: fine-tuning)', choices=['random-init', 'linear-probing', 'fine-tuning'])
    parser.add_argument('--augmented_run', help='Train on augmented dataset (default: False)', action='store_true')
    parser.add_argument('--spec_aug', help='Apply Spectrogram Augmentation (default: False)', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs (default: 20)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed (default: 42)')
    parser.add_argument('--eval_only', help='Evaluate the model only (default: False)', action='store_true')
    parser.add_argument('--do_logging', help='Enable/disable Logging (default: False)', action='store_true')
    args = parser.parse_args()

    # sanity check of arguments
    if not os.path.exists(args.dataset_root): raise ValueError(f"Directory '{args.dataset_root}' does not exist. Specify the correct path to the dataset.")
    if (not args.eval_only) and (not os.path.exists(args.save_model_path)): raise ValueError(f"Training Mode: Directory '{args.save_model_path}' does not exist. Create or specify the correct the directory to save the trained model.")
    if args.eval_only and (not os.path.exists(args.model_path)): raise ValueError(f"Evaluation Mode: Model file '{args.model_path}' does not exist. Specify the correct path to the model file.")
    
    return args




def get_optimizer(model_name, model, training_mode, lr=0.01):
    assert training_mode in ['random-init', 'linear-probing', 'fine-tuning'], f"Unrecognized training mode='{training_mode}'. Choose from ['random-init', 'linear-probing', 'fine-tuning']"

    if model_name == 'efficientnet_b4':
        if training_mode == 'random-init' or training_mode == 'fine-tuning': optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif training_mode == 'linear-probing': optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr)
       
    elif model_name == 'whisper':
        if training_mode == 'random-init' or training_mode == 'fine-tuning': optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif training_mode == 'linear-probing': optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr)

    else: raise ValueError(f"Unrecognized model_name='{model_name}'. Choose from ['efficientnet_b4', 'whisper']")

    return optimizer




def print_dataset_info(train_dataloader, test_dataloader):
    num_batches_train = len(train_dataloader)
    num_batches_test = len(test_dataloader)

    print("\n########################\nDataset Information\n########################\n")
    print("Number of Samples in Train Dataset: ", len(train_dataloader.dataset))
    print("Number of Batches in Train Dataloader: ", num_batches_train)
    print("Train Batch Size: ", train_dataloader.batch_size)
    # Num Classes


    print("Number of Samples in Test Dataset: ", len(test_dataloader.dataset))
    print("Test Batch Size: ", test_dataloader.batch_size)
    print("Number of Batches in Test Dataloader: ", num_batches_test)
    # Num Classes



def print_total_time(now_start, now_end):
	print(f'\nEnd Time & Date = {now_end.strftime("%I:%M %p")} , {now_end.strftime("%d_%b_%Y")}\n')
	duration_in_s = (now_end - now_start).total_seconds()
	days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
	hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
	minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
	seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds
	print(f"Total Time => {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds\n\n")



# decorator to measure the time taken by a function
def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        duration_in_s = end - start
        days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
        hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
        minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
        seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds
        print(f"\nTotal Time => {int(days[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds")
        return result
        
    return wrapper




# Define a Tee class to duplicate output to both stdout and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
 
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
 
    def flush(self):
        for file in self.files:
            file.flush()


# Define a function to redirect stdout and stderr to a log file
def redirect_output_to_log(log_file):
    # Open the log file in append mode
    log = open(log_file, 'a')
 
    # Duplicate stdout and stderr
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)

    return log


# Define a function to setup logging
def setup_logging(args):
    config_name = f"{'data-aug_' if args.augmented_run else 'data-orig_'}{'spec-aug_' if args.spec_aug else 'no-spec-aug_'}{args.training_mode}"
    args.config_name = config_name

    log_dir = os.path.join('logs', args.model_name, config_name) # log file dir
    args.log_dir = log_dir
    

    if args.do_logging:
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, f"{args.config_name}.log")
        if os.path.exists(log_file_path): os.remove(log_file_path)
        json_file_path = os.path.join(log_dir, f"{args.config_name}.json")
        args.json_file_path = json_file_path
        print(f"\nLogging to '{log_file_path}'\n")
        log_file = redirect_output_to_log(log_file_path) # redirect terminal output to log file
    else:
        log_file =None

    return log_file


