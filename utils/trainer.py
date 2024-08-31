import os
import json
import torch
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
from utils.utils import timeit



def train(device, model, model_name, train_dataloader, test_dataloader, criterion, optimizer, labels_unique, start_epoch, n_epochs, training_mode, save_model_path, logs_dir, do_logging=False, args=None):
    
    best_macro_avg_f1 = 0
    best_epoch = 0
    train_losses = []
    test_epochs = []
    test_metrics = []

    # Ensures save_model_path exists
    os.makedirs(save_model_path, exist_ok=True)

    for epoch in range(start_epoch, n_epochs):
        
        # Run training epoch
        class_report, avg_epoch_loss = run_epoch(device, model, model_name, train_dataloader, criterion, optimizer, labels_unique, epoch)
        
        macro_avg_f1_train = class_report['macro avg']['f1-score']
        print(f"\nAverage Train Loss = {avg_epoch_loss:0.4f},   Train F1-Score (Macro Avg) = {macro_avg_f1_train:0.4f}")
        
        train_losses.append(avg_epoch_loss)

        # Save the latest model
        save_model(model, optimizer, epoch, avg_epoch_loss, save_model_path, 'model_latest.pt')

        # Test the model
        print(f"\n\nEvaluating after Epoch = {epoch} ...")
        class_report, avg_test_loss = test(device, model, model_name, test_dataloader, criterion, labels_unique)
        macro_avg_f1_test = class_report['macro avg']['f1-score']
        print_test_scores(class_report, avg_test_loss)

        test_metrics.append([class_report['macro avg']['f1-score'], class_report['macro avg']['precision'], class_report['macro avg']['recall'], class_report['accuracy'], avg_test_loss])
        test_epochs.append(epoch)

        # Save the best model based on macro_avg_f1 score
        if macro_avg_f1_test > best_macro_avg_f1:
            best_macro_avg_f1 = macro_avg_f1_test
            best_epoch = epoch
            print(f"\n{'Best Macro Avg F1-Score':<20} = {best_macro_avg_f1:0.4f}")
            print(f"Saving the best model at '{save_model_path}' ... ")
            save_model(model, optimizer, epoch, avg_epoch_loss, save_model_path, 'model_best.pt')

        # Save results to JSON file
        if do_logging:
            results = {
                'train_epoch': list(range(start_epoch, epoch + 1)),
                'train_loss': train_losses,
                'test_epoch': test_epochs,
                'epoch_test_metrics': test_metrics
            }

            with open(args.json_file_path, 'w') as f:
                json.dump(results, f)

    print(f"\nBest Macro Avg F1-Score = {best_macro_avg_f1:0.4f} was achieved at Epoch = {best_epoch}\n")
    


@timeit
def run_epoch(device, model, model_name, train_dataloader, criterion, optimizer, labels_unique, epoch):
    print(f"\n\n######################################################\nEpoch = {epoch}\n######################################################\n")
    
    model.train() 

    actual_labels = []
    predicted_labels = []
    loss_vals = []

    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        log_mels, labels, _ = batch
        log_mels, labels = log_mels.to(device), labels.to(device)

        if model_name == 'efficientnet_b4':
            log_mels = log_mels.unsqueeze(1)

        optimizer.zero_grad()              # zero the gradiants of the parameters
        logits = model(log_mels)           # forward pass
        loss = criterion(logits, labels)   # compute loss
        loss.backward()                    # compute gradients of the parameters
        optimizer.step()                   # update the weights with gradients

        _, preds = torch.max(logits, 1)
        predicted_labels.extend(preds.cpu().detach().numpy())
        actual_labels.extend(labels.cpu().detach().numpy())
        loss_vals.append(loss.item())

        if batch_idx % 100 == 0:
            print(f"Batch Index = {batch_idx:03},   Loss = {loss.item():0.4f}")

        

    avg_epoch_loss = sum(loss_vals) / len(loss_vals)
    class_report = classification_report(actual_labels, predicted_labels, labels=labels_unique, zero_division=0, output_dict=True)
    if 'accuracy' not in class_report: class_report['accuracy']=accuracy_score(actual_labels, predicted_labels)

    return class_report, avg_epoch_loss


        
def test(device, model, model_name, test_dataloader, criterion, labels_unique):
    # Set the models to evaluation mode
    model.eval()

    actual_labels = []
    predicted_labels = []
    loss_vals = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_dataloader)):
            log_mels, labels, _ = batch
            log_mels, labels = log_mels.to(device), labels.to(device)

            if model_name == 'efficientnet_b4':
                log_mels = log_mels.unsqueeze(1)

            # Forward pass through the model
            logits = model(log_mels)

            # Calculate loss
            loss = criterion(logits, labels)
            loss_vals.append(loss.item())

            # Predictions
            _, preds = torch.max(logits, 1)
            predicted_labels.extend(preds.cpu().detach().numpy())
            actual_labels.extend(labels.cpu().detach().numpy())
 

    # Compute metrics
    avg_test_loss = sum(loss_vals) / len(loss_vals)
    class_report = classification_report(actual_labels, predicted_labels, labels=labels_unique, zero_division=0, output_dict=True)
    if 'accuracy' not in class_report: class_report['accuracy']=accuracy_score(actual_labels, predicted_labels)

    return class_report, avg_test_loss



def save_model(model, optimizer, epoch, epoch_avg_loss, save_model_path, file_name):
    save_path = os.path.join(save_model_path, file_name)

    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'epoch_avg_loss': epoch_avg_loss,
               }, 
              save_path
            )



def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epoch_avg_loss = checkpoint['epoch_avg_loss']

    return epoch, epoch_avg_loss



def print_test_scores(classification_report_test, avg_test_loss):
    print("\n\n================= Test Metrics =================\n")
    print(f"Average Loss = {avg_test_loss:0.4f}")

    report = classification_report_test
    # Extract the desired summary metrics
    summary = {
        'accuracy': report['accuracy'],
        'macro avg': {
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1-score': report['macro avg']['f1-score']
        },
        'weighted avg': {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score']
        }
    }
    
    print("\nAccuracy: {:.4f}".format(summary['accuracy']))
    
    print("\nMacro Average:")
    print("  Precision: {:.4f}".format(summary['macro avg']['precision']))
    print("  Recall: {:.4f}".format(summary['macro avg']['recall']))
    print("  F1-Score: {:.4f}".format(summary['macro avg']['f1-score']))

    print("\nWeighted Average:")
    print("  Precision: {:.4f}".format(summary['weighted avg']['precision']))
    print("  Recall: {:.4f}".format(summary['weighted avg']['recall']))
    print("  F1-Score: {:.4f}".format(summary['weighted avg']['f1-score']))
    print("\n===============================================\n\n")


