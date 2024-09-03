import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def get_dicts(csv_path):
    # Reading the csv file
    df = pd.read_csv(csv_path)

    primary_labels = df['Label_Name'].values.tolist()
    primary_labels_unique = sorted(list(set(primary_labels)))  

    bird2label_dict = {}   # this dictionary will give integer 'label' when a 'bird' name is used in key
    label2bird_dict = {}   # this dictionary will give 'bird' name when integer 'label' is used in key

    for i, bird in enumerate(primary_labels_unique):
        bird2label_dict[bird] = i
        label2bird_dict[i] = bird
    
    return df, bird2label_dict, label2bird_dict


def get_train_test_data(df, bird2label_dict, seed=42):
    lambda_function = lambda bird: bird2label_dict[bird]

    df.insert(1, 'labels', None) # Adding new column 'labels' at index=1. All values will be None in this column
    df['labels'] = df['Label_Name'].apply(lambda_function)   
    df_labels_paths = df[['labels', 'filename_pt']]

    # Using stratify for unbalanced classes of dataset by keeping the random state fixed
    classes_with_few_instances = df_labels_paths['labels'].value_counts()[df_labels_paths['labels'].value_counts() < 3].index

    # Separate data into two parts: one with classes having >= 3 instances and one with the rest
    df_with_few_instances = df_labels_paths[df_labels_paths['labels'].isin(classes_with_few_instances)]
    df_remaining_instances = df_labels_paths[~df_labels_paths['labels'].isin(classes_with_few_instances)]

    # Perform stratified split on the remaining instances
    df_train_remaining, df_test_remaining = train_test_split(
        df_remaining_instances, 
        test_size=0.2, 
        stratify=df_remaining_instances[['labels']], 
        random_state=seed
    )

    # Include one instance from each class in both training and testing sets
    df_train = pd.concat([df_train_remaining, df_with_few_instances.groupby('labels').head(1)])
    df_test = pd.concat([df_test_remaining, df_with_few_instances.groupby('labels').tail(1)])

    # Shuffle the resulting DataFrames
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    audio_files_paths_train = df_train['filename_pt'].values.tolist()
    labels_train = df_train['labels'].values.tolist()

    audio_files_paths_test = df_test['filename_pt'].values.tolist()
    labels_test = df_test['labels'].values.tolist()

    labels_unique = sorted(list(set(df['labels'])))
    
    return audio_files_paths_train, labels_train, audio_files_paths_test, labels_test, labels_unique