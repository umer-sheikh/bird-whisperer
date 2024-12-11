# Dataset Preparation for [Bird Whisperer](https://github.com/umer-sheikh/bird-whisperer)

This document provides step-by-step instructions for preparing the datasets required for training and testing the models. The [Bird Whisperer](https://github.com/umer-sheikh/bird-whisperer) project utilizes the [BirdCLEF 2023](https://www.kaggle.com/competitions/birdclef-2023/data) dataset, followed by further preprocessing.

---

### Step 1: Download the Dataset
1. Download the BirdCLEF 2023 dataset from the official Kaggle page:  
   **[BirdCLEF 2023 Dataset](https://www.kaggle.com/competitions/birdclef-2023/data)**.

2. Move the downloaded ZIP file to the following directory:  

```bash
bird-whisperer/birdclef-preprocess/
```
 
 ---

### Step 2: Generate the Dataset
After placing the ZIP file in the appropriate directory, follow these steps to preprocess the dataset:

1. Open the terminal and navigate to the directory:

```bash
cd birdclef-preprocess
```

2. Run the preprocessing script:


```bash
python birdclef_generation.py
```

---

### Step 3: Output Dataset Structure
Once the preprocessing is complete, the dataset will have the following structure:

```bash
|── birdclef-preprocess/
    |── birdclef2023-dataset/
        |── audio_files/
            |── augmented/
                |── class_1/
                    |── audio_1.ogg
                    |── audio_2.ogg
                    |── ...
                |── class_2/
                    |── audio_1.ogg
                    |── audio_2.ogg
                    |── ...
                .
                .
                .
                |── class_N/
                    |── audio_1.ogg
                    |── audio_2.ogg
                    |── ...
                    
            |── original/
                |── class_1/
                    |── audio_1.ogg
                    |── audio_2.ogg
                    |── ...
                |── class_2/
                    |── audio_1.ogg
                    |── audio_2.ogg
                    |── ...
                .
                .
                .
                |── class_N/
                    |── audio_1.ogg
                    |── audio_2.ogg
                    |── ...
                    
        |── pt_files/
            |── augmented.csv
            |── original.csv
            |── augmented/
                |── class_1/
                    |── pt_file_1.pt
                    |── pt_file_2.pt
                    |── ...
                |── class_2/
                    |── pt_file_1.pt
                    |── pt_file_2.pt
                    |── ...
                .
                .
                .
                |── class_N/
                    |── pt_file_1.pt
                    |── pt_file_2.pt
                    |── ...
                    
            |── original/
                |── class_1/
                    |── pt_file_1.pt
                    |── pt_file_2.pt
                    |── ...
                |── class_2/
                    |── pt_file_1.pt
                    |── pt_file_2.pt
                    |── ...
                .
                .
                .
                |── class_N/
                    |── pt_file_1.pt
                    |── pt_file_2.pt
                    |── ...
            

```

---



