# Bird Whisperer: Leveraging Large Pre-trained Acoustic Model for Bird Call Classification (InterSpeech'24)

> [**Bird Whisperer: Leveraging Large Pre-trained Acoustic Model for Bird Call Classification**](/media/TO_BE_UPDATED.md)<br><br>
> [Muhammad Umer Sheikh](https://scholar.google.com/citations?hl=en&authuser=2&user=xwnfWHEAAAAJ), [Hassan Abid](https://scholar.google.com/citations?user=0kaOLSgAAAAJ&hl=en), [Bhuiyan Sanjid Shafique](),
[Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), and
[Muhammad Haris](https://scholar.google.com/citations?user=ZgERfFwAAAAJ&hl=en)


<!-- [![page](https://img.shields.io/badge/Project-Page-F9D371)](/media/TO_BE_UPDATED.md) -->
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](/media/bird_whisperer_interspeech2024.pdf)
[![poster](https://img.shields.io/badge/Presentation-Poster-blue)](/media/bird_whisperer_poster.pdf)

<p align="center"><img src="https://i.imgur.com/waxVImv.png" alt="Image"></p>

<hr />

| ![main figure](/media/methodology.png)|
|:--| 
| **Introduction**<p align="justify">We propose an innovative approach that adapts the large pre-trained Whisper acoustic model, originally designed for human speech recognition, to classify bird calls. Traditional acoustic models often fail to capture meaningful features from non-human audio data like bird vocalizations, categorizing them as background noise. To overcome this, Bird Whisperer leverages a simple yet effective fine-tuning technique that significantly enhances the Whisper model's ability to extract distinct features from bird calls, resulting in a 15% improvement in F1-score compared to the baseline. The model also addresses the challenge of class imbalance by applying a series of audio and spectrogram augmentations to the BirdCLEF 2023 dataset, further boosting performance. Bird Whisperer showcases the potential of adapting large pre-trained acoustic models for broader bioacoustic classification tasks, providing valuable insights for ecological studies and biodiversity monitoring.</p> |

</br>
<hr />
</br>

| ![main figure](/media/whisper_features.png)|
|:--| 
| **Whisper Fine-tuning in Action**<p align="justify">The figure above illustrates the transformation of feature representations extracted by the Whisper encoder from bird call recordings. Initially, the pre-trained Whisper encoder, designed for human speech, fails to capture distinct features from bird calls, treating them as background noise (left side). The feature representations appear uniform and lack patterns specific to avian vocalizations. However, after fine-tuning the Whisper model on the bird call dataset, the encoder adapts and begins to extract richer, more informative features (right side), significantly improving its ability to classify bird species accurately.</p> |

</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Adapting large pre-trained acoustic models across diverse domains poses a significant challenge in speech processing, particularly when shifting from human to non-human contexts. This study aims to bridge this gap by utilizing the pre-trained Whisper model, initially intended for human speech recognition, for classifying bird calls. Our study reveals that when employed solely as a feature extractor, the Whisper encoder fails to yield meaningful features from bird calls, possibly due to categorizing them as background noise. We propose a simple but effective technique to enhance Whisper’s ability to extract distinctive features from avian vocalizations, resulting in a remarkable 15% increase in F1-score over the baseline. Furthermore, we mitigate the issue of class imbalance within the dataset by introducing a series of data augmentations. Our findings underscore the potential of adapting large pre-trained acoustic models to tackle broader bioacoustic classification tasks.
</i></p>

<!-- </br>
<hr />
</br>

For more details, please refer to our or [arxive paper](). -->

</br>
<hr />
</br>

## Updates :rocket:
- **June 04, 2024** : Accepted in [INTER SPEECH 2024](https://interspeech2024.org/) &nbsp;&nbsp; :confetti_ball: :tada:
- **September 01, 2024** : Released code for Bird Whisperer
- **In Progress** : Preparing the dataset processing instructions

</br>
<hr />
<br>

## Installation :wrench:
1. Create a conda environment
```shell
conda create --name bird-whisperer python=3.8
conda activate bird-whisperer
```
2. Install PyTorch and other dependencies
```shell
git clone https://github.com/umer-sheikh/bird-whisperer
cd bird-whisperer
pip install -r requirements.txt
```

</br>

## Dataset :page_with_curl:
We have performed experiments on the bird call classification dataset: [BirdCLEF 2023](https://www.kaggle.com/competitions/birdclef-2023).

We provide instructions for downloading and processing the dataset used by our method in the [DATASET.md](/birdclef-preprocess/DATASET.md). 

All files after downloading and preprocessing should be placed in a directory named `birdclef2023-dataset` and the path of this directory should be specified in the variable `DATASET_ROOT` in the shell [scripts](/scripts/). The directory structure should be as follows:

```
birdclef2023-dataset/
    ├── audio_files/
        |── original/
        |── augmented/
    ├── pt_files/
        |── original/
        |── augmented/
        |── original.csv
        |── augmented.csv
 ```

</br>

## Run Experiments :zap:

We have performed all experiments on `NVIDIA RTX 4090` GPU. Shell scripts to run experiments can be found in [scripts](/scripts/) folder. Below are the shell commands to run experiments (`fine-tuning`, `linear probing`, or `random initialization`). Before running the commands, please ensure that you set the paths for the dataset root directory and the directory where you want to save the model weights.

```shell
## Fine Tuning
bash scripts/bird_whisperer_finetune.sh
```

```shell
## Linear Probing
bash scripts/bird_whisperer_linearprobing.sh
```

```shell
## Random Initialization
bash scripts/bird_whisperer_randominit.sh
```

Results are saved in `json` format in [logs](/logs/) directory.

To run experiments on the original dataset, simply remove the `--augmented_run` argument from the Shell scripts.

If you want to use the EfficientNet-B4 model as the feature extractor instead of Whisper, change the `MODEL_NAME` variable in the shell [scripts](/scripts) from `'whisper'` to `'efficientnet_b4'`.

</br>
<hr/>
</br>

## Results :microscope:

![main figure](/media/table_1.png)
</br>
</br>
![main figure](/media/figure_1.png)
</br>

<hr/>

## Citation :star:
If you find our work or this repository useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{birdwhisperer2024,
  title     = {Bird Whisperer: Leveraging Large Pre-trained Acoustic Model for Bird Call Classification},
  author    = {Muhammad Umer Sheikh and Hassan Abid and Bhuiyan Sanjid Shafique and Asif Hanif and Muhammad Haris},
  booktitle = {Proceedings of the INTERSPEECH 2024 Conference},
  year      = {2024},
}
```

<hr/>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **hassan.abid@mbzuai.ac.ae**

<hr/>

## Acknowledgement :pray:
We used the [Whisper](https://github.com/openai/whisper) codebase for the feature extraction in our proposed method **Bird Whisperer**. We thank the authors for releasing the codebase.

