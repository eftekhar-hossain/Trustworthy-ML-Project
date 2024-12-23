# (CAP 6938 Course Project)
# Investigating Adversarial Robustness of Hateful Meme Detection Systems 


## Table of Contents
  1. [Instructions](#instructions)
  2. [Models Training](#models-training)
  3. [Models Checkpoint](#models-checkpoint)
  4. [Threat Models](#threat-models)
     - [Black Box Attack on Images](#black-box-attack-on-images)
     - [White Box Attack on Images](#white-box-attack-on-images)
     - [Transfer Attack on Images](#transfer-attack-on-images)
     - [Black Box Attack on Texts](#black-box-attack-on-texts)


## Folder Organization

Folders need to be organized as follows in `Trustworthy-ML-Project`

```
├── Datasets
|   ├── BHM
        ├── Files 
        └── Memes
    ├── MIMOSA
        ├── Memes 
        └── .csv files  
    ├── download_dataset.sh 
|   
├── Scripts
   ├──  Slurm Scripts --> only for server
   └── Baseline Results 
   └── .py files
|   
├── Saved_Models  -->
└── Models Checkpoint 
├── requirements.txt           
```

# Instructions
- To run the scripts, you need to  install `Python=3.10.x`. All the models are implemented using `Pytorch=2.4.0`. 
- If you use any IDE, first clone (`git clone <url>`) the repository. Then, create a virtual environment and activate it.

    `conda create -n Trustworthy-ML-Project Python=3.10.12`<br>
    `conda activate Trustworthy-ML-Project`

- Install all the dependencies.<br>
`pip install -r requirements.txt`

- Download the Datasets.<br>
`cd Datasets` <br>
`bash download_dataset.sh`<br>
`cd ..` --> back to root directory<br>
Ensure the downloaded datasets follow the folder organization.

# Models Training 

We have reproduced the results of three multimodal models. Two datasets were utilized here **Bengali Hateful Memes (BHM)** and **MultIMOdal AggreSsion DAtaset (MIMOSA)**. 

- Dual Co-Attention Framework (DORA)
- Multimodal Attentive Fusion (MAF)
- Multilingual CLIP (m-clip)

Codes are available in the `Scripts` folder. 

- `dataset.py` contains the `Dataloader` for the **BHM** and **MIMOSA** dataset.
- `dora.py` implementation of **DORA** architecture
- `maf.py` implementation of **MAF** architecture
- `mclip.py` implementation of **mclip** architecture
- `wb-attack.py` implementation of two white box attacks e.g., **PGD** and **FGSM**.
- `transfer-attack.py` implementation of transfer attack using **FGSM** on only **ResNet** model.
- `img-bb-attack.py` implementation of various image black box attack e.g., **Gaussian**, **Salt-Pepper**, **News-Print**, and **Random** noise.
- `text-attack.py` implementation of various text black box attack e.g., **Postive Token**, **Emoji**, **Typos**, and **Cross-Lingual Counterpart** noise.

Example:

To run `DORA` on `MIMOSA` dataset run the following command. If you are not in the `Scripts` folder.

```
cd Scripts

python main.py \
  --dataset mimosa \
  --method dora \
  --max_len 50 \
  --heads 2 \
  --epochs 40 \
  --learning_rate 2e-5
```

**Arguments**

- `--dataset`: Specifies the dataset to use (`mimosa` or `bhm`).
- `--method`: Specifies the method (`dora`,`maf`, or `mclip`).
- `--max_len`: Sets the maximum sequence length (`default: 50`). 
- `--heads`: Sets the number of attention heads.
- `--epochs`: Specifies the number of training epochs. (`default: 50`)
- `--learning_rate`: Sets the learning rate (`default: 2e-5`).
- `batch_size` is a default argument set to 16.


(**Note:** `mclip` don't require this `--max_len` and `--heads` arguments.)

To run `mclip` on `BHM` dataset run the following command. If you are not in `Scripts` folder. 

**Note:** Training `mclip` will require 1 hour for each epoch.

```
cd Scripts

python main.py \
  --dataset bhm \
  --method mclip \
  --epochs 50 \
  --learning_rate 5e-4  
```

Trained models will be saved as `<method>_<dataset>_<learning_rate>.pth` into the `Saved_Models` directory. `Early stopping` has been utilized, therefore if the models validation accuracy doesn't improve for consecutive 5 epochs training will be stopped.

## Models Checkpoint

You can use the checkpoints of the already trained models for evaluation. Run the following command to download all the checkpoints. You can find the models on `Model Checkpoint` folder.

```
bash download_checkpoint.sh
```
---
# Threat Models

## Black Box Attack on Images

To perform **Black Box** attack on `DORA` model using `MIMOSA` dataset. If you are not in the `Scripts` folder.

```
cd Scripts

python img-bb-attack.py \
  --dataset mimosa \
  --method dora \
  --noise all
```

**Arguments**

- `--dataset`: Specifies the dataset to use (`mimosa` or `bhm`).
- `--method`: Specifies the method (`dora`,`maf`, or `mclip`).
- `--noise`: Specifies the noise (`guassian`,`salt-peper`,`newsprint` ,`random`, or `all`); <u>default:</u> `guassian`.

You will get results for individual attacks with varying values of their parameters. For example, for `salt-peper` noise, you will get results for `salt` and `pepper` values ranging from `[0.01,0.03,0.05]`.

---
## White Box Attack on Images

To perform a **White Box** attack on the `MAF` model using the `BHM` dataset.

```
python wb-attack.py \ 
  --dataset bhm \ 
  --method maf \ 
  --attack all \
  --epsilon 0.03 \
  --alpha 0.005 \
  --steps 40
```
**Arguments**

- `--dataset`: Specifies the dataset to use (`mimosa` or `bhm`).
- `--method`: Specifies the method (`dora`,`maf`, or `mclip`).
- `--attack`: Specifies the attack name (`pgd`,`fgsm`, or `all`);  <u>default:</u> `pgd`.
- `--epsilon`: Amount of perturbation; <u>default:</u> `0.0015`.
- `--alpha`: Step size; <u>default:</u> `0.0005`.
- `--steps`: Number of steps; <u>default:</u> `30`.

---
## Transfer Attack on Images

To perform **Transfer Attack (FGSM)** on the `MCLIP` model using the `MIMOSA` dataset.

```
python transfer-attack.py \
   --dataset mimosa \
   --method mclip \ 
   --epsilon 0.03 
```
**Arguments**

- `--dataset`: Specifies the dataset to use (`mimosa` or `bhm`).
- `--method`: Specifies the method (`dora`,`maf`, or `mclip`).
- `--epsilon`: Amount of perturbation; <u>default:</u> `0.015`.

---
## Black Box Attack on Texts

To perform **Black Box Text Attack (Add Random Emoji)** on the `DORA` model using the `BHM` dataset.

```
python text-attack.py \
   --dataset BHM \
   --method dora \
   --text_attack emoji \
   --frequency 2
```
**Arguments**

- `--dataset`: Specifies the dataset to use (`mimosa` or `bhm`).
- `--method`: Specifies the method (`dora`,`maf`, or `mclip`).
- `--text_attack`: Specifies the type of text attack (`emoji`,`positive_token`,`typo`,`translate` or `all`).
- `--frequency`: Number of tokens to insert/add; <u>default:</u> `1`.
