# BertSNR

![Image browser window](https://github.com/lhy0322/BertSNR/blob/main/Figure.JPG)

## 1. Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). We applied training on a single NVIDIA TITAN X with 12 GB graphic memory. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.

#### 1.1 Create and activate a new virtual environment

```
conda create -n bertsnr python=3.7
conda activate bertsnr
```

#### 1.2 Install the package and other requirements

(Required)

```
git clone https://github.com/lhy0322/BertSNR
cd BertSNR
conda install --file requirements.txt
```
## 2. Download pre-trained DNABERT
The [DNABERT](https://github.com/jerryji1993/DNABERT) provides four pre-trained models that have been trained on the whole human genome. Please go to their github and download the corresponding k-mer pre-training model. Then unzip the package by running:
```
unzip 3-new-12w-0.zip
unzip 4-new-12w-0.zip
unzip 5-new-12w-0.zip
unzip 6-new-12w-0.zip
```

## 3. Model pipeline
```
BertSNR/
├── Baseline/             
│   ├── DeepLearning_Motif.py     # baseline(DeepSNR and D-AEDNet) motif generation
│   ├── DeepLearning_Test.py      # baseline(DeepSNR and D-AEDNet) test
│   ├── DeepLearning_Train.py     # baseline(DeepSNR and D-AEDNet) train
│   ├── Matching_method.py        # baseline(Matching)
│
├── DNABERT/                # pre-trained model
│   ├── 3-new-12w-0         # 3-mer model
│   ├── ...
│
├── Dataset/               
│   ├── ChIP-seq/           # 188 preprocessed datasets
│   ├── JASPAR/             # original data
│   ├── CreateDataset.py    # data process
│
├── Main/              
│   ├── CrossValidToken.py  # cross-validation
│   ├── GenerateMotif.py    # motif generation
│   ├── Predict.py          # predict TFBS
│   ├── TrainMultitask.py   # train model
│
├── Model/                  # model architecture 
│   ├── BertSNR.py  
│   ├── D_AEDNet.py   
│   ├── DeepSNR.py        
│
└── Utils/               
    ├── BertViz.py         # attention visualisation
    ├── Metrics.py         
    ├── MotifDiscovery.py  # motif discovery algorithm
    ├── Shuffle.py         # dinucleotide frequencies shuffling 
    ├── Visualization.py/  # attention visualisation

```
#### 3.1 Data processing
Use "Dataset/CreateDataset.py" file to generate k-mer sequences files from 188 ChIP-seq datasets

#### 3.2 Train
Use "Main/TrainMultitasking.py" to train model in 188 ChIP-seq datasets

#### 3.3 Predict
Use "Main/Predict.py" to predict TFBS

#### 3.4 Motif
Use "Main/GenerateMotif.py" to generate motif 

## Reference
1. Ji Y, Zhou Z, Liu H, et al. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome[J]. Bioinformatics, 2021, 37(15): 2112-2120.
2. Zhang Y, Wang Z, Zeng Y, et al. High-resolution transcription factor binding sites prediction improved performance and interpretability by deep learning method[J]. Briefings in Bioinformatics, 2021, 22(6): bbab273.
