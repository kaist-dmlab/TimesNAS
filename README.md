# TimesNAS
This is the official implementation of the ***Time Series-Aware Zero-Shot Neural Architecture Search for General Time-Series Analysis*** paper _Under Review_ by IEEE Transactions on Neural Networks and Learning Systems
<!-- [[Paper]()]] -->

## Abstract
Designing effective neural networks from scratch for various time-series analysis tasks, such as activity recognition, fault detection, and traffic forecasting, is time-consuming and heavily relies on human labor. This paper thereby aims to answer the following question: how to build a unified zero-shot neural architecture search framework that effectively designs neural ar- chitectures for a variety of tasks and input time series? However, building a universal framework for different tasks comes with challenges. First, we need a unified backbone search space that performs well across diverse analysis tasks. Second, we need a time series-aware zero-shot proxy that consistently correlates with the downstream performance across different characteristics of time-series datasets. To address these challenges, for the first time, we propose a framework for general Time-series analysis with zero-shot Neural Architecture Search named TimesNAS. TimesNAS extends a state-of-the-art foundation model into a two- level search space and enhances a zero-shot proxy by exploiting decomposed time-series properties. Empirically, we show that the architectures found by TimesNAS gain improvement up to 23.6% over state-of-the-art hand-crafted baselines in five mainstream time-series data mining tasks, including short- and long-term forecasting, classification, anomaly detection, and imputation.

## Benchmark Datasets
| **Downstream Analysis Tasks** | **Benchmarks** | **Evaluation Metrics** | **Series Length** |
|---|---|:---:|:---:|
| Short-term Forecasting | M4 (6 subsets) | SMAPE, MASE, OWA | 6 ~ 48 |
| Long-term Forecasting | ETT (4 subsets), Electricity, Traffic, Weather, Exchange | MSE, MAE | 96 ~ 720 |
| Classification | UEA (10 subsets) | Accuracy | 29 ~ 1751 |
| Anomaly Detection | SMD, MSL, SMAP, SWaT, PSM | Precision, Recall, F1 | 100 |
| Imputation | ETT (4 subsets), Electricity, Weather | MSE, MAE | 96 |

## Dataset descriptions. The dataset size is organized in (Train, Validation, Test)
| **Tasks** | **Dataset** | **Dim** | **Series Length** | **Dataset Size** | **Information (Freq.)** |
|---|:---:|:---:|:---:|:---:|:---:|
| Short-term Forecasting | M4-Yearly | 1 | 6 | (23000, 0, 23000) | Demographic |
|  | M4-Quarterly | 1 | 8 | (24000, 0, 24000) | Finance |
|  | M4-Monthly | 1 | 18 | (48000, 0, 48000) | Industry |
|  | M4-Weekly | 1 | 13 | (359, 0, 359) | Macro |
|  | M4-Daily | 1 | 14 | (4227, 0, 4227) | Micro |
|  | M4-Hourly | 1 | 48 | (414, 0, 414) | Other |
| Long-term Forecasting | ETTm1, ETTm2 | 7 | {96, 192, 336, 720} | (34465, 11521, 11521) | Electricity (15 mins) |
|  | ETTh1, ETTh2 | 7 | {96, 192, 336, 720} | (8545, 2881, 2881) | Electricity (15 mins) |
|  | Electricity | 321 | {96, 192, 336, 720} | (18317, 2633, 5261) | Electricity (Hourly) |
|  | Traffic | 862 | {96, 192, 336, 720} | (12185, 1757, 3509) | Transportation (Hourly) |
|  | Weather | 21 | {96, 192, 336, 720} | (36792, 5271, 10540) | Weather (10 mins) |
|  | Exchange | 8 | {96, 192, 336, 720} | (5120, 665, 1422) | Exchange rate (Daily) |
| Classification | EthanolConcentration | 3 | 1751 | (261, 0, 263) | Alcohol Industry |
|  | FaceDetection | 144 | 62 | (5890, 0, 3524) | Face (250Hz) |
|  | Handwriting | 3 | 152 | (150, 0, 850) | Handwriting |
|  | Heartbeat | 61 | 405 | (204, 0, 205) | Heart Beat |
|  | JapaneseVowels | 12 | 29 | (270, 0, 370) | Voice |
|  | PEMS-SF | 963 | 144 | (267, 0, 173) | Transportation (Daily) |
|  | SelfRegulationSCP1 | 6 | 896 | (268, 0, 293) | Health (256Hz) |
|  | SelfRegulationSCP2 | 7 | 1152 | (200, 0, 180) | Health (256Hz) |
|  | SpokenArabicDigits | 13 | 93 | (6599, 0, 2199) | Voice (11025Hz) |
|  | UWaveGestureLibrary | 3 | 315 | (120, 0, 320) | Gesture |
| Anomaly Detection | SMD | 38 | 100 | (566724, 141681, 708420) | Server Machine |
|  | MSL | 55 | 100 | (44653, 11664, 73729) | Spacecraft |
|  | SMAP | 25 | 100 | (108146, 27037, 427617) | Spacecraft |
|  | SWaT | 51 | 100 | (396000, 99000, 449919) | Infrastructure |
|  | PSM | 25 | 100 | (105984, 26497, 87841) | Server Machine |
| Imputation | ETTm1, ETTm2 | 7 | 96 | (34465, 11521, 11521) | Electricity (15 mins) |
|  | ETTh1, ETTh2 | 7 | 96 | (8545, 2881, 2881) | Electricity (15 mins) |
|  | Electricity | 321 | 96 | (18317, 2633, 5261) | Electricity (15 mins) |
|  | Weather | 21 | 96 | (36792, 5271, 10540) | Weather (10 mins) |

## Default Training Hyperparamters
| **Tasks / Configurations** | **LR*** | **Loss** | **Batch Size** | **Epochs** |
|---|:---:|:---:|:---:|:---:|
| Long-term Forecasting | $10^{−4}$ | MSE | 32 | 10 |
| Short-term Forecasting | $10^{−3}$ | SMAPE | 16 | 10 |
| Imputation | $10^{−3}$ | MSE | 16 | 10 |
| Classification | $10^{−3}$ | Cross Entropy | 16 | 30 |
| Anomaly Detection | $10^{−4}$ | MSE | 128 | 10 |
| * LR means the initial learning rate.  |

## Usage

1. **Installation**. Install Python 3.9. For convenience, please run the following command.

```
pip install -r requirements.txt
```

2. **Prepare Data**. You can obtained the well pre-processed datasets from [Dropbox](https://www.dropbox.com/s/kx0ddu39lxup8nz/all_datasets.zip). Then, place the downloaded data under the folder `./dataset`.

3. **Search and Train-Test Models**. Examples of search and train scripts are in the `scripts.sh` file. For example,

```bash
# run search for classification task on JapaneseVowels dataset
python -u searcher.py --task_name classification --gpu 3 --population_size 1000 --data_name JapaneseVowels
# run the found architecture training and testing
python -u trainer.py --task_name classification --gpu 0 --population_size 1000 --method_name TimesNAS --data_name JapaneseVowels
```

### Notebook Example for Demonstration: `TimesNAS_Example_Demo.ipynb` *(Long-term Forecasting Task)* with `N = 100`

## Citation
```
TBD
```

## Acknowledgement
This project and datasets are constructed based on https://github.com/thuml/Time-Series-Library.
