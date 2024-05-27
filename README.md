<h1 align="center">
    mACPpred2
    <br>
<h1>

<h4 align="center">Standalone program for the paper "mACPpred 2.0: Integrating NLP-derived and conventional ML-based probabilistic features for accurate anticancer peptide identification using stacked deep learning"</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/nhattruongpham/mACPpred2?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/nhattruongpham/mACPpred2?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/nhattruongpham/mACPpred2?" alt="license"></a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-stared">Getting started</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a>
</p>

<p align="center"><img src="https://github.com/nhattruongpham/mACPpred2/raw/main/.github/mACPpred2_Workflow.png" height="250"/></p>

## Introduction
This repository 

## Installation
### Creating conda environment
```shell
conda create -n mACPpred2 python=3.9.12
```
```shell
conda activate mACPpred2
```

### Installing TensorFlow with CUDA support
```shell
conda install -c conda-forge cudatoolkit=11.7.0
```
```shell
python -m pip install nvidia-cudnn-cu11==8.6.0.163 --no-cache-dir
```
```shell
python -m pip install tensorflow==2.11.* --no-cache-dir
```
```shell
python -m pip install chardet --no-cache-dir
```
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
```
```
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
```
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
```
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### Installing bio-embeddings and re-installing PyTorch with CUDA support
```shell
python -m pip install bio-embeddings[seqvec] --no-cache-dir
```
```shell
python -m pip install scipy==1.10.1 --no-cache-dir
```
```shell
python -m pip install protobuf==3.20.* --no-cache-dir
```
```shell
python -m pip install bio-embeddings[all] --no-cache-dir
```
```shell
python -m pip uninstall torch
```
```shell
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
```

### Installing required specific packages
```shell
python -m pip install peptidy==0.0.1 --no-cache-dir
```
```shell
python -m pip install protlearn==0.0.3 --no-cache-dir
```
```shell
python -m pip install catboost==1.2 lightgbm==3.3.5 scikit-learn==0.24.2 xgboost==0.82 --no-cache-dir
```
## Getting started
### Cloning this repository
```
git clone https://github.com/nhattruongpham/mACPpred2.git
```
```
cd mACPpred2
```

## Downloading basline and final models
* Please download the baseline and final models via Zenodo at https://zenodo.org/
* For the baseline models, please extract and put all *.pkl files into the [models/baseline_models](https://github.com/nhattruongpham/mACPpred2/tree/main/models/baseline_models) folder.
* For the final models, please please extract and put all *.h5 files into [models/final_models](https://github.com/nhattruongpham/mACPpred2/tree/main/models/final_models) folder.

### Running prediction
#### Usage
```shell
CUDA_VISIBLE_DEVICES=<GPU_NUMBER> python predictor.py --input_file <PATH_TO_INPUT_FILE> --output_file <PATH_TO_OUTPUT_FILE>
```
#### Example
```shell
CUDA_VISIBLE_DEVICES=0 python predictor.py --input_file examples/test.fasta --output_file result.csv
```

## Citation
If you use this code or part of it, please cite the following papers:
```
@article{Sangaraju2024article,
  title={mACPpred 2.0: Integrating NLP-derived and conventional ML-based probabilistic features for accurate anticancer peptide identification using stacked deep learning},
  author={Sangaraju, Vinoth Kumar and Pham, Nhat Truong and Wei, Leyi and Yu, Xue and Manavalan, Balachandran},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

## References
