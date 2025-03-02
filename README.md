# *STFF: Spatio-Temporal and Frequency Fusion for Video Compression Artifact Removal*


The *PyTorch* implementation for the [STFF: Spatio-Temporal and Frequency Fusion for
Video Compression Artifact Removal] which will be accepted by [IEEE Transactions on Broadcasting].


## 1. Pre-request

### 1.1. Environment
Suppose that you have installed CUDA 11.3, then:
```bash
conda create -n STFF python=3.8 -y  
conda activate STFF

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

git clone --depth=1 https://github.com/Stars-WMX/STFF
cd STFF
```

### 1.2. Dataset

Please check [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset).

### 1.3. Create LMDB
We now generate LMDB to speed up IO during training.
```bash
CUDA_VISIBLE_DEVICES=0 python create_lmdb_MFQEV2.py
```

## 2. Train

We utilize 8 NVIDIA GeForce RTX 3090 GPUs for training, please see script.sh for the specific training commands.

## 3. Test         
Pretrained models can be found here:

We utilize 1 NVIDIA GeForce RTX 3090 GPU for testing:
```bash
python test.py
```

If you want to save the processed results, please run:
```bash
python test_save.py
```
The script will automatically save the processed outputs, including logs, model checkpoints, and visualizations if applicable.


## Citation
If you find this project is useful for your research, please cite:


## Acknowledgements
This work is based on [STDF-Pytoch](https://github.com/RyanXingQL/STDF-PyTorch) and [OVQE](https://github.com/pengliuhan/OVQE). Thank them for sharing the codes.
