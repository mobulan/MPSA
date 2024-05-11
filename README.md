# Multi-granularity Part Sampling Attention for Fine-grained Visual Classification
Source code of the paper Multi-granularity Part Sampling Attention for Fine-grained Visual Classification
## Code Running
### Requirements
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy matplotlib pandas tensorboard scipy tqdm
pip install yacs opencv-python opencv-contrib-python timm einops gpustat
```
### Datasets
You may specific your dataset path in the `/config/{settting}.yaml` under `data/datasets`. Then please check if there is any conflict with the ip in line 103 of `settings/setup_functions.py`.
### Training
1. Put the pre-trained model in `/pretrained/`, and rename it to `Swin Base.npz`.
2. Change the log name and cuda visible by modifing line 13,14 in `/setup.py`.
3. Running the following code according to you pytorch version:
#### If pytorch < 1.12.0
```python -m torch.distributed.launch --nproc_per_node 2 main.py ```
#### If pytorch >= 1.12.0
```torchrun --nproc_per_node 2 main.py```
You need to change the number behind the `-nproc_per_node` to your number of GPUs.
