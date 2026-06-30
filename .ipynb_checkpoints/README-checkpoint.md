## APG-DynaFFNet
 This repository contains the official implementation of APG-DynaFFNet, a novel framework for animal pose estimation that integrates species-aware anatomical priors with dynamic multi-granularity feature fusion.

## Preparation

### 0. Requirements
- Linux
- CUDA (devel/runtime) ≥11.6
- conda

### 1. Clone
```bash
git clone https://github.com/xq1999-glitch/APG-DynaFFNet.git && cd sharpose
```

### 2. Environment
```bash
conda create -n mmlab_0.x python=3.8
conda activate mmlab_0.x

### For GPUs before RTX40XX
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
git clone https://github.com/open-mmlab/mmpose.git && cd mmpose && git switch 0.x
pip install -r requirements.txt
pip install -v . 
cd ..

### For RTX40XX (mod & build mmcv-full from source)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/open-mmlab/mmcv.git && cd mmcv && git checkout tags/v1.7.1
pip install -r requirements.txt
sed -i "160s/self._use_replicated_tensor_module/getattr(self, '_use_replicated_tensor_module', None)/g" mmcv/parallel/distributed.py
sed -i 's/-std=c++14/-std=c++17/' setup.py
MMCV_WITH_OPS=1 pip install -v .
cd ..

git clone https://github.com/open-mmlab/mmpose.git && cd mmpose && git switch 0.x
pip install -r requirements.txt
pip install -v . 
cd ..

### Common 
pip install -r requirements.txt
```

### 3. Dataset
You can also use any dataset that follows the COCO keypoint format

### 4. Checkpoints
链接: https://pan.baidu.com/s/1xTYPEhJjFhOVLqzZD7J50w?pwd=2f9j 提取码: 2f9j 

链接: https://pan.baidu.com/s/1R6jBknTx4-7vuGWYKX7Gkw?pwd=pgm3 提取码: pgm3 

## Evaluation
python test.py path/to/configs.py path/to/best_AP_epoch_XX.pth --out  path/to/out.json --work-dir path/to/workdir --gpu-id 0  --eval mAP

## Training
python train.py path/to/configs.py --work-dir path/to/workdir --gpu-id 0 --cfg-options model.pretrained=path/to/pretrained.pth

## Acknowledgement
- MMPose
- TokenPose
- CF-ViT
- ViTPose
- MAE
