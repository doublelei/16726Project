## Data Prepare
### 1. Prepare SURREAL Texture
Download [SURREAL DATASET](https://www.di.ens.fr/willow/research/surreal/data/) and put it to stylegan3/datasets

### 2. Prepare AGORA Images
Download [AGORA DATASET](https://agora.is.tue.mpg.de/index.html) and put it to stylegan3/datasets

### 3. AGORA Preprocess


### 4. Prepare SMPLX Model
Download [SMPLX Model](https://smpl-x.is.tue.mpg.de/download.php) and put it to stylegan3/datasets




## Install
### 1. Install the [StyaleGAN3 environments](https://github.com/NVlabs/stylegan3)

### 2. Install the [pytorch3d environments](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

### 3. Install SMPLX by
```
pip install smplx 
```

## Train

### 1. Pretrain the StyleGAN3 by
```
cd stylegan3 & python train.py --outdir=training-runs --cfg=stylegan3-t --data==/datasetsall_texture.zip --gpus=1 --batch=4 --gamma=8.2 --mirror=1 --rendered=False 
```

### 2. Train the StyleGAN3 for rendered images 
```
cd stylegan3 & python train.py --outdir=training-runs --cfg=stylegan3-t --data=datasets/AGORA_image_256x256.zip --gpus=1 --batch=4 --gamma=8.2 --mirror=1 --rendered=True --resume=RESUME_PATH
```

### 3. Train the CycleGAN for real images
```
python cyclegan/train.py --dataroot datasets/surreal2agora --name surreal2agora --model cycle_gan --use_wandb
```
Alternatively, you can download [CycleGAN checkpoints](https://drive.google.com/drive/folders/1olY7J_FKLpSaBdA2e-LmkHvxE1V8oWL8?usp=sharing) and place under `./cyclegan/` for inference use.

## Inference
### 1. Generate textures by
```
cd stylegan3 & python gen_images.py --outdir=textures --trunc=1 --seeds=2 --network=MODEL_PKL_PATH
```

### 2. Render person images by
```

```

### 3. Transfer the rendered image to real images by
```
python cyclegan_inference.py --input <path_to_rendered_image_folder> --output <path_to_CycleGAN_output_folder>
```

## Evalute 
### Calculate FID by
```
python evaluate.py REAL_PATH GENERATED_PATH
```
