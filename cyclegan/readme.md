This folder contains code to perform transformation on rendered images through CycleGAN

## File Structure
- `models` contains codes that are necessary for CycleGAN to transform images.
- `cyclegan_inference.ipynb` contains code to load checkpoints, and transform rendered images with CycleGAN generator.

## Pipeline
1. Download checkpoints from https://drive.google.com/drive/folders/1olY7J_FKLpSaBdA2e-LmkHvxE1V8oWL8?usp=sharing and place under this folder
2. Place images rendered with texture map generated from StyleGAN 2 within this folder
3. Execute the coding blocks in `cyclegan_inference.ipynb` to perform image transformation
