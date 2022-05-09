import glob
import os
import imageio
import utils
import torch
import numpy as np
import cv2
from PIL import Image
from data_loader import *


def generate_gif_dc(output_path):
    image_files = sorted(glob.glob(os.path.join(output_path, 'sample*.png')))
    images = []
    print(len(image_files))
    for f in image_files[4::20]:
        image = imageio.imread(f)
        image = cv2.putText(img=np.copy(image), text=str(int(os.path.split(f)[-1][7:13])), org=(0, 12), fontFace=2, fontScale=0.5, color=(255,0,0), thickness=1)
        images.append(image)
    imageio.mimsave(os.path.join(output_path, os.path.split(output_path)[-1]+'sample.gif'), images, fps=2)

def generate_gif_cycle(output_path):
    image_files_X_Y = sorted(glob.glob(os.path.join(output_path, 'sample*X-Y.png')))
    image_files_Y_X = sorted(glob.glob(os.path.join(output_path, 'sample*Y-X.png')))
    images_X_Y = []
    images_Y_X = []
    for f in image_files_X_Y[::10]:
        image = imageio.imread(f)
        image = cv2.putText(img=np.copy(image), text=str(int(os.path.split(f)[-1][7:13])), org=(0, 12), fontFace=2, fontScale=0.5, color=(255,0,0), thickness=1)
        images_X_Y.append(image)
    imageio.mimsave(os.path.join(output_path, os.path.split(output_path)[-1]+'sample_X_Y.gif'), images_X_Y, fps=2)
    for f in image_files_Y_X[::10]:
        image = imageio.imread(f)
        image = cv2.putText(img=np.copy(image), text=str(int(os.path.split(f)[-1][7:13])), org=(0, 12), fontFace=2, fontScale=0.5, color=(255,0,0), thickness=1)
        images_Y_X.append(image)
    imageio.mimsave(os.path.join(output_path, os.path.split(output_path)[-1]+'sample_Y_X.gif'), images_Y_X, fps=2)


def generate_gif_dc_lantent(G, output_path, noise_size=100, num_phase=10, transition_frames=10):
    noise = utils.to_var(torch.rand(num_phase, noise_size) * 2 - 1).unsqueeze(2).unsqueeze(3)
    all_noise = []
    for i in range(num_phase):
        for j in range(transition_frames):
            all_noise.append((noise[i] * (transition_frames - j) + noise[(i+1)%num_phase] * j) / transition_frames)
    generated_images = G(torch.stack(all_noise, 0)).detach().cpu().numpy().transpose(0, 2, 3, 1)
    imageio.mimsave(os.path.join(output_path, os.path.split(output_path)[-1]+'interplot.gif'), ((generated_images+1)/2*255).astype(np.uint8), fps=10)


def generate_video_cycle(G, output_path, input_path, image_size):
    inputs = sorted(glob.glob(os.path.join(input_path, '*.jpg')))
    transform = transforms.Compose([
        transforms.Resize(image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    outputs = []
    input_images = []
    with torch.no_grad():
        for i in inputs[200:400]:
            image = Image.open(i).convert("RGB")
            input_images.append(image)
            tensor_image = transform(image)
            output = G(tensor_image.unsqueeze(0).cuda())
            outputs.append(output.detach().cpu().numpy().transpose(0, 2, 3, 1))
        outputs = np.concatenate(outputs)
        imageio.mimsave(os.path.join(output_path, 'video_output.gif'), ((outputs+1)/2*255).astype(np.uint8), fps=20)
        imageio.mimsave(os.path.join(output_path, 'video_input.gif'), input_images, fps=20)

if __name__ == '__main__':
    # Test generate_gif_dc
    generate_gif_dc('output/vanilla/me_deluxe_color_translation_cutout_spectral_patch')
    # Test generate_gif_cycle
    for folder in glob.glob("output/cyclegan/apple2orange_256/*"):
        print(folder)
        generate_gif_cycle(folder)
    