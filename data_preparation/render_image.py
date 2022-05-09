import argparse
import os
import torch
import numpy as np
import smplx
import cv2

# libraries for reading data from files
from PIL import Image
import pickle
import pytorch3d
import pandas as pd

# Data structures and functions for rendering

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures
)
from pytorch3d.io import load_obj

# add path for demo utils functions 
import sys
import itertools
from multiprocessing import Pool
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentDefaultsHelpFormatter(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # For File I/O
    parser.add_argument("--texture", help="file path of texture maps")
    parser.add_argument("--output", help="output path to store render results")
    parser.add_argument("--agora", help="path of AGORA folder")
    parser.add_argument("--smplx", help="path of smplx model")

    parser.add_argument("--device", default="cpu")


def build_smplx_model_dict(smplx_model_dir, device):
    gender2filename = dict(neutral='SMPLX_NEUTRAL.npz', male='SMPLX_MALE.npz', female='SMPLX_FEMALE.npz')
    gender2path = {k:os.path.join(smplx_model_dir, v) for (k, v) in gender2filename.items()}
    gender2model = {k:smplx.body_models.SMPLX(v).to(device) for (k, v) in gender2path.items()}

    return gender2model

args = parse_args()
# intialize a list of 1008 cameras for random indexing later
# alter dist, elev, azim to generate diverse camera viewpoints
distances = np.arange(3,5,0.1) # 3～5 
elevations = [-45 + 15*i for i in range(7)]
azimuth = [180 - 15*i for i in range(24)]
camera_feat = list(itertools.product(distances, elevations, azimuth))
R_T = [pytorch3d.renderer.cameras.look_at_view_transform(dist=c[0], elev=c[1], azim=c[2]) for c in camera_feat]
cameras = [pytorch3d.renderer.FoVPerspectiveCameras(R=rt[0], T=rt[1], fov=60) for rt in R_T]

device = args.device
smplx_model_dir = args.smplx #"./models/smplx/"
smplx_models_dict = build_smplx_model_dict(smplx_model_dir, device=device)
smplx_uv = "smplx_uv.obj"

verts, faces, aux = load_obj(smplx_uv)
verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)

agora = pd.read_csv("{}/dataset/agora_dataframe.csv".format(args.agora))
poses = list(agora["smplx_path"])
poses = [p.split(".")[0]+".pkl" for p in poses]

textures = os.listdir(args.texture)


def generate_image(camera_idx, pkl_fp_idx, text_fp_idx, tgt_fp, args):
    '''
    render image given camera, pose and texture
    '''
    try:
        camera = cameras[camera_idx]
        pkl_fp = args.agora+"/"+poses[pkl_fp_idx]


        smplx_f = open(pkl_fp, "rb")
        smplx_params = pickle.load(smplx_f, encoding="latin1")
        gender = smplx_params['gender']

        # if gender == "female" and text_fp_idx > 451:
        #     text_fp_idx = np.random.choice(451, 1)[0]
        # #text_fp = "./all_texture/"+textures[text_fp_idx]
        text_fp = "./gen_texture/"+textures[text_fp_idx]

        for k, v in smplx_params.items():
            if type(v) == np.ndarray:
                if 'hand_pose' in k:
                    v = v[:, :6]
                smplx_params[k] = torch.FloatTensor(v)

        smplx_output = smplx_models_dict[gender](**smplx_params)

        with Image.open(text_fp) as image:
            np_image = np.asarray(image.convert("RGB")).astype(np.float32)
        texture_images = torch.from_numpy(np_image / 255.)[None]

        # Create a textures object
        tex = Textures(verts_uvs = verts_uvs, faces_uvs=faces_uvs, maps=texture_images)

        # Initialise the mesh with textures
        # mean_vert = torch.mean(smplx_output.vertices.squeeze(), axis=0)
        min_vert = torch.min(smplx_output.vertices.squeeze(), axis=0)[0]
        max_vert = torch.max(smplx_output.vertices.squeeze(), axis=0)[0]
        vert = smplx_output.vertices.squeeze() - (max_vert+min_vert)/2 # shift vertices to make object at center
        meshes = Meshes(verts=[vert], faces=[faces.verts_idx], textures=tex)



        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 256x256. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. 
        raster_settings = RasterizationSettings(
            image_size=1024, 
            blur_radius=0.0, 
            faces_per_pixel=5, 
        )

        # Place a point light in front of the person. 
        lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=camera,
                lights=lights
            )
        )

        rendered_img = renderer(meshes.to(device), lights=lights.to(device), cameras=camera.to(device))
        final_img = cv2.convertScaleAbs(rendered_img[0, ..., :3].detach().cpu().numpy(), alpha=(255.0))
        cv2.imwrite("./gen_output/{}.png".format(tgt_fp), np.flip(final_img, axis=-1))
    except:
        print(pkl_fp, text_fp)
        pass



def main():
    pool = Pool(os.cpu_count()-4) # 9 cpu available to use

    #textures = os.listdir("./all_texture") # 478 male + 451 female = 929

    # c_ids = np.random.choice(20*7*24, 12251)
    # pkl_ids = np.random.choice(10251, 12251)
    # text_ids = np.random.choice(929, 12251)
    # tgt_fps = np.arange(12251)
    c_ids = np.random.choice(20*7*24, 1001)
    pkl_ids = np.random.choice(10251, 1001)
    text_ids = np.arange(1001)
    tgt_fps = np.arange(1001)
    inputs = zip(c_ids, pkl_ids, text_ids, tgt_fps) 
    
    

    try:
        pool.starmap(generate_image, tqdm(inputs, total=1001))
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()