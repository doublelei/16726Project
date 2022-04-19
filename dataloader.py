from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import pandas as pd
import cv2
import pickle
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures
)

import itertools


# Rendered Images of Full Texture
class Agora_Data(Dataset):
    """Load data under folders"""
    def __init__(self):

        file_path = "./agora_dataframe.csv"
        self.df = pd.read_csv(file_path)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # intialize a list of 1008 cameras for random indexing later
        # alter dist, elev, azim to generate diverse camera viewpoints
        distances = list(range(2,8))
        elevations = [-45 + 15*i for i in range(7)]
        azimuth = [180 - 15*i for i in range(24)]
        camera_feat = list(itertools.product(distances, elevations, azimuth))
        R_T = [pytorch3d.renderer.cameras.look_at_view_transform(dist=c[0], elev=c[1], azim=c[2]) for c in camera_feat]
        self.cameras = [pytorch3d.renderer.FoVPerspectiveCameras(R=rt[0], T=rt[1], fov=60) for rt in R_T]


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        # random select a camera from the list
        i = np.random.randint(0,len(self.cameras), 1)[0]
        camera = self.cameras[i]

        # pose
        j = np.random.randint(0,self.df.shape[0], 1)[0]
        smplx_f = open(self.df.iloc[j]["smplx_path"], "rb")
        smplx = pickle.load(smplx_f, encoding="latin1")

        # image
        img = cv2.imread(self.df.iloc[idx]["tgt_img_path"])

        return camera, smplx, img


def get_data_loader(args):
    """
    Creates training and test data loaders
    """
    agora_dataset = Agora_Data()
    dloader = DataLoader(dataset=agora_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return dloader