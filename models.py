# CMU 16-726 Learning-Based Image Synthesis / Spring 2022, Assignment 3
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - CycleGenerator     --> Used in the CycleGAN in Part 2
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
#   - PatchDiscriminator --> Used in the CycleGAN in Part 2
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ and forward methods in the
# DCGenerator, CycleGenerator, DCDiscriminator, and PatchDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Parameter

import pandas as pd
import pytorch3d
import itertools
import smplx
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures
)
import glob 
import pickle
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from PIL import Image
import random
import numpy as np
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1, scale_factor=2, norm='instance', high_res=False):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    if high_res:
        layers.extend([nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
                      nn.LeakyReLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=1, padding=0)])
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)


def up_conv_residual(in_channels, out_channels, kernel_size, stride=1, padding=1, scale_factor=2, norm='instance'):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(ResnetBlock(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='instance', init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    if norm == 'spectral':
        layers.append(SpectralNorm(conv_layer))
    else:
        layers.append(conv_layer)
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim, norm='instance', num_block=4):
        super(DCGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        self.model = [nn.ConvTranspose2d(noise_size, conv_dim * (2**(num_block-1)), 4, 1, 0, bias=False),
                      nn.InstanceNorm2d(conv_dim * (2**(num_block-1))),
                      nn.LeakyReLU()]

        for i in range(num_block-1):
            self.model.extend([up_conv(conv_dim * (2**(num_block-i-1)), conv_dim * (2**(num_block-i-2)),
                              3, norm='instance', high_res=num_block>4), nn.LeakyReLU()])

        self.model.extend([up_conv(conv_dim, 3, 3, norm=''), nn.Tanh()])
        self.model = nn.Sequential(*self.model)

        R, T = look_at_view_transform(5, 0, 0) 
        cameras = FoVPerspectiveCameras(R=R, T=T)
        raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, bin_size=0, faces_per_pixel=10)
        self.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                                   shader=SoftPhongShader(cameras=cameras, lights=PointLights(location=[[0.0, 0.0, 2.0]])))
        verts, faces, aux = load_obj("stylegan3/smplx_uv.obj")
        self.verts_uvs = aux.verts_uvs  # (1, V, 2)
        self.faces_uvs = faces.textures_idx  # (1, F, 3)
        self.faces = faces.verts_idx
        self.verts = verts
        
        self.smplx_files = glob.glob("stylegan3/datasets/smplx_gt/*/*.pkl")
    
        # intialize a list of 1008 cameras for random indexing later
        # alter dist, elev, azim to generate diverse camera viewpoints
        distances = list(np.linspace(3, 5, 20))
        elevations = [-45 + 15*i for i in range(7)]
        azimuth = [180 - 15*i for i in range(24)]
        camera_feat = list(itertools.product(distances, elevations, azimuth))
        R_T = [pytorch3d.renderer.cameras.look_at_view_transform(dist=c[0], elev=c[1], azim=c[2]) for c in camera_feat]
        self.cameras = [pytorch3d.renderer.FoVPerspectiveCameras(R=rt[0], T=rt[1], fov=60) for rt in R_T]
        self.smplx_model = smplx.create('stylegan3/datasets/models', 'smplx', use_pca=False)

    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        texture_images = self.model(z)
        gts = [pickle.load(open(random.choice(self.smplx_files), "rb"), encoding="latin1") for _ in range(len(texture_images))]
        smplx_gts = [self.smplx_model(
            betas=torch.tensor(gt['betas'][:, :10], dtype=torch.float, device=texture_images.device), 
            global_orient=torch.tensor(gt['global_orient'], dtype=torch.float, device=texture_images.device),
            body_pose=torch.tensor(gt['body_pose'], dtype=torch.float, device=texture_images.device),
            left_hand_pose=torch.tensor(gt['left_hand_pose'], dtype=torch.float, device=texture_images.device),
            right_hand_pose=torch.tensor(gt['right_hand_pose'], dtype=torch.float, device=texture_images.device),
            transl=torch.tensor(gt['transl'], dtype=torch.float, device=texture_images.device),
            expression=torch.tensor(gt['expression'], dtype=torch.float, device=texture_images.device), 
            jaw_pose=torch.tensor(gt['jaw_pose'], dtype=torch.float, device=texture_images.device),
            leye_pose=torch.tensor(gt['leye_pose'], dtype=torch.float, device=texture_images.device),
            reye_pose=torch.tensor(gt['reye_pose'], dtype=torch.float, device=texture_images.device), pose2rot=True) for gt in gts]
        
        tex = Textures(verts_uvs=[self.verts_uvs.to(texture_images.device)]*len(texture_images), faces_uvs=[self.faces_uvs.to(texture_images.device)]*len(texture_images), maps=texture_images.permute((0,2,3,1)))
        # Initialise the mesh with textures
        meshes = [Meshes(verts=[smplx_gt.vertices.squeeze().to(texture_images.device)], faces=[self.faces.to(texture_images.device)], textures=texture) for smplx_gt, texture in zip(smplx_gts, tex)]
        
        self.renderer.to(texture_images.device)
        rendered_images = torch.cat([self.renderer(mesh, cameras=random.choice(self.cameras).to(texture_images.device)) for mesh in meshes], axis=0)

        return texture_images, rendered_images.permute((0, 3, 1, 2))[:, :3, :, :]


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, norm, high_res=False):
        super(ResnetBlock, self).__init__()
        if high_res:
            self.conv_layer = nn.Sequential(nn.ReflectionPad2d(1),
                                            conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=0, norm=norm),
                                            nn.LeakyReLU(),
                                            nn.ReflectionPad2d(1),
                                            conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=0, norm=norm))
        else:
            self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out



class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=64, norm='instance', num_layers=3):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        model = []
        model.extend([conv(3, conv_dim, 4, norm=norm), nn.LeakyReLU(0.2, True)])
        for i in range(num_layers):
            model.extend([conv(conv_dim*(2**i), conv_dim*(2**(i+1)), 4, norm=norm), nn.LeakyReLU(0.2, True)])
        model.extend([conv(conv_dim*(2**num_layers), 1, 4, norm=None)])
        self.model = nn.Sequential(*model)

    def forward(self, x):

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        x = self.model(x)

        return x


class PatchDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=64, num_layers=2, norm='instance'):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # Hint: it should look really similar to DCDiscriminator.
        model = []
        model.extend([conv(3, conv_dim, 4, norm=norm), nn.LeakyReLU(0.2, True)])
        for i in range(num_layers):
            model.extend([conv(conv_dim*(2**i), conv_dim*(2**(i+1)), 4, stride=1 if i == num_layers-1 else 2, norm=norm), nn.LeakyReLU(0.2, True)])
        model.extend([conv(conv_dim*(2**num_layers), 1, 4, stride=1, norm=None)])
        self.model = nn.Sequential(*model)

    def forward(self, x):

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        x = self.model(x)
        return x
