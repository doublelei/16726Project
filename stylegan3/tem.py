import pickle
import cv2
import torch
import imageio
with open('training-runs/00010-stylegan2-texture-256x256-gpus8-batch16-gamma8.2/network-snapshot-008240.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda() 
z1 = torch.randn([1, G.z_dim]).cuda()  
z2 = torch.randn([1, G.z_dim]).cuda()  
c = None           
images = []
k = 50
for i in range(0, k+1):            
    img = G(z1*(k-i)/k+z2*i/k, c)                         
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    images.append(img[0].cpu().numpy()[:, :, ::-1])

imageio.mimsave('inter/result.gif', images, fps=10)
for i in range(0, k+1):
    cv2.imwrite("inter/{}.png".format(i), images[i])