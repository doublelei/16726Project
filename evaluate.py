from pytorch_fid import fid_score
import numpy as np 
import sys

def calculate_FID(path_real, path_fake):
    fid = fid_score.calculate_fid_given_paths(paths=[path_real, path_fake], batch_size=64, device='cuda:0', dims=2048)
    return fid


if __name__ == '__main__':
    print(calculate_FID(sys.argv[1], sys.argv[2]))