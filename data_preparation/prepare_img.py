import matplotlib.pyplot as plt
import numpy as np
import smplx
import cv2

import pickle
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import os

from utils import get_final_image, get_paths

def get_paths(raw_name, raw_index, dataset):
    '''
    raw_name: image_name, e.g., ag_trainset_renderpeople_bfh_archviz_5_10_cam02_00001.png
    raw_index: index of person in the image (dataframe["min_occ_idx"])
    dataset: for example, train_0
    '''

    # generate img path
    img_name = raw_name.replace('.png','_1280x720.png')
    img_name_ele = img_name.split("_")
    img_path = "./{}/{}".format(dataset, img_name)


    img_name_ele[-2] = "0"+img_name_ele[-2]
    if (raw_index+1<10):
        img_name_ele.insert(-1,"0000{}".format(raw_index+1)) 
    else:
        img_name_ele.insert(-1,"000{}".format(raw_index+1)) 
    
    # generate target path
    tgt_path = "_".join(img_name_ele) # for example, ag_trainset_renderpeople_bfh_archviz_5_10_cam02_000001_00001_1280x720.png
    tgt_path = "./dataset/{}/{}_{}".format(dataset.split("_")[0], dataset, tgt_path)

    
    # generate mask path
    mask_folder = "_".join(img_name_ele[:5])

    if (img_name_ele[-4].startswith("cam")):
        img_name_ele.insert(-4,"mask")
    else:
        img_name_ele.insert(-3,"mask")

    mask_name = "_".join(img_name_ele) # for example, ag_trainset_renderpeople_bfh_archviz_5_10_mask_cam02_000001_00001_1280x720.png
    mask_path = "./train_masks_1280x720/train/{}/{}".format(mask_folder,mask_name) 

    return img_path, tgt_path, mask_path

def get_final_image(img_path, tgt_path, mask_path):
    try:
        # get mask image of selected person
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0) # for foreground (person)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        new_mask = np.logical_not(mask) # for background => we want white background eventually
        masked_img[new_mask]=255 # new_mask contains boolean entries and therefore can be used in this way
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        
        # crop image from the mask
        c = np.nonzero(mask)
        x_min = int(min(c[1]))
        x_max = int(max(c[1]))
        y_min = int(min(c[0]))
        y_max = int(max(c[0]))
        cropped_img = masked_img[y_min:y_max, x_min:x_max]

        w = x_max - x_min
        h = y_max - y_min

        # scale the cropped image
        scale = 200/max(w, h)
        resized_w = int(scale*w)
        resized_h = int(scale*h)
        resized_cropped_img = cv2.resize(cropped_img, (resized_w, resized_h))

        # generate final result (256*256 white background image)
        final_result = np.zeros((256,256,3))
        final_c_x = 128
        final_c_y = 128
        final_result += 255

        final_result[int(final_c_y-resized_h/2):int(final_c_y+resized_h/2),int(final_c_x-resized_w/2):int(final_c_x+resized_w/2)] = resized_cropped_img
        final_result = final_result.astype(int) # necessary

        plt.imshow(final_result)
        plt.axis("off")
        plt.savefig("{}".format(tgt_path))
    except:
        print (img_path, tgt_path, mask_path)
        pass

def main():
    pool = Pool(os.cpu_count()-4) # 9 cpu available to use

    train_df = pd.read_csv("dev_dataframe.csv")
    inputs = zip(list(train_df["src_img_path"]), list(train_df["tgt_img_path"]), list(train_df["mask_path"]))

    try:
        pool.starmap(get_final_image, tqdm(inputs, total=len(list(train_df["src_img_path"]))))
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()