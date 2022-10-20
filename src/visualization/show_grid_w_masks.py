"""
Plot a grid of cols x rows images and use their labels as respective titles.
Plot the corresponding masks on top of the images with a 50% opacity.

"""


import pandas as pd
from matplotlib import pyplot as plt
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image
import torchvision.transforms.functional as F
import numpy as np
def show_grid_w_masks(images_directory,masks_directory,df : pd.DataFrame, cols=3,rows=3,figsize=(8,8)):
    figure = plt.figure(figsize=figsize)
    for i in range(1, cols * rows + 1):
        img_name,mask_name, label = df.iloc[i]['image'],df.iloc[i]['mask'],df.iloc[i]['class']
        img=read_image(images_directory+img_name)
        mask=read_image(masks_directory+mask_name)>0
        img_w_mask=draw_segmentation_masks(img,mask,alpha=0.5)
        img_w_mask=np.asarray(F.to_pil_image(img_w_mask))
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img_w_mask.squeeze(), cmap="gray")

