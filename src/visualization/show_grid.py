"""
Plot a grid of cols x rows images and use their labels as respective titles

"""

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def show_grid( images_directory,df : pd.DataFrame, cols=3,rows=3,figsize=(8,8)):
    figure = plt.figure(figsize=figsize)
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img_name, label = df.iloc[i]['image'],df.iloc[i]['class']
        img=mpimg.imread(images_directory+img_name)
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")


