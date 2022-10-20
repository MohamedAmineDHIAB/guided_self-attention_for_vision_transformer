
import json
import os
import random
from argparse import ArgumentParser

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
# Globals
# Random seed used for augmentation
RANDOM_SEED = 2022


# Data directories
global images_directory
global masks_directory

# Dataset to be augmented
global data_file

# Where you want your augmented data to be saved
global augmented_directory

"""
Example of usage :

    python3 ./src/data/augment.py -idir './data/image' -mdir './data/image_mask' -df './data/data.tsv'

"""


def parse_arguments():
    global images_directory
    global masks_directory
    global data_file
    parser = ArgumentParser(description='Parse arguments')

    parser.add_argument('-idir', '--images_directory',
                        help='directory of microscopic images',
                        required=False)
    parser.add_argument('-mdir', '--masks_directory',
                        help='directory of images masks',
                        required=False)

    parser.add_argument('-df', '--data_file',
    help='dataset to be augmented',
    required=True)

    args = parser.parse_args()
    images_directory = args.images_directory
    masks_directory = args.masks_directory
    data_file=args.data_file


def augment(
     images_directory,
     masks_directory,data_file):

    print('Augmenting the training dataset...\n')
    # create augmentation directory
    augmented_directory =os.path.join(os.path.dirname(images_directory), 'augmented')
    os.mkdir(augmented_directory)
    # Creating directories of augmented images, masks and their corresponding labels
    aug_img_dir= os.path.join(augmented_directory, 'image')
    aug_mask_dir= os.path.join(augmented_directory, 'image_mask')
    aug_labels_dir= os.path.join(augmented_directory, 'labels')
    os.mkdir(aug_img_dir)
    os.mkdir(aug_mask_dir)
    os.mkdir(aug_labels_dir)
    # importing dataframe to be augmented
    df=pd.read_csv(data_file)
    # define array containing random transformation indexes for dominant class
    random_transform_index= np.random.randint(5,size=len(df))+1
    i=0
    for _,row in tqdm(df.iterrows(),total=len(df)):
        img_file,label,mask_file=row['image'],row['class'],row['mask']
        # define the transformations to be used
        transformations = [
                A.Compose([
                ]),A.Compose([
                        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=-1, rotate_limit=0, p=1),
                        A.RandomCrop(height=1024, width=1024,p=1),
                ]),A.Compose([
                        A.HorizontalFlip(p=1),
                        A.CLAHE(p=1),
                ]),A.Compose([
                        A.MultiplicativeNoise(multiplier=[0.25, 1.5], elementwise=True, per_channel=True, p=1)
                ]),A.Compose([
                        A.MedianBlur(blur_limit=11, p=1)
                ]),A.Compose([
                        A.RandomBrightnessContrast(brightness_limit=-0.25, p=1)])
            ]
        if label != 'Tubus - dys':
            # Getting the image and mask files and copying them to the augmentation directory
            image_path=os.path.join(images_directory,img_file)
            mask_path=os.path.join(masks_directory,mask_file)
            img=np.array(Image.open(image_path))
            mask=np.array(Image.open(mask_path))

            for j,transform in enumerate(transformations):
                with open(os.path.join(aug_labels_dir,f'label_{i+1}_{j+1}.json'),'w') as f:
                    json.dump({'class':label},f)
                random.seed(RANDOM_SEED)
                transformed = transform(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                augmented_image = Image.fromarray(transformed_image)
                augmented_mask = Image.fromarray(transformed_mask)
                aug_img_filepath=os.path.join(aug_img_dir,f'image_{i+1}_{j+1}.jpg')
                aug_mask_filepath=os.path.join(aug_mask_dir,f'mask_{i+1}_{j+1}.jpg')
                augmented_image.save(aug_img_filepath)
                augmented_mask.save(aug_mask_filepath)
        else :

            # Getting the image and mask files and copying them to the augmentation directory
            image_path=os.path.join(images_directory,img_file)
            mask_path=os.path.join(masks_directory,mask_file)
            img=np.array(Image.open(image_path))
            mask=np.array(Image.open(mask_path))
            sub_transformations = [A.Compose([
            ]),transformations[random_transform_index[i]]]
            for j,transform in enumerate(sub_transformations):
                with open(os.path.join(aug_labels_dir,f'label_{i+1}_{j+1}.json'),'w') as f:
                    json.dump({'class':label},f)
                random.seed(RANDOM_SEED)
                transformed = transform(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']
                augmented_image = Image.fromarray(transformed_image)
                augmented_mask = Image.fromarray(transformed_mask)
                aug_img_filepath=os.path.join(aug_img_dir,f'image_{i+1}_{j+1}.jpg')
                aug_mask_filepath=os.path.join(aug_mask_dir,f'mask_{i+1}_{j+1}.jpg')
                augmented_image.save(aug_img_filepath)
                augmented_mask.save(aug_mask_filepath)
        i+=1
    return()



if __name__ == '__main__':
    parse_arguments()
    augment(
     images_directory,
     masks_directory,
     data_file)


