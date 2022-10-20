
from argparse import ArgumentParser
import json
import os
import pandas as pd
# Globals


# Data directories
global labels_directory
global images_directory
global masks_directory


"""
Example of usage :

    python3 ./src/data/create_table.py -ldir './data/processed/GRAZ/labels' -idir './data/processed/GRAZ/image' -mdir './data/processed/GRAZ/image_mask'

The table will be saved in the same folder as your image directory
"""


def parse_arguments():
    global labels_directory
    global images_directory
    global masks_directory
    global table_path
    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('-ldir', '--labels_directory', help='directory of labels',
                        required=True)
    parser.add_argument('-idir', '--images_directory',
                        help='directory of microscopic images',
                        required=False)
    parser.add_argument('-mdir', '--masks_directory',
                        help='directory of images masks',
                        required=False)


    args = parser.parse_args()
    labels_directory = args.labels_directory
    images_directory = args.images_directory
    masks_directory = args.masks_directory

def create_table( labels_directory,
                images_directory,
                masks_directory,table_name=None
                ):
    if table_name is None:
        table_path=os.path.dirname(images_directory)+'/data.tsv'
    else :
        table_path=os.path.join(os.path.dirname(images_directory),table_name)
    df=pd.DataFrame(data=[],columns=['image','mask','class','class_nr'])
    i=0
    classes_df=pd.read_csv('./classes.tsv',sep=',')
    classes=classes_df['class'].tolist()
    for (img_file,label_file,mask_file) in zip(os.listdir(images_directory),os.listdir(labels_directory),os.listdir(masks_directory)):
        label_filename=os.path.join(labels_directory,label_file)
        f=open(label_filename,'r')
        label_data=json.load(f)
        file_class=label_data['class']
        df.loc[i]=[img_file,mask_file,file_class,classes.index(file_class)]
        i+=1
    df.to_csv(table_path,index=False,header=True)
    return(df)


if __name__ == '__main__':
    parse_arguments()
    create_table(labels_directory,
                images_directory,
                masks_directory)

