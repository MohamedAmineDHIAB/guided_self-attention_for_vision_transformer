import cv2


from argparse import ArgumentParser
import json
import os
import pandas as pd
from src.data.create_table import create_table
from tqdm import tqdm

def extract_crops_all(img,width,height,img_save_dir=None,filename=None,label=None,label_save_dir=None):
    h,w=img.shape[0],img.shape[1]
    is_color_img=len(img.shape)==3
    list_crops=[]
    possible_ys=list(range(0,h-height+1,420))
    if h % height != 0:
        possible_ys.append(h-height)
    possible_xs=list(range(0,w-width+1,420))
    if w % width != 0:
        possible_xs.append(w-width)
    for j in possible_ys:
        for i in possible_xs:
            cropped_img=img[j:j+height,i:i+width,:] if is_color_img else img[j:j+height,i:i+width]
            if img_save_dir and filename:
                prep_img_filepath=os.path.join(img_save_dir,f'{filename}_{i+1}_{j+1}.jpg')
                cv2.imwrite(prep_img_filepath,cropped_img)
            else:
                raise Exception('You must provide a save directory and a file name')
            if label and label_save_dir:
                with open(os.path.join(label_save_dir,f'{filename}_{i+1}_{j+1}.json'),'w') as f:
                    json.dump({'class':label},f)
            
            list_crops.append(cropped_img)
    return(list_crops)

def preprocess(preprocess_cfg,img_dir,mask_dir,main_df,train_fraction,validation_fraction,random_seed):
    print('Preprocessing the training dataset...\n')
    # create preprocessed directory
    preprocessed_directory =os.path.join(os.path.dirname(img_dir), '../preprocessed')
    os.makedirs(preprocessed_directory,exist_ok=True)
    # Creating directories of preprocessed images, masks and their corresponding labels
    prep_img_dir= os.path.join(preprocessed_directory, 'image')
    prep_mask_dir= os.path.join(preprocessed_directory, 'image_mask')
    prep_labels_dir= os.path.join(preprocessed_directory, 'labels')
    os.makedirs(prep_img_dir,exist_ok=True)
    os.makedirs(prep_mask_dir,exist_ok=True)
    os.makedirs(prep_labels_dir,exist_ok=True)
    crop_cfg=preprocess_cfg.get('crop')
    if crop_cfg:
        if crop_cfg.get('type')=='all':
            for _,row in tqdm(main_df.iterrows(),total=len(main_df)):
                img_file,label,mask_file=row['image'],row['class'],row['mask']
                # Getting the image and mask files
                image_path=os.path.join(img_dir,img_file)
                mask_path=os.path.join(mask_dir,mask_file)
                img=cv2.imread(image_path,cv2.IMREAD_COLOR)
                mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
                extract_crops_all(img,crop_cfg.get('width'),crop_cfg.get('height'),prep_img_dir,img_file.split('.')[0],label,prep_labels_dir)  
                extract_crops_all(mask,crop_cfg.get('width'),crop_cfg.get('height'),prep_mask_dir,mask_file.split('.')[0])  

    else:
        pass
    
    prep_main_df=create_table(prep_labels_dir,prep_img_dir,prep_mask_dir)
    # Creating train, val and test dataframes without preprocessing
    train_df=prep_main_df.sample(frac=train_fraction,random_state=random_seed)
    if train_fraction < 1:
        val_df=pd.concat([main_df, train_df]).drop_duplicates(keep=False).sample(frac=validation_fraction/(1-train_fraction),random_state=random_seed)
        test_df=pd.concat([main_df, train_df,val_df]).drop_duplicates(keep=False)
    else :
        val_df=pd.DataFrame([],columns=train_df.columns)
        test_df=pd.DataFrame([],columns=train_df.columns)

    # Saving dataframes
    train_df.to_csv(preprocessed_directory+'/train.tsv',index=False)
    val_df.to_csv(preprocessed_directory+'/val.tsv',index=False)
    test_df.to_csv(preprocessed_directory+'/test.tsv',index=False)



