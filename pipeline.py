"""
Run a pipline with pre-defined configuration.

Example of usage :

python3 pipeline.py --cfg './cfg.yml'

"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ast
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml
from torch import load, nn, optim

import wandb
from src.data.augment import augment
from src.data.create_table import create_table
from src.data.dataset import GuSADataset, make_loader
from src.inference.test import test
from src.models.densenet_gusa import DenseNetGuSA
from src.models.vit_gusa import ViTGuSA
from src.models.vit_cnn_gusa import ViT_CNN_GuSA
from src.data.preprocess import preprocess
from src.train.train import train
from utils.dice_loss import DiceLoss
from utils.iou import IoU

# Globals

# YAML configuration file used to build the pipeline
global CONFIG_FILE


# data directories
global IMG_DIR
global MASK_DIR
global LABEL_DIR
# device used for training and inference , gpu or cpu
global DEVICE
# Run Name used in wandb initialization
global NAME
# Job Type used in wandb initialization
global JOB_TYPE
# pipeline is a list of processes to be executed each process is defined by a dict
global PIPELINE
# length of our pipeline to be executed
global PIPELINE_LEN
# random seed defined for reproducible results
global RANDOM_SEED
# boolean variable to use wandb for watching a run and saving logs
global WITH_WANDB
# use or not multiple gpus
global MULTI_GPUS
# model configuration dictionary
global MODEL_CFG



def parse_arguments():
    global CONFIG_FILE
    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('--cfg', help='path to pipeline YAML configuration file',
                        required=True)
    args = parser.parse_args()
    CONFIG_FILE=args.cfg

def process_pipe(idx,train_fraction,validation_fraction,is_augment,preprocess_cfg):
    # taking the main directory containing the images, example: /data/raw
    DATA_DIR=os.path.dirname(IMG_DIR)
    # Making the data
    print(f'\n{idx+1}/{PIPELINE_LEN}  preparing the dataset ...\n')
    main_df=create_table( LABEL_DIR,IMG_DIR,MASK_DIR)
    print(f'Dataset is of length : {len(main_df)}\n')

    IMG_TO_AUGMENT_DIR=IMG_DIR
    MASK_TO_AUGMENT_DIR=MASK_DIR
    if preprocess_cfg:
        preprocess(preprocess_cfg,IMG_DIR,MASK_DIR,main_df,train_fraction,validation_fraction,RANDOM_SEED)
        # defining the directory for the preprocessed images to be augmented 
        IMG_TO_AUGMENT_DIR=os.path.join(DATA_DIR,'../preprocessed/image')
        # normalizing the directory path
        IMG_TO_AUGMENT_DIR=os.path.normpath(IMG_TO_AUGMENT_DIR)
        # defining the directory for the preprocessed masks to be augmented 
        MASK_TO_AUGMENT_DIR=os.path.join(DATA_DIR,'../preprocessed/image_mask')
        # normalizing the directory path
        MASK_TO_AUGMENT_DIR=os.path.normpath(MASK_TO_AUGMENT_DIR)
        print(preprocess_cfg)
    else :
        # Creating train, val and test dataframes without preprocessing
        train_df=main_df.sample(frac=train_fraction,random_state=RANDOM_SEED)
        if train_fraction < 1:
            val_df=pd.concat([main_df, train_df]).drop_duplicates(keep=False).sample(frac=validation_fraction/(1-train_fraction),random_state=RANDOM_SEED)
            test_df=pd.concat([main_df, train_df,val_df]).drop_duplicates(keep=False)
        else :
            val_df=pd.DataFrame([],columns=train_df.columns)
            test_df=pd.DataFrame([],columns=train_df.columns)

        # Saving dataframes
        train_df.to_csv(DATA_DIR+'/train.tsv',index=False)
        val_df.to_csv(DATA_DIR+'/val.tsv',index=False)
        test_df.to_csv(DATA_DIR+'/test.tsv',index=False)
    exit()

    if is_augment :
        augment(IMG_TO_AUGMENT_DIR,
                MASK_TO_AUGMENT_DIR,
                os.path.dirname(IMG_TO_AUGMENT_DIR)+'/train.tsv')

        train_df_aug=create_table( os.path.join(DATA_DIR, '../augmented/labels'),
                     os.path.join(DATA_DIR, '../augmented/image'),
                     os.path.join(DATA_DIR, '../augmented/image_mask'),table_name='train.tsv')
    else : 
        train_df_aug=train_df


    print(f'\nTrain set is of length : {len(train_df_aug)}\n')
    print(f'Validation set is of length : {len(val_df)}\n')
    print(f'Test set is of length : {len(test_df)}\n')
    return()

def train_pipe(idx,train_cfg):
    # image size to be used for training
    img_size=ast.literal_eval(train_cfg['img_size'])
    # check if augmentation is available
    with_aug = train_cfg['with_augmentation']
    # Creating train and validation GuSADataset objects using our data
    ## Making train dataset based on "with_augmentation" value
    original_data_dir=os.path.dirname(IMG_DIR)
    if with_aug:
        aug_dir=os.path.join(original_data_dir,'augmented')
        img_dir=os.path.join(aug_dir,'image')
        mask_dir=os.path.join(aug_dir,'image_mask')
        train_data=os.path.join(aug_dir,'train.tsv')
    else :
        img_dir=IMG_DIR
        mask_dir=MASK_DIR
        train_data=os.path.join(original_data_dir,'train.tsv')
    train_dataset=GuSADataset(annotations_file=train_data,img_dir=img_dir,mask_dir=mask_dir,img_size=img_size,patch_size=None if MODEL_CFG['backbone_name'] != 'vit_base' else MODEL_CFG['patch_size'])
    # Making validation dataset
    validation_data=os.path.join(original_data_dir,'val.tsv')
    validation_dataset=GuSADataset(annotations_file=validation_data,img_dir=IMG_DIR,mask_dir=MASK_DIR,img_size=img_size,patch_size=None if MODEL_CFG['backbone_name'] != 'vit_base' else MODEL_CFG['patch_size'])
    # Get list of classes
    classes_df=pd.read_csv('./classes.tsv',sep=',')
    classes=classes_df['class'].to_numpy()
    # Get the total number of classes
    num_classes=len(classes)
    # Make the model
    if MODEL_CFG['backbone_name'] == 'densenet169':
        model = DenseNetGuSA(num_classes=num_classes)
    elif MODEL_CFG['backbone_name'] == 'vit_base':
        if 'n_features' in MODEL_CFG:
            model=ViT_CNN_GuSA(num_classes=num_classes,device=DEVICE,backbone_name=MODEL_CFG['backbone_name'],patch_size=MODEL_CFG['patch_size'],freeze_backbone=MODEL_CFG['freeze_backbone'],pretraining_method=MODEL_CFG['pretraining_method'],n_features=MODEL_CFG['n_features'])
        else:
            model=ViTGuSA(num_classes=num_classes,device=DEVICE,backbone_name=MODEL_CFG['backbone_name'],patch_size=MODEL_CFG['patch_size'],freeze_backbone=MODEL_CFG['freeze_backbone'],pretraining_method=MODEL_CFG['pretraining_method'],head_depth=MODEL_CFG['head_depth'])
        if MULTI_GPUS:
            model=nn.DataParallel(model)
    model.to(DEVICE)
    # log train set length
    print(f'\n** Training Set is of length : {len(pd.read_csv(train_data))} **\n')
    # log validation set length
    print(f'\n** Validation Set is of length : {len(pd.read_csv(validation_data))} **\n')

    # Starting the train pipe
    print(f'\n{idx+1}/{PIPELINE_LEN}  training the model ...\n')
    # Make the train loader
    train_loader = make_loader(train_dataset, batch_size=train_cfg['batch_size'])
    # Make the validation loader
    validation_loader = make_loader(validation_dataset, batch_size=train_cfg['batch_size'])
    # Making the IoU metric used to assess the segmentation branch
    jaccard = IoU().to(DEVICE)
    # Making the Cross-Entropy Loss and the Dice Loss used in the backward pass
    loss_cl=nn.CrossEntropyLoss().to(DEVICE)
    loss_mask=DiceLoss().to(DEVICE)
    # Making the optimizer
    if train_cfg['optimizer']=='adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg['learning_rate'])
    # Making scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=train_cfg['lr_decay'])
    train(DEVICE,WITH_WANDB,model, train_loader, validation_loader,loss_cl,loss_mask, optimizer, train_cfg,jaccard,scheduler)

def test_pipe(idx,test_cfg,test_data_path):
    # image size to be used for testing
    img_size=ast.literal_eval(test_cfg['img_size'])
    # Get list of classes
    classes_df=pd.read_csv('./classes.tsv',sep=',')
    classes=classes_df['class'].to_numpy()
    # Get the total number of classes
    num_classes=len(classes)
    # Making test dataset
    test_dataset=GuSADataset(annotations_file=test_data_path,img_dir=IMG_DIR,mask_dir=MASK_DIR,img_size=img_size,patch_size=None if MODEL_CFG['backbone_name'] != 'vit_base' else MODEL_CFG['patch_size'])
    # Make the model
    if MODEL_CFG['backbone_name'] == 'densenet169':
        model = DenseNetGuSA(num_classes=num_classes)
    elif MODEL_CFG['backbone_name'] == 'vit_base':
        if 'n_features' in MODEL_CFG:
            model=ViT_CNN_GuSA(num_classes=num_classes,device=DEVICE,backbone_name=MODEL_CFG['backbone_name'],patch_size=MODEL_CFG['patch_size'],freeze_backbone=MODEL_CFG['freeze_backbone'],pretraining_method=MODEL_CFG['pretraining_method'],n_features=MODEL_CFG['n_features'])
        else:
            model=ViTGuSA(num_classes=num_classes,device=DEVICE,backbone_name=MODEL_CFG['backbone_name'],patch_size=MODEL_CFG['patch_size'],freeze_backbone=MODEL_CFG['freeze_backbone'],pretraining_method=MODEL_CFG['pretraining_method'],head_depth=MODEL_CFG['head_depth'])
        if MULTI_GPUS:
            model=nn.DataParallel(model)
    model.to(DEVICE)
    # Starting the test pipe
    print(f'\n{idx+1}/{PIPELINE_LEN}  testing the model ...\n')
    # log test set length
    print(f'\n** Test Set is of length : {len(pd.read_csv(test_data_path))} **\n')
    # Making the IoU Jacard metric used to assess the segmentation branch
    jaccard = IoU().to(DEVICE)
    # Importing model weights
    model.load_state_dict(load(test_cfg['weights'],map_location=DEVICE))
    # Make the test loader
    test_loader = make_loader(test_dataset, batch_size=test_cfg['batch_size'])
    test(DEVICE,WITH_WANDB,model, test_loader,jaccard,classes)
    if WITH_WANDB:
        wandb.save(test_cfg['weights'])
    return()

def browse_pipeline():

    # Execute pipes
    for i,pipe in enumerate(PIPELINE):
                if pipe['name']=='process':
                    train_fraction=pipe['train_fraction']
                    validation_fraction=pipe['validation_fraction']
                    preprocess_cfg=pipe.get('preprocess')
                    is_augment=pipe['is_augment']
                    process_pipe(i,train_fraction,validation_fraction,is_augment,preprocess_cfg)
                elif pipe['name']=='train':

                    train_pipe(idx=i,train_cfg=pipe['train_cfg'])
                elif pipe['name']=='test':
                    test_pipe(idx=i,test_cfg=pipe['test_cfg'],test_data_path=pipe['test_data'])

    return()

def pipeline(cfg_file=None):
    global IMG_DIR
    global MASK_DIR
    global LABEL_DIR
    global DEVICE
    global NAME
    global PIPELINE
    global PIPELINE_LEN
    global RANDOM_SEED
    global WITH_WANDB
    global MULTI_GPUS
    global MODEL_CFG

    with open(cfg_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_cfg_path=os.path.join('models',config['model_cfg'])
    # Extracting model configuration path
    with open(model_cfg_path) as f:
        MODEL_CFG = yaml.load(f, Loader=yaml.FullLoader)
    # Extracting variables from YAML configuration file
    IMG_DIR=config['img_dir']
    MASK_DIR=config['mask_dir']
    LABEL_DIR=config['label_dir']
    DEVICE=config['device']
    PIPELINE=config['pipeline']
    NAME=config['run_name']
    JOB_TYPE=config['job_type']
    PIPELINE_LEN=len(PIPELINE)
    RANDOM_SEED=config['random_seed']
    WITH_WANDB = config['with_wandb']
    MULTI_GPUS=config['multi_gpus']
    # Check if using wandb is set to true
    if WITH_WANDB :
        wandb_config=None
        for pipe in PIPELINE:
            if "train_cfg" in pipe:
                wandb_config=pipe['train_cfg']
        print(f"WANDB CONFIG:\nwandb_config")
        # Initializing wandb
        with wandb.init(project="guided-soft-attention", entity="empaia",name=NAME,job_type=JOB_TYPE,config=wandb_config):
            browse_pipeline()
    else :
        browse_pipeline()



if __name__ == '__main__':
    parse_arguments()
    pipeline(CONFIG_FILE)

