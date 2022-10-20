"""
Train pipe implementation
Train a guided soft attention model on a custom dataset
"""
from src.train.train_batch import train_batch
import wandb
from tqdm import tqdm
import torch
from src.inference.validate import validate
def train(device,with_wandb,model, train_loader,validation_loader, loss_cl,loss_mask, optimizer, config,jaccard,scheduler):

    # Run training and track with wandb
    total_batches = len(train_loader) * config['epochs']
    example_ct = 0  # number of examples seen
    batch_ct = 0 # number of batches processed
    early_stop = 0 # number of epochs where validation loss stopped decreasing
    lamda=config['lamda']
    epochs=config['epochs']
    best_val_loss,val_loss_cl_value,val_loss_mask_value,val_IoU,val_correct=validate(device,validation_loader, model, loss_cl,loss_mask,lamda,jaccard,with_wandb)
    print(f"Validation results before training : Total Loss :{best_val_loss:.3f}, Dice Loss :{val_loss_mask_value:.3f}, Cross-entropy Loss :{val_loss_cl_value:.3f}, IoU:{val_IoU:.3f}, Accuracy:{val_correct:.3f}\n")
    # Save the model in pth format
    torch.save(model.state_dict(), config['save_path'])
    for epoch in range(epochs):
        if early_stop < 5 :
            for _, batch_dict in enumerate(tqdm(train_loader,leave=True,desc=f'Epoch {epoch+1}/{epochs}  ')):
                images,masks, labels=batch_dict["image_tensor"],batch_dict["mask_tensor"],batch_dict['label']

                loss,loss_cl_value,loss_mask_value,IoU = train_batch(device,images,masks, labels, model, optimizer, loss_cl,loss_mask,lamda,jaccard)

                example_ct +=  len(images)
                batch_ct += 1
                # Report metrics every 5th batch
                if (batch_ct  % 5) == 0 :
                    # Wandb logs for training results
                    if with_wandb:
                        wandb.log({"Epochs": epoch+1, "Total Loss (Train)": loss,"Cross-entropy Loss (Train)":loss_cl_value,"Dice Loss (Train)":loss_mask_value,"Intersection over Union (Train)":IoU, "Number of Batches":batch_ct}, step=example_ct)
            # getting validation results after each epoch
            val_loss,val_loss_cl_value,val_loss_mask_value,val_IoU,val_correct=validate(device,validation_loader, model, loss_cl,loss_mask,lamda,jaccard,with_wandb,example_ct)

            # Check if the validation loss is decreasing else increment early stopping
            if val_loss < best_val_loss:
                # store the validation loss in best_val_loss
                best_val_loss = val_loss
                # Save the model in pth format
                torch.save(model.state_dict(), config['save_path'])
                if with_wandb:
                    wandb.save(config['save_path'])
            else :
                early_stop+=1

            # logging traning and validation results
            print(f'\n===> Epoch {epoch+1}/{epochs}  Train results:  Dice Loss : {loss_mask_value:.3f} || Cross-entropy Loss : {loss_cl_value:.3f} || IoU : {IoU:.3f}')
            print(f'\n===> Epoch {epoch+1}/{epochs}  Validation results:  Dice Loss : {val_loss_mask_value:.3f} || Cross-entropy Loss : {val_loss_cl_value:.3f} || IoU : {val_IoU:.3f}\n')
            scheduler.step()
        else :
            print("!!! Validation loss is no longer decreasing, early stopping will be triggered !!!\n")
            # break the training loop
            break


    return()
