
from torch.nn.functional import interpolate
import torch
import wandb
import gc

def validate(device,validation_loader, model, loss_cl,loss_mask,lamda,jaccard,with_wandb,step=0):
    '''

    Validation function used to validate the model by running inference on the validation
    set and logging metric into Weights and Biases if with_wandb is set to True.

    '''
    # Run the model on some validation examples
    with torch.no_grad():
        batch_ct,correct, total,IoU,loss,loss_cl_value,loss_mask_value = 0,0, 0,0,0,0,0

        model.eval()
        for batch_dict in validation_loader:
            images,masks, labels=batch_dict["image_tensor"],batch_dict["mask_tensor"],batch_dict['label']
            images, labels = images.to(device), labels.to(device)
            # Running inference on one batch ➡
            outputs,rois = model(images)
            _, predicted = torch.max(outputs.data, -1)
            ## Resize the masks to match RoIs Size
            masks_resized=interpolate(masks,size=rois.shape[-2:])
            masks_resized_bool=masks_resized > 0
            ## Pass the resized masks to <device>:
            masks_resized_bool = masks_resized_bool.to(device)
            masks_resized = masks_resized.to(device)
            if len(outputs.shape)==1:
                outputs=outputs.unsqueeze(0)
            loss_cl_value+=loss_cl(outputs, labels)

            loss_mask_value+=loss_mask(rois,masks_resized)
            rois_threshold = rois > rois.mean(dim=(-2,-1),keepdim=True)
            IoU+=jaccard(rois_threshold,masks_resized_bool)
            correct += (predicted == labels).sum().item()
            total+=labels.size(0)
            batch_ct+=1
            del images
            del labels
            del masks_resized
            del masks_resized_bool
            gc.collect()
            torch.cuda.empty_cache()
        loss=(loss_cl_value+lamda*loss_mask_value)/batch_ct
        # Wandb logs for validation results
        if with_wandb:
            wandb.log({ "Total Loss (Validation)": loss,"Cross-entropy Loss (Validation)":loss_cl_value/batch_ct,"Dice Loss (Validation)":loss_mask_value/batch_ct,"Intersection over Union (Validation)":IoU/batch_ct, "Accuracy (Validation)":correct/total}, step=step)
        torch.cuda.empty_cache()
        return (loss,loss_cl_value/batch_ct,loss_mask_value/batch_ct,IoU/batch_ct,correct/total)
