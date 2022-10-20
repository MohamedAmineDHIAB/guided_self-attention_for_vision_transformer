
from torch.nn.functional import interpolate
import torch
import gc
def train_batch(device,images,masks, labels, model, optimizer, loss_cl,loss_mask,lamda,jaccard):
    torch.cuda.empty_cache()
    model.train()
    images, labels = images.to(device), labels.to(device)
    # Forward pass ➡
    outputs,rois = model(images)
    ## Resize the masks to match RoIs Size
    masks_resized=interpolate(masks,size=rois.shape[-2:])
    masks_resized_bool=masks_resized > 0
    ## Pass the resized masks to <device>:
    masks_resized_bool = masks_resized_bool.to(device)
    masks_resized = masks_resized.to(device)

    loss_cl_value=loss_cl(outputs, labels)
    loss_mask_value=loss_mask(rois,masks_resized)
    loss = loss_cl_value+lamda*loss_mask_value
    rois_threshold = rois > rois.mean(dim=(-2,-1),keepdim=True)
    IoU=jaccard(rois_threshold,masks_resized_bool)
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()
    del images
    del labels
    del masks_resized
    del masks_resized_bool
    gc.collect()
    torch.cuda.empty_cache()

    return loss,loss_cl_value,loss_mask_value,IoU
