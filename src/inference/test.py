import gc
import os
import time

import torch
import wandb
from src.visualization.save_figure import save_figure
from torch.nn.functional import interpolate
from tqdm import tqdm

def test(device,with_wandb,model, test_loader,jaccard,classes):
    torch.cuda.empty_cache()
    # Run the model on some test examples
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        IoU=0
        fig_ct=0
        batch_ct=0


        for batch_dict in tqdm(test_loader,total=len(test_loader)):
            images,masks, labels=batch_dict["image_tensor"],batch_dict["mask_tensor"],batch_dict['label']
            img_paths=batch_dict["img_path"]
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs,rois = model(images)
            ## Apply threshold to RoIs
            rois_threshold = rois > rois.mean(dim=(-2,-1),keepdim=True)
            ## Resize the masks to match RoIs Size
            masks_resized=interpolate(masks,size=rois.shape[-2:])
            masks_resized_bool=masks_resized > 0
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0)
            IoU+=jaccard(rois_threshold,masks_resized_bool)
            correct += (predicted == labels).sum().item()
            batch_ct+=1

            if fig_ct < 50 :
                for (output,roi,mask,label,img_path) in zip(outputs[:50],rois[:50],masks[:50],labels[:50],img_paths[:50]):

                        # get the image name from it's path
                        img_name= os.path.basename(img_path)
                        dim_ct=len(roi.shape)
                        if dim_ct < 4 :
                            roi=roi.unsqueeze(0)

                        ## Resize the RoI to match the mask size
                        roi_resized=interpolate(roi ,size=mask.shape[-2],mode='bilinear')

                        ## Squeeze the RoIs
                        roi_resized = roi_resized.squeeze()
                        ## Squeeze the mask to get one channel
                        mask=mask.squeeze()
                        ## Get predicted class for one image
                        predicted_label=torch.max(output.data, -1).indices
                        predicted_class=classes[predicted_label]
                        save_figure(with_wandb=with_wandb,image_path=img_path,mask=mask.cpu(),roi=roi_resized.cpu(),ground_truth=classes[label],predicted_class=predicted_class,fig_name=img_name,save_local=True)
                        fig_ct+=1
            del images
            del labels
            del masks
            gc.collect()
            torch.cuda.empty_cache()


        print(f"===> Running inference on the total of {total} " +
              f"test images:    Accuracy : {100 * correct / total:.3f}% || IoU : {IoU / batch_ct:.3f}\n")
        if with_wandb:
            wandb.log({"Test Accuracy": correct / total,"Test IoU":IoU / batch_ct})
        print(f"===> Picking {fig_ct} random test image(s) and saving corresponding inference results in ./reports/figures "
              )
        time.sleep(1)
