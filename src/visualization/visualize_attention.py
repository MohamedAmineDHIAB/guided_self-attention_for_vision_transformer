from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from PIL import Image



def visualize_attention(rois,classes,image,mask=None,figsize=(8,8),show_original=False,truth=None):
    # Set figure size
    figure = plt.figure(figsize=figsize)
    size_figure=len(rois)+1
    for i in range(size_figure-1):
        roi=rois[i]
        # Resize the RoI to match the mask size
        roi_resized=interpolate(roi ,size=image.size,mode='bilinear')
        ax1=figure.add_subplot(1, size_figure, i+1)
        ax1.imshow(image)
        ax1.imshow(roi_resized.squeeze(),cmap="inferno",alpha=0.5)
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax1.axes.set_title(classes[i])
    if show_original and truth:
        ax2=figure.add_subplot(1, size_figure, size_figure)
        ax2.imshow(image)
        if mask:
            ax2.imshow(mask,cmap="inferno",alpha=0.5)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax2.axes.set_title(truth)
