from matplotlib import pyplot as plt
from utils.transform import transform
from torchvision.transforms import ToPILImage
from PIL import Image
import wandb
from PIL import Image
def save_figure(with_wandb,image_path,mask,roi,ground_truth,predicted_class,fig_name,save_local):
    figure = plt.figure(figsize=(12, 12))
    image=Image.open(image_path)
    image=transform(image,img_size=(mask.shape[-2],mask.shape[-1]),with_norm=False)
    image=ToPILImage()(image)
    figure.add_subplot(1, 2, 1)
    plt.title(ground_truth)
    plt.axis("off")
    plt.imshow(image)
    plt.imshow(mask,cmap="inferno",alpha=0.4)
    figure.add_subplot(1, 2, 2)
    plt.title(predicted_class)
    plt.axis("off")
    plt.imshow(image)
    plt.imshow(roi,cmap="inferno",alpha=0.8)
    if save_local:
        plt.savefig(f'./reports/figures/{fig_name}',bbox_inches='tight')
    if with_wandb:
        fig_res=Image.open(f'./reports/figures/{fig_name}')
        wandb.log({fig_name:wandb.Image(fig_res)})
    plt.cla()
    plt.close(figure)