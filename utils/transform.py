'''
transformations to apply on an image before feeding to the model

'''

from torchvision import transforms
def transform(img, img_size=None,patch_size=None,with_norm=True):
    if img_size is not None:
        img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    if with_norm:
        try :
            img=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        except :
            pass
    if patch_size is not None:
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h]
    return img