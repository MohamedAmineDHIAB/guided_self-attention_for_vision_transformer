'''
Guided soft attention model class

'''
import math
from collections import OrderedDict

import torch
from src.models.vision_transformer_dino import Block, VitGenerator
from torch import nn


class ViT_CNN_GuSA(nn.Module):
    def __init__(self,num_classes,device="cpu",backbone_name='vit_base',patch_size=8,freeze_backbone=True,pretraining_method="dino",n_features=1):

        super(ViT_CNN_GuSA,self).__init__()
        self.patch_size = patch_size
        self.backbone_name = backbone_name
        self.n_features=n_features
        # Backbone Initialization
        if pretraining_method=="dino":
            self.backbone=VitGenerator(backbone_name, patch_size,device, evaluate=freeze_backbone, random=False)
        #Reduction layer  (input=n_features*768 (embedding size in vit_base),ouput=512)
        self.reduction=nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(n_features*768)),
            ('conv', nn.Conv2d(n_features*768, 512,
                                          kernel_size=1, stride=1, bias=True)),
        ]))

        #RoI_map Layer
        self.roi=nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(512, 1,
                                          kernel_size=1, stride=1, bias=True)),
            ('sigmoid',nn.Sigmoid())
        ]))
        # classification layer
        self.to_classifier=nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(512, num_classes,
                                          kernel_size=1, stride=1, bias=True))
        ]))

    def forward(self,x):
        # get hidden embeddings from the last `n_features` blocks of the backbone
        intermediate_features=self.backbone.model.get_intermediate_layers(x,n=self.n_features)
        intermediate_features=torch.cat(intermediate_features,dim=-1)[:,1:]
        # reshape concatenated features to fit reduction input
        batch_size=intermediate_features.shape[0]
        w=math.isqrt(intermediate_features.shape[1])
        h=math.isqrt(intermediate_features.shape[1])
        n_filters=intermediate_features.shape[-1]
        intermediate_features=intermediate_features.reshape(batch_size,n_filters,w,h)
        # pass them through first cnn layer
        reduced_features=self.reduction(intermediate_features)
        # pass reduced features through soft attention layer
        roi_map=self.roi(reduced_features)
        # multiply the roi by the reduced features and pass
        # the result into the classification layer
        classifier_input=self.to_classifier(reduced_features*roi_map)
        # average over width and height and apply softmax for normalization
        classifier_output=nn.AvgPool2d(kernel_size=classifier_input.shape[-2:])(classifier_input)

        final_output=classifier_output.squeeze()
        return(final_output,roi_map)
