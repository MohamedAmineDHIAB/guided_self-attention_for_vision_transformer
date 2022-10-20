'''
Guided soft attention model class

'''
from collections import OrderedDict

import torch
from torch import nn
from src.models.vision_transformer_dino import VitGenerator, Block

class ViTGuSA(nn.Module):
    def __init__(self,num_classes,device="cpu",backbone_name='vit_base',patch_size=8,freeze_backbone=True,pretraining_method="dino",head_depth=0):

        super(ViTGuSA,self).__init__()
        self.patch_size = patch_size
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.head_depth=head_depth
        # Backbone Initialization
        if pretraining_method=="dino":
            self.backbone=VitGenerator(backbone_name, patch_size,device, evaluate=freeze_backbone, random=False)
        if head_depth > 0 :
            # adding new trainable transformer head blocks to the model
            self.transformer_head_blocks =nn.ModuleList([
            Block()
            for i in range(head_depth)])
        #classification Head
        self.cls_head=nn.Linear(self.backbone.model.num_features, num_classes)
    def forward(self,x):
        backbone_features=self.backbone(x)
        # if the head_depth is non null then the model is composed of backbone + transformer heads
        if self.head_depth > 0 :
            last_features=backbone_features
            for i, blk in enumerate(self.transformer_head_blocks):
                if i < self.head_depth - 1:
                    last_features = blk(last_features)
                else:
                    # return attention of the last block
                    attentions,last_features= blk(last_features, return_attention=True),blk(last_features)
            cls_head_output=self.cls_head(last_features[:,0])
        # if the backbone is not frozen then simply fine tune the backbone without additional transformers in the head
        else :
            cls_head_output=self.cls_head(backbone_features[:,0])
            attentions=self.backbone.get_last_selfattention(x)
        final_output=cls_head_output.squeeze()

        batch_size=attentions.shape[0]
        # number of attention heads
        nh = attentions.shape[1]
        # width and height of the attention maps
        w_featmap = x.shape[-2] // self.patch_size
        h_featmap = x.shape[-1] // self.patch_size
        # keep only the output patch attention (first row of the attention matrix except first element )
        cls_attentions = attentions[:, :, 0, 1:].reshape(batch_size,nh,w_featmap,h_featmap)
        roi_map=torch.mean(cls_attentions,dim=1,keepdim=True)
        return(final_output,roi_map)
