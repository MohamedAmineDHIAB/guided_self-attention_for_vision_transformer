'''
Guided soft attention model class

'''
from collections import OrderedDict

import torch
from torch import nn
from torchvision import models


class DenseNetGuSA(nn.Module):
    def __init__(self,num_classes,backbone_name='densenet169'):

        super(DenseNetGuSA,self).__init__()
        if backbone_name == 'densenet169':
            self.backbone=models.densenet169(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad= False

        #Reduction layer C4_Reduction (input=1280,ouput=512)
        self.c4_reduction=nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(1280)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(1280, 512,
                                          kernel_size=1, stride=1, bias=True)),
        ]))

        self.features_dict={}
        def get_activation(name):
            #get activation of the layer called name
            def hook(model, input, output):
                    self.features_dict[name] = output.detach()
            return(hook)

        #Attaching the hooks to the corresponding layers
        self.backbone.features.relu0.register_forward_hook(get_activation('relu0'))
        self.backbone.features.denseblock1.register_forward_hook(get_activation('denseblock1'))
        self.backbone.features.denseblock2.register_forward_hook(get_activation('denseblock2'))
        self.backbone.features.denseblock3.register_forward_hook(get_activation('denseblock3'))



        #Reduction layer C5_Reduction (input=1344,ouput=512)
        self.c5_reduction=nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(1344)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(1344, 512,
                                          kernel_size=1, stride=1, bias=True)),
        ]))

        #RoI_map Layer
        self.roi=nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(512, 1,
                                          kernel_size=1, stride=1, bias=True)),
            ('sigmoid',nn.Sigmoid())
        ]))
        self.c6=nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(512, num_classes,
                                          kernel_size=1, stride=1, bias=True))
        ]))
    def forward(self,x):
        self.backbone.eval()
        self.backbone(x)
        c1_output=self.features_dict['relu0']
        c2_db1_output=self.features_dict['denseblock1']
        c3_db2_output=self.features_dict['denseblock2']
        c4_db3_output=self.features_dict['denseblock3']
        c4_reduction_output=self.c4_reduction(c4_db3_output)
        avg_pool_c1_output = nn.AvgPool2d(kernel_size=(2,2),stride=8)(c1_output)
        avg_pool_c2_db1_output = nn.AvgPool2d(kernel_size=(2,2),stride=4)(c2_db1_output)
        avg_pool_c3_db2_output = nn.AvgPool2d(kernel_size=(2,2),stride=2)(c3_db2_output)
        concatenation_output=torch.cat((avg_pool_c1_output,avg_pool_c2_db1_output,avg_pool_c3_db2_output,c4_reduction_output),dim=1)
        c5_reduction_ouput=self.c5_reduction(concatenation_output)
        roi_map = self.roi(c5_reduction_ouput)
        c6_output=self.c6(c5_reduction_ouput*roi_map)
        c6_output_avg=nn.AvgPool2d(kernel_size=c6_output.shape[-2:])(c6_output)
        final_output=c6_output_avg.squeeze()
        return(final_output,roi_map)
