from torch import nn


class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        IoU = (intersection)/(inputs.sum() + targets.sum() + smooth - intersection)

        return IoU