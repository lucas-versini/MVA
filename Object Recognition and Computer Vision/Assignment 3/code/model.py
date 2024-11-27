import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from timm.models import create_model
from transformers import ViTForImageClassification
from transformers import AutoImageProcessor, ViTForImageClassification

use_cuda = torch.cuda.is_available()

nclasses = 500

##############################################
#       See data.py for the transforms       #
# See model_factory.py for model + transform #
##############################################

# EVA-02
# https://arxiv.org/abs/2303.11331
class eva02(nn.Module):
    def __init__(self):
        super(eva02, self).__init__()
        self.model = create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained = True)
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        return self.model(x)


# DeiT
# https://arxiv.org/abs/2012.12877
class Deit(nn.Module):
    def __init__(self):
        super(Deit, self).__init__()
        self.model = create_model('deit_base_distilled_patch16_384', pretrained = True)
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, num_ftrs)
        self.model.head_dist = nn.Linear(num_ftrs, num_ftrs)
        self.fc = nn.Linear(num_ftrs, nclasses)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.model(x)
        # x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_features(self, x):
        return self.model(x)


class DeitBis(nn.Module):
    def __init__(self):
        super(DeitBis, self).__init__()
        self.model = create_model('deit3_base_patch16_384', pretrained = True)
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, nclasses)
        self.model.head_dist = nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        return self.model(x)


# VIT
# https://arxiv.org/abs/2010.11929
class vit_b_16(nn.Module):
    def __init__(self):
        super(vit_b_16, self).__init__()
        self.model = models.vit_b_16(weights = "DEFAULT")
        num_ftrs = self.model.heads[0].in_features
        self.model.heads = nn.Sequential(nn.Linear(num_ftrs, nclasses))

    def forward(self, x):
        return self.model(x)

class vit_patch_16(nn.Module):
    def __init__(self):
        super(vit_patch_16, self).__init__()
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.classifier = torch.nn.Linear(768, 250)
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, x):
        inputs = self.image_processor(x, return_tensors = "pt", do_rescale = False)
        if use_cuda:
            inputs['pixel_values'] = inputs['pixel_values'].cuda()
        return self.model(**inputs).logits


# ResNext101
# https://arxiv.org/abs/1611.05431
class Resnext101(nn.Module):
    def __init__(self):
        super(Resnext101, self).__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        return self.model(x)


# ResNet34
# https://arxiv.org/abs/1512.03385
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights = "DEFAULT")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, nclasses)

    def forward(self, x):
        return self.model(x)


# Basic network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size = 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)