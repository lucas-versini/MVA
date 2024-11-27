""" File containing data augmentation functions for different models """

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as transforms

#################################################
# See model.py and model_factory.py for details #
#################################################

# Remark: this file contains data augmentation functions for different models
# but also transformations compatible for all models (AugMix, Mixup, CutMix)


##############################################
#   General data augmentation functions      #
##############################################

# AugMix
# https://arxiv.org/abs/1912.02781
class AugMix:
    def __init__(self, augmentation_list, severity=3, width=3, depth=-1, alpha=1.0):
        self.augmentation_list = augmentation_list
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def __call__(self, img):
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = np.zeros_like(np.array(img)).astype(np.float32)
        for i in range(self.width):
            image_aug = img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentation_list)
                image_aug = op(image_aug, self.severity)
            mix += ws[i] * np.array(image_aug).astype(np.float32)

        mixed = (1 - m) * np.array(img) + m * mix
        return Image.fromarray(mixed.astype(np.uint8))

def augment_sharpness(img, severity):
    factors = np.linspace(0.1, 1.9, 10)
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factors[severity])

def augment_contrast(img, severity):
    factors = np.linspace(0.1, 1.9, 10)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factors[severity])

def augment_brightness(img, severity):
    factors = np.linspace(0.1, 1.9, 10)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factors[severity])

def augment_color(img, severity):
    factors = np.linspace(0.1, 1.9, 10)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factors[severity])

def augment_solarize(img, severity):
    threshold = int(np.linspace(256, 0, 10)[severity])
    return ImageOps.solarize(img, threshold)

augmentation_list = [augment_sharpness,
                     augment_contrast,
                     augment_brightness,
                     augment_color,
                     augment_solarize]

augmix_transform = AugMix(augmentation_list = augmentation_list)


# Mixup
# https://arxiv.org/abs/1710.09412

def mixup_data(x, y, alpha = 1.0, use_cuda = True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# CutMix
# https://arxiv.org/abs/1905.04899

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return mixed_x, y, y[index], lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


##############################################
# Model-specific data augmentation functions #
##############################################

# DeiT
data_transforms_deit_train = transforms.Compose([transforms.RandomResizedCrop(size = (384,384), scale = (0.9,1.0)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                      std = [0.229, 0.224, 0.225])
])

list_data_transforms_deit_train = [
    transforms.Compose([transforms.Resize((384, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.RandomResizedCrop(size = (384, 384), scale = (0.7, 1.0)),
                        transforms.RandomHorizontalFlip(p = 0.5),
                        transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.Resize((384, 384)),
                        transforms.RandomHorizontalFlip(p = 0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.Resize((384, 384)),
                        transforms.RandomPerspective(distortion_scale = 0.5, p = 0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ])
]

data_transforms_deit_train_augmix = transforms.Compose([transforms.Resize((384, 384)),
                                                        transforms.RandomHorizontalFlip(),
                                                        augmix_transform,
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                             std = [0.229, 0.224, 0.225])
])

data_transforms_deit_test = transforms.Compose([transforms.Resize((384, 384)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                     std = [0.229, 0.224, 0.225])
])


# ViT
list_data_transforms_vit_train = [
    transforms.Compose([transforms.RandomResizedCrop(size = (224, 224), scale = (0.7, 1.0)),
                        transforms.RandomVerticalFlip(p = 0.5),
                        transforms.RandomHorizontalFlip(p = 0.5),
                        transforms.RandomPerspective(distortion_scale = 0.5, p = 0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.Resize((224, 224)),
                        transforms.RandomVerticalFlip(p = 0.5),
                        transforms.RandomHorizontalFlip(p = 0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.Resize((224, 224)),
                        transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ])
]

data_transforms_vit_train_augmix = transforms.Compose([transforms.Resize((224, 224)),
                                                       transforms.RandomHorizontalFlip(),
                                                       augmix_transform,
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                            std = [0.229, 0.224, 0.225])
])

data_transforms_vit_test = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                    std = [0.229, 0.224, 0.225])
])


data_transforms_vit_patch_train = transforms.Compose([transforms.Resize((224, 224)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                           std = [0.229, 0.224, 0.225])
])
data_transforms_vit_patch_test = data_transforms_vit_patch_train


# Basic network
data_transforms_train = transforms.Compose([transforms.Resize((64, 64)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                 std = [0.229, 0.224, 0.225])
                                            ])
data_transforms_test = data_transforms_train

# EVA-02
list_data_transforms_eva02_train = [
    transforms.Compose([transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.RandomResizedCrop(size = (448, 448), scale = (0.7, 1.0)),
                        transforms.RandomHorizontalFlip(p = 0.5),
                        transforms.RandomPerspective(distortion_scale = 0.5, p = 0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.Resize((448, 448)),
                        transforms.RandomHorizontalFlip(p = 0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([transforms.Resize((448, 448)),
                        transforms.RandomPerspective(distortion_scale = 0.5, p = 0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
    ])
]

data_transforms_eva02_train_augmix = transforms.Compose([transforms.Resize((448, 448)),
                                                         transforms.RandomHorizontalFlip(),
                                                         augmix_transform,
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                              std = [0.229, 0.224, 0.225])
])

data_transforms_eva02_test = transforms.Compose([transforms.Resize((448, 448)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                      std = [0.229, 0.224, 0.225])
])