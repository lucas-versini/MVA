""" File containing the ModelFactory class, which is used to create the model and the data transformations """

from model import Net, ResNet34, vit_b_16, Deit, DeitBis, Resnext101, vit_patch_16, eva02

from data import data_transforms_train, data_transforms_test
from data import list_data_transforms_vit_train, data_transforms_vit_test
from data import data_transforms_vit_patch_train, data_transforms_vit_patch_test
from data import data_transforms_deit_test, list_data_transforms_deit_train, data_transforms_deit_train_augmix
from data import data_transforms_eva02_test, data_transforms_eva02_train_augmix, list_data_transforms_eva02_train

class ModelFactory:
    def __init__(self, model_name: str, data_transf: int):
        self.model_name = model_name
        self.data_transf = data_transf
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet34":
            return ResNet34()
        elif self.model_name == "vit_b_16":
            return vit_b_16()
        elif self.model_name == "deit":
            return Deit()
        elif self.model_name == "deit_bis":
            return DeitBis()
        elif self.model_name == "resnext101":
            return Resnext101()
        elif self.model_name == "vit_patch_16":
            return vit_patch_16()
        elif self.model_name == "eva02":
            return eva02()
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

    def init_transform(self):
        if self.model_name in ["basic_cnn", "resnet34"]:
            return data_transforms_train, data_transforms_train
        elif self.model_name == "vit_b_16":
            index = self.data_transf
            return list_data_transforms_vit_train[index], data_transforms_vit_test
        elif self.model_name == "vit_b_16_modified":
            return list_data_transforms_vit_train[1], data_transforms_vit_test
        elif self.model_name == "deit":
            if self.data_transf == -1: # AugMix
                return data_transforms_deit_train_augmix, data_transforms_deit_test
            else:
                return list_data_transforms_deit_train[self.data_transf], data_transforms_deit_test
        elif self.model_name == "deit_bis":
            return list_data_transforms_deit_train[self.data_transf], data_transforms_deit_test
        elif self.model_name == "resnext101":
            return data_transforms_train, data_transforms_test
        elif self.model_name == "vit_patch_16":
            return data_transforms_vit_patch_train, data_transforms_vit_patch_test
        elif self.model_name == "eva02":
            if self.data_transf == -1: # AugMix
                return data_transforms_eva02_train_augmix, data_transforms_eva02_test
            else:
                return list_data_transforms_eva02_train[self.data_transf], data_transforms_eva02_test
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
