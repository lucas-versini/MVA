## Object recognition and computer vision 2024/2025
### Assignment 3: Sketch image classification
### Lucas Versini

See the [course website](https://gulvarol.github.io/teaching/recvis24/) and the [Kaggle challenge](https://www.kaggle.com/competitions/mva-recvis-2024).

#### Requirements
PyTorch is required. Additional dependencies can be found in `requirements.txt`.

```bash
pip install -r requirements.txt
```

#### Files
Here is a brief description of each file:
- `aggregate.py`: to generate a Kaggle submission using several models. The paths to the models have to be written at the beginning of the file.
- `data.py`: definition of all augmentation techniques.
- `evaluate.py`: to generate a Kaggle submission using a single model.
- `main.py`: to train a model. Arguments such as `model_name`, `batch-size`, `epoch`...
- `model.py`: definition of all models.
- `model_factory.py`: to get both the model and the augmentations.

#### Data

By default, the scripts expect the data to be organized in a folder named `data_sketches`, structured as follows (the name of the images does not have any importance):
```
data_sketches/
│
├── train_images/
│   ├── class_1/
│   │   ├── image1.jpeg
│   │   └── ...
│   ├── class_2/
│   │   ├── image1.jpeg
│   │   └── ...
│   └── ...
│
├── val_images/
│   ├── class_1/
│   │   ├── image1.jpeg
│   │   └── ...
│   ├── class_2/
│   │   ├── image1.jpeg
│   │   └── ...
│   └── ...
│
├── test_images/
│   ├── mistery_category/
│   │   ├── image1.jpeg
│   │   └── ...
```

#### Training and validating a model
Run the script `main.py` to train a model.

Two examples are given:
```bash
python main.py --model_name deit --epochs 20 --model_name_save model_deit --cutmix 1
python main.py --model_name eva02 --epochs 10 --model_name_save model_eva --optimizer sgd --batch-size 8
```

#### Generate the Kaggle submission
For a single model:
```bash
python evaluate.py --model_name eva02 --model experiment/model_eva.pth
```

For several models, first put the names of the models in `aggregate.py`, then simply use:
```bash
python aggregate.py
```
