import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets
from tqdm import tqdm

from model_factory import ModelFactory

from PIL import Image

# Pretrained models to use
list_names_models = []
list_names_models.append(("deit", "/experiment/model_deit.pth"))
list_names_models.append(("eva02", "/experiment/model_eva.pth"))

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument("--data",
                        type = str,
                        default = "data_sketches",
                        metavar = "D",
                        help = "folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument("--seed", type=int, default = 1, metavar = "S", help = "random seed (default: 1)")
    parser.add_argument("--experiment",
                        type = str,
                        default = "experiment",
                        metavar = "E",
                        help = "folder where experiment outputs (models) are located.")
    parser.add_argument("--num_workers",
                        type = int,
                        default = 10,
                        metavar = "NW",
                        help = "number of workers for data loading")
    parser.add_argument("--outfile",
                        type = str,
                        default="experiment/kaggle.csv",
                        metavar = "D",
                        help = "name of the output csv file")
    parser.add_argument("--combine",
                        type = str,
                        default = "avg",
                        metavar = "CMB",
                        help = "how to combine the models (avg, max, majority)")
    args = parser.parse_args()
    return args

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

def test(list_names_models, use_cuda, args):
    outputs = []
    test_dir = args.data + "/test_images/mistery_category"
    print(f"Test dir: {test_dir}")
    with torch.no_grad():
        for i, (name, path) in enumerate(list_names_models):
            model, (_, transf) = ModelFactory(name, 0).get_all()
            model.load_state_dict(torch.load(path))
            model.eval()
            if use_cuda:
                model.cuda()
            
            for j, f in tqdm(enumerate(os.listdir(test_dir)), total=len(os.listdir(test_dir))):
                if "jpeg" in f:
                    data = transf(pil_loader(test_dir + "/" + f))
                    data = data.view(1, data.size(0), data.size(1), data.size(2))
                    if use_cuda:
                        data = data.cuda()
                    output = model(data)

                    if args.combine == "avg":
                        if i == 0:
                            outputs.append(nn.functional.softmax(output, dim=1))
                        else:
                            outputs[j] += nn.functional.softmax(output, dim=1)
                    elif args.combine == "max":
                        if i == 0:
                            outputs.append(output)
                        else:
                            outputs[j] = torch.max(outputs[j], output)
                    elif args.combine == "majority":
                        proba, predicted_class = output.data.max(1, keepdim = True)
                        if i == 0:
                            outputs.append([(predicted_class.item(), proba.item())])
                        else:
                            outputs[j].append((predicted_class.item(), proba.item()))
            del model
            torch.cuda.empty_cache()
        
        with open(args.outfile, "w") as output_file:
            output_file.write("Id,Label\n")
            for j, f in tqdm(enumerate(os.listdir(test_dir))):
                if "jpeg" in f:
                    if args.combine == "majority":
                        # Get the majority vote
                        votes = {}
                        for i in range(len(list_names_models)):
                            predicted_class, proba = outputs[j][i]
                            if predicted_class in votes:
                                votes[predicted_class][0] += 1
                                votes[predicted_class][1] = max(votes[predicted_class][1], proba)
                            else:
                                votes[predicted_class] = [1, proba]
                        pred = max(votes, key=votes.get)
                        output_file.write("%s,%d\n" % (f[:-5], pred))
                    else:
                        pred = outputs[j].data.max(1, keepdim=True)[1]
                        output_file.write("%s,%d\n" % (f[:-5], pred))
        
        print("Succesfully wrote "
              + args.outfile
              + ", you can upload this file to the kaggle competition website")

def predict(model, val_loader, use_cuda):
    # Store in an array the predictions
    outputs = []
    for data, _ in tqdm(val_loader):
        if use_cuda:
            data = data.cuda()
        output = model(data)
        outputs.append(nn.functional.softmax(output, dim=1))
    outputs = torch.cat(outputs, 0).cpu()
    return outputs

def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    test(list_names_models, use_cuda, args)

if __name__ == "__main__":
    main()
