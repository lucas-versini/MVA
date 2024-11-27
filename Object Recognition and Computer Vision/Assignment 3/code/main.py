import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets

from model_factory import ModelFactory
from data import cutmix_data, cutmix_criterion, mixup_data, mixup_criterion

num_classes = 500

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument("--data",
                        type = str,
                        default = "data_sketches",
                        metavar = "D",
                        help = "folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument("--model_name",
                        type = str,
                        default = "deit",
                        metavar = "MOD",
                        help = "Name of the model for model and transform instantiation")
    parser.add_argument("--batch-size",
                        type = int,
                        default = 64,
                        metavar = "B",
                        help = "input batch size for training (default: 64)")
    parser.add_argument("--epochs",
                        type = int,
                        default = 10,
                        metavar = "N",
                        help = "number of epochs to train (default: 10)")
    parser.add_argument("--lr",
                        type = float,
                        default = 0.0001,
                        metavar = "LR",
                        help = "learning rate (default: 0.0001)")
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        metavar = "M",
                        help = "SGD momentum (default: 0.9)")
    parser.add_argument("--seed",
                        type = int,
                        default = 1,
                        metavar = "S",
                        help = "random seed (default: 1)")
    parser.add_argument("--log-interval",
                        type = int,
                        default = 10,
                        metavar = "N",
                        help = "how many batches to wait before logging training status")
    parser.add_argument("--experiment",
                        type = str,
                        default = "experiment",
                        metavar = "E",
                        help = "folder where experiment outputs are located.")
    parser.add_argument("--num_workers",
                        type = int,
                        default = 10,
                        metavar = "NW",
                        help = "number of workers for data loading")
    parser.add_argument("--label_smoothing",
                        type = int,
                        default = 0,
                        metavar = "LS",
                        help = "label smoothing")
    parser.add_argument("--model_name_save",
                        type = str,
                        default = "",
                        metavar = "MN",
                        help = "name of the model to save")
    parser.add_argument("--weight_decay",
                        type = float,
                        default = 0,
                        metavar = "WD",
                        help = "weight decay")
    parser.add_argument("--data_transformation",
                        type = int,
                        default = 0,
                        help = "data transformation (-1 for AugMix)")
    parser.add_argument("--cutmix",
                        type = int,
                        default = 0,
                        help = "use CutMix")
    parser.add_argument("--mixup",
                        type = int,
                        default = 0,
                        help = "use MixUp")
    parser.add_argument("--checkpoint",
                        type = str,
                        default = "",
                        help = "use a checkpoint model")
    parser.add_argument("--optimizer",
                        type = str,
                        default = "adam",
                        help = "optimizer to use")
    parser.add_argument("--binary",
                        type = int,
                        default = 0,
                        help = "use binary images")
    args = parser.parse_args()
    return args

def train(model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: torch.utils.data.DataLoader,
          use_cuda: bool, epoch: int, args: argparse.ArgumentParser) -> None:
    """Default training loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        if args.label_smoothing:
            coeff = 0.1
            smooth_target = torch.full(size = (target.size(0), num_classes), fill_value = coeff / num_classes).to(target.device)
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - coeff)
        if args.binary:
            data[:, 0, :, :] = data[:, 0, :, :] > data[:, 0, :, :].mean()
            data[:, 1, :, :] = data[:, 1, :, :] > data[:, 1, :, :].mean()
            data[:, 2, :, :] = data[:, 2, :, :] > data[:, 2, :, :].mean()

        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

        # CutMix and MixUp
        if args.cutmix:
            data, target_a, target_b, lam = cutmix_data(data, target, 1.0, use_cuda)
            output = model(data)
            loss = cutmix_criterion(criterion, output, target_a, target_b, lam)
        elif args.mixup:
            data, target_a, target_b, lam = mixup_data(data, target, 1.0, use_cuda)
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target if not args.label_smoothing else smooth_target)
            
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100.0 * batch_idx / len(train_loader),
                                                                           loss.data.item()))
    print("\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(correct,
                                                            len(train_loader.dataset),
                                                            100.0 * correct / len(train_loader.dataset)))

def validation(model: nn.Module, val_loader: torch.utils.data.DataLoader, use_cuda: bool) -> float:
    """Default validation loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction = "mean")
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        print("\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(validation_loss,
                                                                                         correct,
                                                                                         len(val_loader.dataset),
                                                                                         100.0 * correct / len(val_loader.dataset)))
        return validation_loss

def main():
    """Default main function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Load model and transform
    model, (data_transforms_train, data_transforms_test) = ModelFactory(args.model_name, args.data_transformation).get_all()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + "/train_images", transform = data_transforms_train),
                                               batch_size = args.batch_size,
                                               shuffle = True,
                                               num_workers = args.num_workers)
    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + "/val_images", transform=data_transforms_test),
                                             batch_size = args.batch_size,
                                             shuffle = False,
                                             num_workers = args.num_workers)

    # Setup optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = [0.9, 0.999])
    elif args.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, betas = [0.9, 0.999])
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
    print(f"Using {args.optimizer} optimizer\n")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # training loop
        train(model, optimizer, train_loader, use_cuda, epoch, args)
        scheduler.step()
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best" + args.model_name_save + ".pth"
            torch.save(model.state_dict(), best_model_file)

        print(f". You can run `python evaluate.py --model_name {args.model_name} --model "
              + model_file
              + "` to generate the Kaggle formatted csv file\n")

if __name__ == "__main__":
    main()
