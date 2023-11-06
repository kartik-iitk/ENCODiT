"""Applied temporal transformation prediction.
python train_gait.py --log test --transformation_list low_pass high_pass identity
"""
import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from model import Regressor as regressor

from preprocess import RoboCupMSL


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        orig_data_wins, transformed_data_wins, transformation = data
        orig_data_wins = orig_data_wins.to(device)
        transformed_data_wins = transformed_data_wins.to(device)
        target_transformations = transformation.clone().detach().to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(orig_data_wins, transformed_data_wins)  # return logits here
        loss = criterion(outputs, target_transformations)
        # print("Outputs: {}, targets: {}, loss: {}".format(outputs, target_transformations, loss.item()))
        loss.backward()
        optimizer.step()
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(target_transformations == pts).item()
        # print statistics and write summary every N batch
        # if i % args.pf == 0:
    avg_loss = total_loss / len(train_dataloader)
    avg_acc = correct / len(train_dataloader.dataset)
    print(
        "[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}".format(
            epoch, i, avg_loss, avg_acc
        )
    )
    step = (epoch - 1) * len(train_dataloader) + i
    writer.add_scalar("train/CrossEntropyLoss", avg_loss, step)
    writer.add_scalar("train/Accuracy", avg_acc, step)


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader, 1):
        # get inputs
        orig_data_wins, transformed_data_wins, transformation = data
        orig_data_wins = orig_data_wins.to(device)
        transformed_data_wins = transformed_data_wins.to(device)
        target_transformations = transformation.clone().detach().to(device)
        # forward
        outputs = model(orig_data_wins, transformed_data_wins)  # return logits here
        loss = criterion(outputs, target_transformations)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(target_transformations == pts).item()
        # print('correct: {}, {}, {}'.format(correct, target_transformations, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    writer.add_scalar("val/CrossEntropyLoss", avg_loss, epoch)
    writer.add_scalar("val/Accuracy", avg_acc, epoch)
    print("[VAL] loss: {:.3f}, acc: {:.3f}".format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        orig_data_wins, transformed_data_wins, transformation = data
        orig_data_wins = orig_data_wins.to(device)
        transformed_data_wins = transformed_data_wins.to(device)
        target_transformations = torch.tensor(transformation).to(device)
        # forward
        outputs = model(orig_data_wins, transformed_data_wins)
        print("output: {}, target: {}".format(outputs, target_transformations))
        loss = criterion(outputs, target_transformations)
        # compute loss and acc
        total_loss += loss.item()
        print("Loss: ", loss.item())
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(target_transformations == pts).item()
        # print('correct: {}, {}, {}'.format(correct, target_transformations, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print("[TEST] loss: {:.3f}, acc: {:.3f}".format(avg_loss, avg_acc))
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Video Win Order Prediction")
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--wl", type=int, default=10, help="win length")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--wgtDecay", default=5e-4, type=float)
    parser.add_argument("--lrMul", type=float, default=10.0)
    parser.add_argument("--momentum", type=float, default=0.95, help="momentum")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--log", type=str, default="./log", help="log directory")
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path")
    parser.add_argument("--desp", type=str, help="additional description")
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument("--bs", type=int, default=3, help="mini-batch size")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument(
        "--pf", type=int, default=100, help="print frequency every batch"
    )
    parser.add_argument(
        "--seed", type=int, default=100, help="seed for initializing training."
    )
    parser.add_argument(
        "--train_root_dir",
        type=str,
        default="./dataset/train",
        help="training data directory",
    )
    parser.add_argument(
        "--cal_root_dir",
        type=str,
        default=".",
        help="calibration data directory",
    )
    parser.add_argument(
        "--test_root_dir",
        type=str,
        default="./dataset/test",
        help="test data directory",
    )
    parser.add_argument(
        "--transformation_list",
        "--names-list",
        nargs="+",
        default=["low_pass", "high_pass", "identity", "high_low"],
    )  # high_pass low_high high_low identity
    parser.add_argument(
        "--split_ratio",
        type=list,
        default=[7, 6, 5],
        help="Ratio to split the dataset into train, calibration and test",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))

    # Use CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:{}".format(args.gpu))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    cudnn.benchmark = True

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    net = regressor(num_classes=len(args.transformation_list)).to(device)

    if args.ckpt != "":
        net.load_state_dict(torch.load(args.ckpt))

    if args.mode == "train":  ########### Train #############
        if args.ckpt:  # resume training
            net.load_state_dict(torch.load(args.ckpt))
            log_dir = os.path.dirname(args.ckpt)
        else:
            if args.desp:
                exp_name = "wl{}_{}_{}".format(
                    args.wl, args.desp, time.strftime("%m%d%H%M")
                )
            else:
                exp_name = "wl{}_{}".format(args.wl, time.strftime("%m%d%H%M"))
            log_dir = os.path.join(args.log, exp_name)
        writer = SummaryWriter(log_dir)

        train_dataset = RoboCupMSL(
            root_dir=args.train_root_dir,
            win_len=args.wl,
            train=True,
            cal=False,
            transformation_list=args.transformation_list,
            train_path=args.train_root_dir,
            test_path=args.test_root_dir,
            split_ratios=args.split_ratio,
        )

        val_dataset = RoboCupMSL(
            root_dir=args.cal_root_dir,
            win_len=args.wl,
            train=False,
            cal=True,
            in_dist_test=False,
            transformation_list=args.transformation_list,
            train_path=args.train_root_dir,
            test_path=args.test_root_dir,
            split_ratios=args.split_ratio,
        )

        print(
            "TRAIN video number: {}, VAL video number: {}.".format(
                len(train_dataset), len(val_dataset)
            )
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            shuffle=False,
            num_workers=args.workers,
            multiprocessing_context="forkserver",  # Important on MacOS
            persistent_workers=True,  # Important on MacOS
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.bs,
            shuffle=False,
            num_workers=args.workers,
            multiprocessing_context="forkserver",  # Important on MacOS
            persistent_workers=True,  # Important on MacOS
        )

        ### loss function, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()

        # setup optimizer
        optimizer = optim.Adam(
            params=net.parameters(), lr=args.lr, weight_decay=args.wgtDecay
        )

        prev_best_val_loss = float("inf")
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            time_start = time.time()
            train(
                args, net, criterion, optimizer, device, train_dataloader, writer, epoch
            )
            # print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            val_loss = validate(
                args, net, criterion, device, val_dataloader, writer, epoch
            )
            # scheduler.step(val_loss)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            # save model every 20 epoches
            if epoch % 20 == 0:
                torch.save(
                    net.state_dict(), os.path.join(log_dir, "model_{}.pt".format(epoch))
                )
            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(log_dir, "best_model_{}.pt".format(epoch))
                torch.save(net.state_dict(), model_path)
                prev_best_val_loss = val_loss
                if prev_best_model_path:
                    os.remove(prev_best_model_path)
                prev_best_model_path = model_path

    elif args.mode == "test":  ########### Test #############
        net.load_state_dict(torch.load(args.ckpt))
        net.eval()

        test_dataset = RoboCupMSL(
            root_dir=args.test_root_dir,
            win_len=args.wl,
            train=False,
            cal=False,
            in_dist_test=False,
            transformation_list=args.transformation_list,
            train_path=args.train_root_dir,
            test_path=args.test_root_dir,
            split_ratios=args.split_ratio,
        )
        print("Test dataset len: ", test_dataset.__len__())
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.bs,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            multiprocessing_context="forkserver",  # Important on MacOS
            persistent_workers=True,  # Important on MacOS
        )
        print("TEST video number: {}.".format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, net, criterion, device, test_dataloader)
