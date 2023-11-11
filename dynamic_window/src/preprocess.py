from __future__ import print_function
import os
import os.path
import numpy as np
import random

import torch
import torch.utils.data as data
import os

from scipy.ndimage import maximum_filter1d
from scipy.ndimage import minimum_filter1d


class RoboCupMSL(data.Dataset):
    """RoboCup MSL Ball Dataset
    Args:
        train (bool): train split or test split.
        win_len (int): number of frames in win.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(
        self,
        root_dir,
        win_len=16,
        train=True,
        cal=False,
        in_dist_test=False,
        transformation_list=["speed", "shuffle", "reverse", "periodic", "identity"],
        train_path="./dataset/train",
        test_path="./dataset/test",
        split_ratios=[7, 6, 5],
    ):
        self.root_dir = root_dir
        self.win_total_datapoints = win_len
        self.train = train
        self.cal = cal
        self.in_dist_test = in_dist_test
        self.traces = []
        self.transformation_list = transformation_list
        self.num_classes = len(self.transformation_list)
        self.train_path = train_path
        self.test_path = test_path
        self.split_ratios = split_ratios

        def dataloader():
            folders = os.listdir(self.train_path)
            train_files = []
            test_files = []
            for i in folders:
                folder = os.path.join(self.train_path, i)
                files = os.listdir(folder)
                for j in files:
                    if j.endswith(".txt"):
                        train_files.append(os.path.join(folder, j))

            random.shuffle(train_files)
            split_points = [
                int(ratio * len(train_files) / sum(self.split_ratios))
                for ratio in self.split_ratios
            ]
            part1, part2, part3 = (
                train_files[: split_points[0]],
                train_files[split_points[0] : split_points[0] + split_points[1]],
                train_files[split_points[0] + split_points[1] :],
            )
            print("partitions", len(part1), len(part2), len(part3))
            tst_data = os.listdir(self.test_path)
            for j in tst_data:
                if j.endswith(".txt"):
                    test_files.append(os.path.join(self.test_path, j))
            return part1, part2, part3, test_files

        (
            self.training_traces,
            self.cal_traces,
            self.id_test_traces,
            self.ood_test_traces,
        ) = dataloader()

        if self.train:
            for training_trace_id in self.training_traces:
                f = open(training_trace_id, "r")
                cur_trace_data = []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    data = [float(num.strip()) for num in line.split(",")]
                    data = data[1:]  # excluding time data
                    if len(data) == 3:
                        cur_trace_data.append(data)
                f.close()
                if len(cur_trace_data) >= 60:  # >= self.win_total_datapoints:
                    self.traces.append(cur_trace_data)  # 3D
        elif self.cal:
            for cal_trace_id in self.cal_traces:
                f = open(cal_trace_id, "r")
                cur_trace_data = []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    # data = [float(item) for item in line.split()]
                    data = [float(num.strip()) for num in line.split(",")]
                    data = data[1:]  # excluding time data

                    if len(data) == 3:
                        cur_trace_data.append(data)
                f.close()
                if len(cur_trace_data) >= 60:  # >= self.win_total_datapoints:
                    self.traces.append(cur_trace_data)  # 3D
        else:
            if self.in_dist_test == True:  # Testing for in-distribution
                for id_test_trace_id in self.id_test_traces:
                    f = open(id_test_trace_id, "r")
                    cur_trace_data = []
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        data = [float(num.strip()) for num in line.split(",")]
                        data = data[1:]  # excluding time data
                        if len(data) == 3:
                            cur_trace_data.append(data)
                    f.close()
                    if len(cur_trace_data) >= 60:  # >= self.win_total_datapoints:
                        self.traces.append(np.asarray(cur_trace_data))  # 3D
            else:
                for ood_test_trace_id in self.ood_test_traces:
                    f = open(ood_test_trace_id, "r")
                    cur_trace_data = []
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        data = [float(num.strip()) for num in line.split(",")]
                        data = data[1:]  # excluding time data
                        if len(data) == 3:
                            cur_trace_data.append(data)
                    f.close()
                    if len(cur_trace_data) >= 60:  # >= self.win_total_datapoints:
                        self.traces.append(cur_trace_data)

    def __len__(self):
        return len(self.traces)

    def transform_win(self, win):  # win is a 2D list
        trans_win = torch.FloatTensor(
            win
        )  # trans_win becomes a 2D tensor of win_len X 12 (sensor measurements for 1 time step)
        trans_win = trans_win.unsqueeze(
            0
        )  # trans_win is now a 3D tensor. 1 (no. of channels for conv) X win_len X 12

        return trans_win

    def apply_filter_on_2D_data(self, input_data, filter_coeffs):  # input_data is 2D
        input_data = np.array(input_data)
        num_cols = len(input_data[0])
        num_rows = len(input_data)

        output = []
        for c in range(num_cols):
            output.append(np.convolve(input_data[:, c], filter_coeffs, "valid"))

        output = np.array(output)
        output = output.transpose()

        return output

    def apply_periodic_filter_on_2D_data(
        self, input_data, filter1_coeffs, filter2_coeffs
    ):  # input_data is 2D
        input_data = np.array(input_data)
        num_cols = len(input_data[0])
        num_rows = len(input_data)

        output = []
        for c in range(num_cols // 2):
            output.append(np.convolve(input_data[:, c], filter1_coeffs, "valid"))

        for c in range(num_cols // 2, num_cols):
            output.append(np.convolve(input_data[:, c], filter2_coeffs, "valid"))

        output = np.array(output)
        output = output.transpose()

        return output

    def apply_erosion(
        self, input_data, filter_size
    ):  # input_data is 2D = win_size X 12
        input_data = np.array(input_data)
        num_cols = len(input_data[0])
        num_rows = len(input_data)

        output = []
        for c in range(num_cols):
            output.append(
                minimum_filter1d(input_data[:, c], size=filter_size, mode="nearest")
            )

        output = np.array(output)
        output = output.transpose()

        return output

    def apply_dilation(
        self, input_data, filter_size
    ):  # input_data is 2D = win_size X 12
        input_data = np.array(input_data)
        num_cols = len(input_data[0])
        num_rows = len(input_data)

        output = []
        for c in range(num_cols):
            output.append(
                maximum_filter1d(input_data[:, c], size=filter_size, mode="nearest")
            )

        output = np.array(output)
        output = output.transpose()

        return output

    def __getitem__(self, idx):  # this is only for CAL/TRAIN
        """
        Returns:
            original win data, transformed win data and the applied transformation
        """

        tracedata = self.traces[idx]  # tracedata = 2D list : time steps X data

        length = len(tracedata)
        if length <= self.win_total_datapoints:
            print(tracedata)
        win_start = random.randint(0, length - self.win_total_datapoints)

        orig_win = tracedata[
            win_start : win_start + self.win_total_datapoints
        ]  # 2D list
        trans_win = tracedata[
            win_start : win_start + self.win_total_datapoints
        ]  # 2D list

        # random transformation selection with 0: 2x speed 1: shuffle, 2: reverse, 3: periodic (forward, backward), 4: Identity
        transform_id = random.randint(0, self.num_classes - 1)

        if self.transformation_list[transform_id] == "low_pass":
            trans_win = self.apply_filter_on_2D_data(trans_win, [1 / 3, 1 / 3, 1 / 3])

        elif self.transformation_list[transform_id] == "high_pass":
            trans_win = self.apply_filter_on_2D_data(trans_win, [-1 / 2, 0, 1 / 2])

        elif self.transformation_list[transform_id] == "dilation":
            trans_win = self.apply_dilation(trans_win, filter_size=3)

        elif self.transformation_list[transform_id] == "erosion":
            trans_win = self.apply_erosion(trans_win, filter_size=3)

        elif (
            self.transformation_list[transform_id] == "low_high"
        ):  # low-pass on half, high pass on half
            trans_win = self.apply_periodic_filter_on_2D_data(
                trans_win,
                filter1_coeffs=[1 / 3, 1 / 3, 1 / 3],
                filter2_coeffs=[-1 / 2, 0, 1 / 2],
            )

        elif (
            self.transformation_list[transform_id] == "high_low"
        ):  # low-pass on half, high pass on half
            trans_win = self.apply_periodic_filter_on_2D_data(
                trans_win,
                filter1_coeffs=[-1 / 2, 0, 1 / 2],
                filter2_coeffs=[1 / 3, 1 / 3, 1 / 3],
            )

        elif self.transformation_list[transform_id] == "identity":  # identity
            pass

        else:
            raise Exception("Invalid transformation")

        # converting to tensors and new dim = 1 (no. of channels for conv) X win_len X 12
        trans_win = self.transform_win(trans_win)

        orig_win = self.transform_win(orig_win)
        orig_win = orig_win[:, 1:-1, :]

        if (
            self.transformation_list[transform_id] == "identity"
            or self.transformation_list[transform_id] == "dilation"
            or self.transformation_list[transform_id] == "erosion"
        ):
            trans_win = trans_win[:, 1:-1, :]

        return orig_win, trans_win, transform_id

    def __get_test_item__(
        self, idx
    ):  # generator for getting sequential shuffled tuples/windows on test data
        tracedata = self.traces[idx]

        length = len(tracedata)

        for i in range(0, length - (1 * self.win_total_datapoints) + 1):
            win_start = i

            transform_id = random.randint(0, self.num_classes - 1)

            orig_win = tracedata[win_start : win_start + self.win_total_datapoints]
            trans_win = tracedata[win_start : win_start + self.win_total_datapoints]

            if self.transformation_list[transform_id] == "low_pass":
                trans_win = self.apply_filter_on_2D_data(
                    trans_win, [1 / 3, 1 / 3, 1 / 3]
                )

            elif self.transformation_list[transform_id] == "high_pass":
                trans_win = self.apply_filter_on_2D_data(trans_win, [-1 / 2, 0, 1 / 2])

            elif self.transformation_list[transform_id] == "dilation":
                trans_win = self.apply_dilation(trans_win, filter_size=3)

            elif self.transformation_list[transform_id] == "erosion":
                trans_win = self.apply_erosion(trans_win, filter_size=3)

            elif (
                self.transformation_list[transform_id] == "low_high"
            ):  # low-pass on half, high pass on half
                trans_win = self.apply_periodic_filter_on_2D_data(
                    trans_win,
                    filter1_coeffs=[1 / 3, 1 / 3, 1 / 3],
                    filter2_coeffs=[-1 / 2, 0, 1 / 2],
                )

            elif (
                self.transformation_list[transform_id] == "high_low"
            ):  # low-pass on half, high pass on half
                trans_win = self.apply_periodic_filter_on_2D_data(
                    trans_win,
                    filter1_coeffs=[-1 / 2, 0, 1 / 2],
                    filter2_coeffs=[1 / 3, 1 / 3, 1 / 3],
                )

            elif self.transformation_list[transform_id] == "identity":  # identity
                pass

            else:
                raise Exception("Invalid transformation")

            # converting to tensors and new dim = 1 (no. of channels for conv) X win_len X 12
            trans_win = self.transform_win(trans_win)

            orig_win = self.transform_win(orig_win)
            orig_win = orig_win[:, 1:-1, :]

            if (
                self.transformation_list[transform_id] == "identity"
                or self.transformation_list[transform_id] == "dilation"
                or self.transformation_list[transform_id] == "erosion"
            ):
                trans_win = trans_win[:, 1:-1, :]

            # if 'periodic' in self.tranformation_list:
            #     trans_win = np.delete(trans_win, len(trans_win)//2)
            #     orig_win = np.delete(orig_win, len(trans_win)//2)

            yield orig_win, trans_win, transform_id
