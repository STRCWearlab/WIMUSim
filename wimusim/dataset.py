import warnings
from typing import List, Dict
from collections import defaultdict
import random
import itertools
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset

from wimusim.wimusim import WIMUSim
from wimusim.wimusim import utils


class CPM(Dataset):

    def __init__(
        self,
        B_list: List[WIMUSim.Body],
        D_list: List[WIMUSim.Dynamics],
        P_list: List[WIMUSim.Placement],
        H_list: List[WIMUSim.Hardware],
        window: int,
        stride: int,
        acc_only: bool = False,
        gyro_only: bool = False,
        target_list: List[torch.Tensor] = None,
        groups: List = None,  # Optionally used to group the parameters for sampling
        device: torch.device = None,
        dataset_type="supervised",
        scale_config: Dict = None,
    ):
        """
        :param B_list: List of B matrices
        :param dataset_type: when type is supervised, target_list must be provided
        """
        self.B_list = B_list
        self.D_list = D_list
        self.P_list = P_list
        self.H_list = H_list
        self.target_list = target_list
        self.window = window
        self.stride = stride

        self.acc_only = acc_only
        self.gyro_only = gyro_only

        # Groupingの仕様はもう少し考える。基本的には、最終的なClassのバランスを揃えることができるようにしたい。
        # だから、Dynamicsのパラメータに対してだけGroupを指定できるようにするだけでも良いかもしれない。
        self.groups = groups  # Dict["B": List of len(B_list), "D": List of len(D_list), "P": List of len(P_list), "H": List of len(H_list)]

        self._len = None
        self._data = None
        self._target = None

        self._index_range_dict = None
        self._combinations = None

        self._scale_config = scale_config
        # NOTE: Possibly add a list of columns names for the dataset. It would be useful.
        # self.col_names = []

        # Check if CUDA is available
        if device is None:
            if torch.cuda.is_available():
                print("CUDA is available! Use GPU.")
                self.device = torch.device("cuda")
            else:
                print("CUDA is not available. Use CPU.")
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.dataset_type = dataset_type  # supervised or unsupervised

        self._check_parameters()  # validate the parameters
        self._disable_grad()

    def _check_parameters(self):
        """
        Check if the parameters are valid. Also check if all the tensors are on the same device.

        :return:
        """
        for B in self.B_list:
            assert (
                type(B) == WIMUSim.Body
            ), "B_list's elements must be of type WIMUSim.Body"
            assert B.device == self.device

        for D in self.D_list:
            assert (
                type(D) == WIMUSim.Dynamics
            ), "D_list's elements must be of type WIMUSim.Dynamics"
            assert D.device == self.device

        for P in self.P_list:
            assert (
                type(P) == WIMUSim.Placement
            ), "P_list's elements must be of type WIMUSim.Placement"
            assert P.device == self.device

        for H in self.H_list:
            assert (
                type(H) == WIMUSim.Hardware
            ), "H_list's elements must be of type WIMUSim.Hardware"
            assert H.device == self.device

        if self.groups is not None:
            assert len(self.groups) == len(
                self.D_list
            ), "Number of groups must be the same as number of B matrices"
            assert all(
                [group in range(len(self.B_list)) for group in self.groups]
            ), "All groups must be in range of B matrices"

        if self.dataset_type == "supervised":
            assert (
                self.target_list is not None
            ), "Target list must be provided for supervised dataset"
            assert len(self.D_list) == len(
                self.target_list
            ), "Length of D_list and target_list must be the same"

            for i, (D, target) in enumerate(zip(self.D_list, self.target_list)):
                assert (
                    D.n_samples == target.shape[0]
                ), f"Number of samples in D_list[{i}] and target_list[{i}] must be the same"

    def _disable_grad(self):
        """
        Disable the gradient computation for all the parameters
        :return:
        """
        for B in self.B_list:
            for b in B.rp.values():
                b.requires_grad_(False)
        for D in self.D_list:
            for d in D.translation.values():
                d.requires_grad_(False)
            for d in D.orientation.values():
                d.requires_grad_(False)
        for P in self.P_list:
            for p in P.rp.values():
                p.requires_grad_(False)
            for p in P.ro.values():
                p.requires_grad_(False)
        for H in self.H_list:
            for h in H.ba.values():
                h.requires_grad_(False)
            for key, h in H.sa.items():
                H.sa[key] = (
                    torch.Tensor(h.detach()).to(self.device).requires_grad_(False)
                )
                # h.requires_grad_(False)
            for h in H.bg.values():
                h.requires_grad_(False)
            for key, h in H.sg.items():
                H.sg[key] = (
                    torch.Tensor(h.detach()).to(self.device).requires_grad_(False)
                )

                # h.requires_grad_(False)

    def _generate_virtual_imu_data(
        self,
        B: WIMUSim.Body,
        D: WIMUSim.Dynamics,
        P: WIMUSim.Placement,
        H: WIMUSim.Hardware,
        order: str = "alternate",
    ):
        """
        Generate the virtual IMU data from the given parameters
        :return:
        """

        wimusim_env = WIMUSim(B=B, D=D, P=P, H=H)
        virtual_imu_dict = wimusim_env.simulate(mode="generate")
        if self.acc_only:
            virtual_imu_data = torch.concat(
                [virtual_imu_dict[imu_name][0] for imu_name in wimusim_env.P.imu_names],
                dim=1,
            )
        elif self.gyro_only:
            virtual_imu_data = torch.concat(
                [virtual_imu_dict[imu_name][1] for imu_name in wimusim_env.P.imu_names],
                dim=1,
            )
        else:
            if order == "alternate":
                """
                Concatenate the accelerometer and gyroscope data for each IMU in an alternate order
                e.g. IMU1_AccX..Z, IMU1_GyroX..Z, IMU2_AccX..Z, IMU2_GyroX..Z, ...
                """
                virtual_imu_data = torch.concat(
                    [
                        torch.concat(virtual_imu_dict[imu_name], axis=1)
                        for imu_name in wimusim_env.P.imu_names
                    ],
                    axis=1,
                )
            else:
                """
                Concatenate the accelerometer and gyroscope data for each IMU in a sequential order
                e.g. IMU1_AccX..Z, IMU2_AccX..Z, ..., IMU1_GyroX..Z, IMU2_GyroX..Z, ...
                """
                virtual_imu_data = torch.concat(
                    [
                        torch.concat(
                            [
                                virtual_imu_dict[imu_name][0]
                                for imu_name in wimusim_env.P.imu_names
                            ],
                            dim=1,
                        ),
                        torch.concat(
                            [
                                virtual_imu_dict[imu_name][1]
                                for imu_name in wimusim_env.P.imu_names
                            ],
                            dim=1,
                        ),
                    ],
                    dim=1,
                )

        return virtual_imu_data

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.data is None:
            raise ValueError(
                "Data is not loaded yet. Run CPM.generate_data() first to load data."
            )

        for data_idx, idx_range in self._index_range_dict.items():
            if idx in idx_range:
                local_idx = idx - idx_range[0]
                start = local_idx * self.stride
                end = start + self.window
                data = self.data[data_idx][start:end]
                target = self.target[data_idx][end - 1]

                return data, target, idx

        # If the index is out of range
        raise ValueError("Index out of range")

    def generate_data(self, n_combinations: int = -1):
        """
        Generate the Virtual IMU data for all the combinations of parameters
        :param n_combinations: Number of combinations to sample. If -1, all combinations are used.
        :return:
        """

        combinations_list = self._generate_param_combinations(n_combinations)

        print("Generating virtual IMU data...")
        data = []
        target = []
        for B_idx, D_idx, P_idx, H_idx in tqdm(combinations_list):
            virtual_imu_data = self._generate_virtual_imu_data(
                self.B_list[B_idx],
                self.D_list[D_idx],
                self.P_list[P_idx],
                self.H_list[H_idx],
            )

            # Scale the data if needed
            if self._scale_config is not None:
                config = self._scale_config
                if config["type"] == "standardize":
                    if config["mean"] is None:
                        mean = torch.mean(virtual_imu_data, dim=0, device=self.device)
                    else:
                        mean = torch.tensor(config["mean"], device=self.device)

                    if config["std"] is None:
                        std = torch.std(virtual_imu_data, dim=0, device=self.device)
                        std[std == 0] = 1
                    else:
                        std = torch.tensor(config["std"], device=self.device)

                    virtual_imu_data = utils.standardize(
                        virtual_imu_data, mean, std
                    ).float()
                else:
                    raise ValueError(f"Unknown scaling method: {config['type']}")

            if self.dataset_type == "supervised":
                data.append(virtual_imu_data)
                target.append(self.target_list[D_idx])
            else:
                data.append(virtual_imu_data)

        self.data = data
        self.target = target
        self._combinations = combinations_list

    def _generate_param_combinations(
        self,
        n_combinations: int = -1,
    ):
        """
        Sample n_samples from each of the parameter groups and generate the Virtual IMU data
        :param n_samples:
        :return:
        """

        if n_combinations == -1:
            if self.groups is not None:
                warnings.warn(
                    "n_combinations is not provided. self.groups is ignored. Using all combinations."
                )
            combinations_list = list(
                itertools.product(
                    range(len(self.B_list)),
                    range(len(self.D_list)),
                    range(len(self.P_list)),
                    range(len(self.H_list)),
                )
            )

        else:

            if self.groups is None:
                combinations_list = []  # (B_idx, D_idx, P_idx, H_idx)
                for _ in range(n_combinations):
                    # Sample indices for B, D, P, H
                    B_idx = random.randint(0, len(self.B_list) - 1)
                    P_idx = random.randint(0, len(self.P_list) - 1)
                    D_idx = random.randint(0, len(self.D_list) - 1)
                    H_idx = random.randint(0, len(self.H_list) - 1)

                    combination = (B_idx, D_idx, P_idx, H_idx)
                    combinations_list.append(combination)

            else:
                n_groups = len(torch.unique(torch.tensor(self.groups)))

                samples_per_group = n_combinations // n_groups
                remainder = n_combinations % n_groups

                sample_num_per_group = {
                    group_id: samples_per_group for group_id in self.groups
                }
                for group_k in random.choices(self.groups, k=remainder):
                    sample_num_per_group[group_k] += 1

                assert sum(sample_num_per_group.values()) == n_combinations

                group_ids_list = []
                for group_id, count in sample_num_per_group.items():
                    group_ids_list.extend([group_id] * count)
                random.shuffle(group_ids_list)

                # Create a mapping from group_id to D_list indices
                group_to_indices = defaultdict(list)
                for idx, group_id in enumerate(self.groups):
                    group_to_indices[group_id].append(idx)

                combinations_list = []  # (B_idx, D_idx, P_idx, H_idx)
                for group_id in group_ids_list:
                    # Sample indices for B, D, P, H
                    B_idx = random.randint(0, len(self.B_list) - 1)
                    P_idx = random.randint(0, len(self.P_list) - 1)
                    D_idx = random.choice(group_to_indices[group_id])
                    H_idx = random.randint(0, len(self.H_list) - 1)

                    combination = (B_idx, D_idx, P_idx, H_idx)
                    combinations_list.append(combination)

        return combinations_list

    @property
    def data(self):
        if self._data is None:
            print("Warning: 'data' is not set. Run CPM.sample() to generate data.")
        else:
            # Perform validation or processing if needed
            pass
        return self._data

    @data.setter
    def data(self, new_data):
        # Optionally, add validation for the new value here
        self._data = new_data

        idx_range_dict: Dict[int, range] = {}
        for i in range(len(self.data)):
            prev_end = 0 if i == 0 else idx_range_dict[i - 1][-1]
            data_i_n_windows = (self.data[i].shape[0] - self.window) // self.stride + 1
            if i == 0:
                idx_range_dict[i] = range(prev_end, prev_end + data_i_n_windows)
            else:
                idx_range_dict[i] = range(prev_end + 1, prev_end + data_i_n_windows)
            self.len = prev_end + data_i_n_windows  # Update the length of the dataset

        # Update the range of the last element to include the last index
        idx_range_dict[len(self.data) - 1] = range(
            idx_range_dict[len(self.data) - 1][0],
            idx_range_dict[len(self.data) - 1][-1] + 2,
        )

        self._index_range_dict = idx_range_dict
        # TODO: Update the length of the dataset
        # self.len = len(new_data) (or something similar)

    @property
    def len(self):
        return self._len

    @len.setter
    def len(self, new_len):
        self._len = new_len

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, new_target):
        self._target = new_target

    @property
    def scale_config(self):
        return self._scale_config

    @target.setter
    def scale_config(self, new_scale_config):
        print("Setting scale config:", new_scale_config)
        self._scale_config = new_scale_config
