import argparse
from collections import OrderedDict
import io

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from argparse_range import range_action
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, Subset
from tqdm import tqdm
from kde1d import kde1d
import get_datasets
import sys
from typing import Tuple, Union
import torchinfo


def is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


def normalize_columns(data: torch.Tensor) -> torch.Tensor:
    data_min = data.min(dim=0)[0]
    data_range = data.max(dim=0)[0] - data_min
    return (data - data_min) / data_range


def filter_data_by_variance(
    data: torch.Tensor, variance_threshold: float
) -> Union[torch.Tensor, None]:
    if variance_threshold is None:
        return None
    vars = data.var(axis=0)
    cols = np.where(vars > variance_threshold)[0]
    filtered_data = data[:, cols]
    return filtered_data


def save_means_and_vars(data: torch.Tensor, filtered_data: torch.Tensor = None) -> None:
    means = data.mean(axis=0)
    vars = data.var(axis=0)

    if filtered_data is not None:
        filtered_means = filtered_data.mean(axis=0)
        filtered_vars = filtered_data.var(axis=0)

    with open("means_and_vars.txt", "w") as f:
        f.writelines("column\tmean\tvar\n")
        for v in range(len(means)):
            f.writelines(f"{v}\t{means[v]}\t{vars[v]}\n")
        if filtered_data is not None:
            f.writelines("\nfiltered_column\tfiltered_mean\tfiltered_var\n")
            for v in range(len(filtered_means)):
                f.writelines(f"{v}\t{filtered_means[v]}\t{filtered_vars[v]}\n")


def load_torch_dataset(name: str, step: int, output: str) -> Tuple[Dataset, Dataset]:
    train, test = get_datasets.get_dataset(name)
    train = Subset(train, range(0, len(train), step))

    with open(output.rsplit("/", maxsplit=1)[0] + "/" + name + "_cols.txt", "w") as f:
        for row in test:
            f.writelines(f"{row[1]}\n")

    train_data = torch.stack([row[0] for row in train])
    save_means_and_vars(train_data)

    return train, test


def load_labels(labels: io.TextIOWrapper) -> Union[torch.Tensor, None]:
    read_labels = None
    if labels:
        read_labels = np.loadtxt(labels)
        read_labels = torch.from_numpy(read_labels).float()
        labels.close()
    return read_labels


def load_text_file(
    input_file: io.TextIOWrapper,
    step: int,
    header: bool,
    exclude_cols: list,
    variance_threshold: float,
) -> torch.Tensor:
    cols = None
    if header:
        input_file.readline()
    if exclude_cols:
        last_pos = input_file.tell()
        ncols = len(input_file.readline().strip().split(" "))
        input_file.seek(last_pos)
        cols = np.arange(0, ncols, 1)
        cols = tuple(np.delete(cols, exclude_cols))

    X = np.loadtxt(input_file, usecols=cols)

    input_file.close()

    data = np.array(X[::step, :])

    filtered = filter_data_by_variance(data, variance_threshold)
    save_means_and_vars(data, filtered)
    if filtered is not None:
        data = filtered

    data = torch.from_numpy(data).float()
    data = normalize_columns(data)

    # for i in range(data.shape[1]):
    #     data[:, i] = (data[:, i] - data[:, i].min()) / (
    #         data[:, i].max() - data[:, i].min()
    #     )

    return data


def load_npy_file(
    input_file: io.TextIOWrapper,
    step: int,
    exclude_cols: list,
    variance_threshold: float,
) -> torch.Tensor:
    name = input_file.name
    input_file.close()
    data = np.load(name)
    cols = data.shape[1]
    data = data[::step, :]
    if exclude_cols:
        data = np.delete(data, exclude_cols, axis=1)

    filtered = filter_data_by_variance(data, variance_threshold)
    save_means_and_vars(data, filtered)
    if filtered is not None:
        data = filtered

    data = torch.from_numpy(data).float()
    data = normalize_columns(data)

    # for i in range(data.shape[1]):
    #     data[:, i] = (data[:, i] - data[:, i].min()) / (
    #         data[:, i].max() - data[:, i].min()
    #     )

    return data


def Hbeta(D: torch.Tensor, beta: float) -> tuple:
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p_job(data: tuple, tolerance: float, max_iterations: int = 50) -> tuple:
    i, Di, logU = data
    beta = 1.0
    beta_min = -torch.inf
    beta_max = torch.inf

    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    it = 0
    while it < max_iterations and torch.abs(Hdiff) > tolerance:
        if Hdiff > 0:
            beta_min = beta
            if torch.isinf(torch.tensor(beta_max)):
                beta *= 2
            else:
                beta = (beta + beta_max) / 2
        else:
            beta_max = beta
            if torch.isinf(torch.tensor(beta_min)):
                beta /= 2
            else:
                beta = (beta + beta_max) / 2

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        it += 1
    return i, thisP


def x2p(
    X: torch.Tensor,
    perplexity: int,
    tolerance: float,
    use_kde_diff: bool,
) -> torch.Tensor:
    n = X.shape[0]
    logU = torch.log(torch.tensor([perplexity], device=X.device))

    sum_X = torch.sum(torch.square(X), dim=1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.mT), sum_X).T, sum_X)

    idx = (1 - torch.eye(n)).type(torch.bool)
    # if not use_kde_diff:
    D = D[idx].reshape((n, -1))

    P = torch.zeros(n, n, device=X.device)

    for i in range(n):
        if not use_kde_diff:
            P[i, idx[i]] = x2p_job((i, D[i], logU), tolerance)[1]
        else:
            density = kde1d(D[i], n)[0]
            # with open("density.txt", "a") as f:
            #     for number in density:
            #         f.write(str(number) + " ")
            #     f.write("\n")
            density = torch.from_numpy(density).float().to(X.device)
            normalized_density = density / torch.sum(density)
            P[i, :] = normalized_density
    if use_kde_diff:
        P = P * idx
    return P


class NeuralNetwork(nn.Module):
    def __init__(
        self, initial_features: int, n_components: int, multipliers: list
    ) -> None:
        super(NeuralNetwork, self).__init__()
        layers = OrderedDict()
        layers["0"] = nn.Linear(
            initial_features, int(multipliers[0] * initial_features)
        )
        for i in range(1, len(multipliers)):
            layers["ReLu" + str(i - 1)] = nn.ReLU()
            layers[str(i)] = nn.Linear(
                int(multipliers[i - 1] * initial_features),
                int(multipliers[i] * initial_features),
            )
            layers["ReLu" + str(i)] = nn.ReLU()
        if len(multipliers) == 1:
            layers["ReLu" + str(len(multipliers) - 1)] = nn.ReLU()
        layers[str(len(multipliers))] = nn.Linear(
            int(multipliers[-1] * initial_features), n_components
        )
        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ParametericTSNE:
    def __init__(
        self,
        loss_fn,
        n_components: int,
        perplexity: int,
        batch_size: int,
        early_exaggeration_epochs: int,
        early_exaggeration_value: float,
        max_iterations: int,
        features: int,
        multipliers: list,
        n_jobs: int = 0,
        tolerance: float = 1e-5,
        use_kde_diff: bool = False,
        force_cpu: bool = False,
    ):
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.model = NeuralNetwork(features, n_components, multipliers).to(self.device)
        torchinfo.summary(
            self.model,
            input_size=(batch_size, 1, features),
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
        )

        self.n_components = n_components
        self.perplexity = perplexity
        self.batch_size = batch_size
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        self.n_jobs = n_jobs
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.use_kde_diff = use_kde_diff

        self.loss_fn = self.set_loss_fn(loss_fn)

    def set_loss_fn(self, loss_fn):
        if loss_fn == "kl_divergence":
            return self._kl_divergence

    def save_model(self, filename: str):
        torch.save(self.model.state_dict(), filename)

    def read_model(self, filename: str):
        self.model.load_state_dict(torch.load(filename))

    def split_dataset(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        train_size: float = None,
        test_size: float = None,
    ) -> Tuple[DataLoader, DataLoader]:
        if train_size is None and test_size is None:
            train_size = 0.8
            test_size = 1 - train_size
        elif train_size is None:
            train_size = 1 - test_size
        elif test_size is None:
            test_size = 1 - train_size

        # X, y = self._adjust_size(X, y)

        if y is None:
            dataset = TensorDataset(X)
        else:
            dataset = TensorDataset(X, y)
        train_size = int(train_size * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        if train_size == 0:
            train_dataset = None
        if test_size == 0:
            test_dataset = None

        return self.create_dataloaders(train_dataset, test_dataset)

    def create_dataloaders(
        self, train: Dataset, test: Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = (
            DataLoader(
                train,
                batch_size=self.batch_size,
                drop_last=True,
                pin_memory=True,
                num_workers=self.n_jobs if self.device == "cpu" else 0,
            )
            if train is not None
            else None
        )
        test_loader = (
            DataLoader(
                test,
                batch_size=self.batch_size,
                drop_last=True,
                pin_memory=True,
                num_workers=self.n_jobs if self.device == "cpu" else 0,
            )
            if test is not None
            else None
        )
        return train_loader, test_loader

    def _calculate_P(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        n = len(dataloader.dataset)
        P = torch.zeros((n, self.batch_size), device=self.device)
        for i, (X, *_) in tqdm(
            enumerate(dataloader), unit="batch", total=len(dataloader)
        ):
            batch = x2p(X, self.perplexity, self.tolerance, self.use_kde_diff)
            batch[torch.isnan(batch)] = 0
            if not self.use_kde_diff:
                batch = batch + batch.mT
                batch = batch / batch.sum()
            batch = torch.maximum(
                batch.to(self.device), torch.tensor([1e-12], device=self.device)
            )
            P[i * self.batch_size : (i + 1) * self.batch_size] = batch
        return P

    def _kl_divergence(self, Y: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        sum_Y = torch.sum(torch.square(Y), dim=1)
        eps = torch.tensor([1e-15], device=self.device)
        D = sum_Y + torch.reshape(sum_Y, [-1, 1]) - 2 * torch.matmul(Y, Y.mT)
        Q = torch.pow(1 + D / 1.0, -(1.0 + 1) / 2)
        Q *= 1 - torch.eye(self.batch_size, device=self.device)
        Q /= torch.sum(Q)
        Q = torch.maximum(Q, eps)
        C = torch.log((P + eps) / (Q + eps))
        C = torch.sum(P * C)
        return C

    def _adjust_size(self, X, y=None):
        if X.shape[0] % self.batch_size != 0:
            X = X[: -(X.shape[0] % self.batch_size)]
        if y is not None:
            if y.shape[0] % self.batch_size != 0:
                y = y[: -(y.shape[0] % self.batch_size)]
        return X, y


class Classifier(pl.LightningModule):
    def __init__(
        self,
        tsne: ParametericTSNE,
        shuffle: bool,
        optimizer: str = "adam",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.tsne = tsne
        self.batch_size = tsne.batch_size
        self.model = self.tsne.model
        self.loss_fn = tsne.loss_fn
        self.exaggeration_epochs = tsne.early_exaggeration_epochs
        self.exaggeration_value = tsne.early_exaggeration_value
        self.shuffle = shuffle
        self.lr = lr
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x = batch[0]
        _P_batch = self.P_copy[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]

        if self.shuffle:
            p_idxs = torch.randperm(x.shape[0])
            x = x[p_idxs]
            _P_batch = _P_batch[p_idxs, :]
            _P_batch = _P_batch[:, p_idxs]

        logits = self.model(x)
        loss = self.loss_fn(logits, _P_batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def _set_optimizer(self, optimizer: str, optimizer_params: dict):
        if optimizer == "adam":
            return optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer == "sgd":
            return optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer == "rmsprop":
            return optim.RMSprop(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError("Unknown optimizer")

    def configure_optimizers(self):
        return self._set_optimizer(self.optimizer, {"lr": self.lr})

    def on_train_start(self) -> None:
        self.P = self.tsne._calculate_P(self.trainer.train_dataloader)

    def on_train_epoch_start(self) -> None:
        self.P_copy = self.P.clone()
        if (
            self.exaggeration_epochs > 0
            and self.current_epoch < self.exaggeration_epochs
        ):
            self.P_copy *= self.exaggeration_value

    def on_train_epoch_end(self) -> None:
        del self.P_copy

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model(batch[0])


class FileTypeWithExtensionCheck(argparse.FileType):
    def __init__(self, mode="r", valid_extensions=None, **kwargs):
        super().__init__(mode, **kwargs)
        self.valid_extensions = valid_extensions

    def __call__(self, string):
        if self.valid_extensions:
            if not string.endswith(self.valid_extensions):
                raise argparse.ArgumentTypeError("Not a valid filename extension!")
        return super().__call__(string)


class FileTypeWithExtensionCheckPredefined(FileTypeWithExtensionCheck):
    def __call__(self, string):
        if len(available_datasets) > 0 and string in available_datasets:
            return string
        if self.valid_extensions:
            if not string.endswith(self.valid_extensions):
                raise argparse.ArgumentTypeError("Not a valid filename extension!")
        return super().__call__(string)


if __name__ == "__main__":
    if "get_datasets" in sys.modules:
        global available_datasets
        available_datasets = get_datasets.get_available_datasets()

    parser = argparse.ArgumentParser(description="t-SNE Algorithm")
    parser.add_argument(
        "input_file",
        type=FileTypeWithExtensionCheckPredefined(
            valid_extensions=("txt", "data", "npy")
        ),
        help="Input file",
    )
    parser.add_argument(
        "-iter", type=int, default=1000, help="Number of iterations", required=False
    )
    parser.add_argument(
        "-labels",
        type=FileTypeWithExtensionCheck(valid_extensions=("txt", "data")),
        help="Labels file",
        required=False,
    )
    parser.add_argument(
        "-no_dims", type=int, help="Number of dimensions", required=True, default=2
    )
    parser.add_argument(
        "-perplexity",
        type=float,
        help="Perplexity of the Gaussian kernel",
        required=True,
        default=30.0,
    )
    parser.add_argument(
        "-exclude_cols", type=int, nargs="+", help="Columns to exclude", required=False
    )
    parser.add_argument(
        "-step", type=int, help="Step between samples", required=False, default=1
    )
    parser.add_argument(
        "-exaggeration_iter",
        type=int,
        help="Early exaggeration end",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-exaggeration_value",
        type=float,
        help="Early exaggeration value",
        required=False,
        default=12,
    )
    parser.add_argument(
        "-o", type=str, help="Output filename", required=False, default="result.txt"
    )
    parser.add_argument(
        "-model_save",
        type=str,
        help="Model save filename",
        required=False,
    )
    parser.add_argument(
        "-model_load",
        type=str,
        help="Model filename to load",
        required=False,
    )
    parser.add_argument("-shuffle", action="store_true", help="Shuffle data")
    parser.add_argument(
        "-train_size",
        type=float,
        action=range_action(0, 1),
        help="Train size",
        required=False,
    )
    parser.add_argument(
        "-test_size",
        type=float,
        action=range_action(0, 1),
        help="Test size",
        required=False,
    )
    parser.add_argument(
        "-kde_diff",
        action="store_true",
        help="Use KDE instead of calculating distances to probabilities",
    )
    parser.add_argument(
        "-jobs", type=int, help="Number of jobs", required=False, default=1
    )
    parser.add_argument(
        "-batch_size", type=int, help="Batch size", required=False, default=1000
    )

    parser.add_argument("-header", action="store_true", help="Data has header")
    parser.add_argument(
        "-net_multipliers",
        type=float,
        nargs="+",
        help="Network multipliers",
        default=[0.75, 0.75, 0.75],
    )
    parser.add_argument("-variance_threshold", type=float, help="Variance threshold")
    parser.add_argument("-cpu", action="store_true", help="Use CPU")
    parser.add_argument(
        "-early_stopping_delta", type=float, help="Early stopping delta", default=1e-5
    )
    parser.add_argument(
        "-early_stopping_patience", type=int, help="Early stopping patience", default=3
    )

    args = parser.parse_args(
        [
            # "tsne/colvar-ala1-wtm-tail.npy",
            # "swiss_roll_2.txt",
            "fashion_mnist",
            "-no_dims",
            "2",
            # "-labels",
            # "swiss_roll_colors.txt",
            "-perplexity",
            "30",
            "-step",
            "1",
            "-iter",
            "1000",
            "-o",
            "praca/fmnist.txt",
            "-model_save",
            "praca/fmnist.pth",
            "-shuffle",
            # "-kde_diff",
            "-jobs",
            "6",
            # "-exclude_cols",
            # "-1",
            # "-header",
            "-batch_size",
            "1024",
            # "-train_size",
            # "1",
            "-net_multipliers",
            # "1",
            "2",
            "3",
            "2",
            "1",
            "0.8",
            "0.75",
            "0.6",
            "0.5",
            # "0.45",
            # "1.5",
            # "1.7",
            # "0.9",
            # "1.2",
            # "0.6",
            # "0.85",
            # "-variance_threshold",
            # "1e-4",
        ]
    )

    if args.kde_diff and not is_power_of_2(args.batch_size):
        raise ValueError("Batch size should be a power of 2 when using kde_diff")

    skip_loading_data = False
    if (
        not isinstance(args.input_file, io.TextIOWrapper)
        and len(available_datasets) > 0
        and (name := args.input_file.lower()) in available_datasets
    ):
        train, test = load_torch_dataset(name, args.step, args.o)
        skip_loading_data = True
        shape = np.prod(train.dataset.data.shape[1:])

    if not skip_loading_data:
        labels = load_labels(args.labels)

        if args.input_file.name.endswith(".npy"):
            data = load_npy_file(
                args.input_file, args.step, args.exclude_cols, args.variance_threshold
            )
        else:
            data = load_text_file(
                args.input_file,
                args.step,
                args.header,
                args.exclude_cols,
                args.variance_threshold,
            )
        shape = data.shape[1]

    tsne = ParametericTSNE(
        "kl_divergence",
        args.no_dims,
        args.perplexity,
        args.batch_size,
        args.exaggeration_iter,
        args.exaggeration_value,
        args.iter,
        shape,
        args.net_multipliers,
        args.jobs,
        use_kde_diff=args.kde_diff,
        force_cpu=args.cpu,
    )

    early_stopping = EarlyStopping(
        "train_loss_epoch",
        min_delta=args.early_stopping_delta,
        patience=args.early_stopping_patience,
        verbose=False,
    )

    is_gpu = tsne.device == torch.device("cuda:0")
    trainer = pl.Trainer(
        accelerator="gpu" if is_gpu else "cpu",
        devices=1 if is_gpu else tsne.n_jobs,
        log_every_n_steps=1,
        max_epochs=args.iter,
        callbacks=[early_stopping],
        auto_lr_find=True,
    )
    classifier = Classifier(tsne, args.shuffle)

    if args.model_load:
        tsne.read_model(args.model_load)
        train, test = (
            tsne.split_dataset(data, y=labels, test_size=1)
            if not skip_loading_data
            else tsne.create_dataloaders(train, test)
        )
        Y = trainer.predict(classifier, test)
    else:
        if args.train_size is not None:
            train_size = args.train_size
        elif args.test_size is not None:
            train_size = 1 - args.test_size
        else:
            train_size = 0.8
        train, test = (
            tsne.split_dataset(data, y=labels, train_size=train_size)
            if not skip_loading_data
            else tsne.create_dataloaders(train, test)
        )
        trainer.fit(classifier, train)
        if args.model_save:
            tsne.save_model(args.model_save)
        if test is not None:
            Y = trainer.predict(classifier, test)

    if test is not None:
        with open(args.o, "w") as f:
            f.writelines(f"{args.step}\n")
            for i, batch in tqdm(enumerate(Y), unit="samples", total=(len(Y))):
                for px, py in batch:
                    f.writelines(f"{px}\t{py}\n")
