import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import flatten, nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda
from tqdm import tqdm
import torchvision


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
            if np.isinf(beta_max):
                beta *= 2
            else:
                beta = (beta + beta_max) / 2
        else:
            beta_max = beta
            if np.isinf(beta_min):
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
) -> torch.Tensor:
    n = X.shape[0]
    logU = torch.log(torch.tensor([perplexity], device=X.device))

    sum_X = torch.sum(torch.square(X), dim=1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.mT), sum_X).T, sum_X)

    idx = (1 - torch.eye(n)).type(torch.bool)
    D = D[idx].reshape((n, -1))

    P = torch.zeros(n, n)

    for i in range(n):
        P[i, idx[i]] = x2p_job((i, D[i], logU), tolerance)[1]
    return P


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ParametericTSNE:
    def __init__(
        self,
        loss_fn,
        optimizer,
        optimizer_params,
        n_components: int,
        perplexity: int,
        batch_size: int,
        early_exaggeration_epochs: int,
        early_exaggeration_value: float,
        max_iterations: int,
        n_jobs: int = 0,
        tolerance: float = 1e-5,
    ):
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = NeuralNetwork().to(self.device)

        self.n_components = n_components
        self.perplexity = perplexity
        self.batch_size = batch_size
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        self.n_jobs = n_jobs
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.optimizer = self.set_optimizer(optimizer, optimizer_params)
        self.loss_fn = self.set_loss_fn(loss_fn)

    def set_optimizer(self, optimizer: str, optimizer_params: dict):
        if optimizer == "adam":
            return optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer == "sgd":
            return optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer == "rmsprop":
            return optim.RMSprop(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError("Unknown optimizer")

    def set_loss_fn(self, loss_fn):
        if loss_fn == "kl_divergence":
            return self._kl_divergence

    def save_model(self, filename: str):
        torch.save(self.model.state_dict(), filename)

    def read_model(self, filename: str):
        self.model.load_state_dict(torch.load(filename))
        self.model = self.model.to(self.device)

    def split_dataset(self, X, y=None, train_size=None, test_size=None):
        if train_size is None and test_size is None:
            train_size = 0.8
            test_size = 1 - train_size
        elif train_size is None:
            train_size = 1 - test_size  # type: ignore
        elif test_size is None:
            test_size = 1 - train_size  # type: ignore
        if y is None:
            dataset = TensorDataset(X)
        else:
            dataset = TensorDataset(X, y)  # type: ignore
        train_size = int(train_size * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        if train_size > 0:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.n_jobs,
            )
        else:
            train_loader = None
        if test_size > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.n_jobs,
            )
        else:
            test_loader = None

        return train_loader, test_loader

    def _calculate_P(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        n = len(dataloader.dataset)
        P = torch.zeros((n, self.batch_size), device=self.device)
        for i, (X, *_) in tqdm(
            enumerate(dataloader), unit="batches", total=len(dataloader)
        ):
            batch = x2p(X, self.perplexity, self.tolerance)
            batch[torch.isnan(batch)] = 0
            batch = batch + batch.T
            batch = batch / batch.sum()
            batch = torch.maximum(batch, torch.Tensor([1e-12]))  # type: ignore
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


class Classifier(pl.LightningModule):
    def __init__(self, tsne: ParametericTSNE):
        super().__init__()
        self.tsne = tsne
        self.batch_size = tsne.batch_size
        self.model = self.tsne.model
        self.loss_fn = tsne.loss_fn

    def training_step(self, batch, batch_idx):
        x = batch[0]

        p_idxs = torch.randperm(x.shape[0])
        x = x[p_idxs]
        _P_batch = self.P_copy[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        _P_batch = _P_batch[p_idxs, :]
        _P_batch = _P_batch[:, p_idxs]

        logits = self.model(x)
        loss = self.loss_fn(logits, _P_batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return self.tsne.optimizer

    def on_train_start(self) -> None:
        self.P = self.tsne._calculate_P(self.trainer.train_dataloader)

    def on_train_epoch_start(self) -> None:
        self.P_copy = self.P.clone()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE Algorithm")
    parser.add_argument(
        "input_file",
        type=FileTypeWithExtensionCheck(valid_extensions=("txt", "data")),
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
        "-start_dims",
        type=int,
        help="Number of dimensions to start with after initial reduction using PCA",
        required=False,
        default=0,
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
        "-exaggeration",
        type=float,
        help="Early exaggeration end",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-o", type=str, help="Output filename", required=False, default="result.txt"
    )
    parser.add_argument("-model", type=str, help="Model filename", required=False)

    args = parser.parse_args(
        [
            "tsne/colvar-tf.data",
            "-no_dims",
            "2",
            "-perplexity",
            "100",
            "-exclude_cols",
            "-1",
            "0",
            "-step",
            "50",
            "-iter",
            "200",
            "-o",
            "mnist.txt"
            # "-model",
            # "model.pth",
        ]
    )
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([ToTensor(), Lambda(flatten)]),
    )

    batch_size = 1024

    train = DataLoader(
        training_data,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
    )
    test = DataLoader(test_data, batch_size=batch_size, drop_last=True, pin_memory=True)

    for X, y in test:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    tsne = ParametericTSNE(
        "kl_divergence",
        "adam",
        {"lr": 1e-3},
        args.no_dims,
        args.perplexity,
        batch_size,
        0,
        0,
        args.iter,
        5,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        max_epochs=args.iter,
    )
    classifier = Classifier(tsne)

    if args.model:
        tsne.read_model(args.model)
        Y = trainer.predict(test)
    else:
        trainer.fit(classifier, train)
        tsne.save_model("mnist.pth")
        Y = trainer.predict(classifier, test)

    with open(args.o, "w") as f:
        for i, batch in tqdm(enumerate(Y), unit="samples", total=len(Y)):
            for px, py in batch:
                f.writelines(f"{px}\t{py}\n")

    with open("fmnist_cols.txt", "w") as f:
        for row in test_data:
            f.writelines(f"{row[1]}\n")
