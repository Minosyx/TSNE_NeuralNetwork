import argparse
import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.optim as optim
from tqdm import tqdm


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
    X: torch.Tensor, perplexity: int, tolerance: float, n_jobs: int
) -> torch.Tensor:
    n = X.shape[0]
    logU = torch.log(torch.tensor([perplexity], device=X.device))

    sum_X = torch.sum(torch.square(X), dim=1)  # tensor contains only value 16
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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
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
        self.__device = device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = NeuralNetwork().to(device)

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

    def fit(self, dataloader: torch.utils.data.DataLoader):  # type: ignore
        size = len(dataloader.dataset)
        P = self._calculate_P(dataloader)

        epoch = 0
        while epoch < self.max_iterations:
            train_loss = 0
            epoch += 1
            print(f"Epoch {epoch}/{self.max_iterations}")
            P_copy = P.clone()

            if epoch < self.early_exaggeration_epochs:
                P_copy *= self.early_exaggeration_value

            self.model.train()
            for i, (X, *y) in tqdm(
                enumerate(dataloader), unit="batch", total=len(dataloader)
            ):
                X = X.to(self.__device)
                if len(y) > 0:
                    y = y[0].to(self.__device)
                self.optimizer.zero_grad()

                y_pred = self.model(X)

                # inp = torch.log_softmax(y_pred, dim=1)
                loss = self.loss_fn(
                    y_pred, P_copy[i * self.batch_size : (i + 1) * self.batch_size]
                )
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= size
            tqdm.write(f"Loss: {train_loss:>8f}")
            del P_copy
        return self

    def transform(self, dataloader):
        self.model.eval()
        i = 0
        results = np.empty(shape=(0, self.n_components))
        with torch.no_grad():
            for X, *y in dataloader:
                X = X.to(self.__device)
                X_new = self.model(X)
                results = np.append(results, X_new.cpu().detach().numpy(), axis=0)
                i += 1
        return results

    def fit_transform(self, dataloader: torch.utils.data.DataLoader):  # type: ignore
        self.fit(dataloader)
        X_new = self.transform(dataloader)
        return X_new

    def test(self, dataloader: torch.utils.data.DataLoader):  # type: ignore
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, *y in dataloader:
                X = X.to(self.__device)
                if len(y) > 0:
                    y = y[0].to(self.__device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                test_loss += loss.item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n)"
        )

    def split_dataset(self, X, y=None, train_size=None, test_size=None):
        if train_size is None and test_size is None:
            train_size = 0.8
            test_size = 1 - train_size
        elif train_size is None:
            train_size = 1 - test_size  # type: ignore
        elif test_size is None:
            test_size = 1 - train_size  # type: ignore

        # X, y = self._adjust_size(X, y)

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
                shuffle=True,
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
        P = torch.zeros((n, self.batch_size), device=self.__device)
        for i, (X, *_) in tqdm(
            enumerate(dataloader), unit="batch", total=len(dataloader)
        ):
            batch = x2p(X, self.perplexity, self.tolerance, self.n_jobs)
            batch[torch.isnan(batch)] = 0
            batch = batch + batch.T
            batch = batch / batch.sum()
            batch = torch.maximum(batch, torch.Tensor([1e-12]))  # type: ignore
            P[i * self.batch_size : (i + 1) * self.batch_size] = batch
        return P

    def _kl_divergence(self, Y: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        sum_Y = torch.sum(torch.square(Y), dim=1)
        eps = torch.tensor([1e-15], device=self.__device)
        D = sum_Y + torch.reshape(sum_Y, [-1, 1]) - 2 * torch.matmul(Y, Y.mT)
        Q = torch.pow(1 + D / 1.0, -(1.0 + 1) / 2)
        Q *= 1 - torch.eye(self.batch_size, device=self.__device)
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
    def __init__(self, tsne: ParametericTSNE):
        super().__init__()
        self.tsne = tsne
        self.batch_size = tsne.batch_size
        self.model = self.tsne.model
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = self.P_copy[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x = batch
    #     y = self.P_copy[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
    #     logits = self.model(x)
    #     loss = self.loss_fn(logits, y)
    #     self.log("val_loss", loss)

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
    # parser.add_argument('-max_rows', type=int, help='Number of rows to read', required=False)
    # parser.add_argument('-skip_rows', type=int, help='Number of rows to skip', default=0, required=False)
    parser.add_argument(
        "-o", type=str, help="Output filename", required=False, default="result.txt"
    )
    parser.add_argument("-model", type=str, help="Model filename", required=False)

    # print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")
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
            "100",
            "-iter",
            "25",
            "-o",
            "lightning.txt"
            # "-model",
            # "model.pth",
        ]
    )
    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels)
        args.labels.close()

    cols = None
    if args.exclude_cols:
        args.input_file.readline()
        last_pos = args.input_file.tell()
        ncols = len(args.input_file.readline().strip().split(" "))
        args.input_file.seek(last_pos)
        cols = np.arange(0, ncols, 1)
        cols = tuple(np.delete(cols, args.exclude_cols))

    # X = np.loadtxt(args.input_file, usecols=cols, max_rows=args.max_rows, skiprows=args.skip_rows)
    X = np.loadtxt(args.input_file, usecols=cols)

    args.input_file.close()

    data = np.array(X[:: args.step, :])

    means = data.mean(axis=0)
    vars = data.var(axis=0)

    with open("means_and_vars.txt", "w") as f:
        f.writelines("column\tmean\tvar\n")
        for v in range(len(means)):
            # print(f"Column {v} mean: {means[v]}, var: {vars[v]}")
            f.writelines(f"{v}\t{means[v]}\t{vars[v]}\n")

    data = torch.from_numpy(data).float()
    # loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    tsne = ParametericTSNE(
        "kl_divergence",
        "adam",
        {"lr": 1e-3},
        args.no_dims,
        args.perplexity,
        1000,
        0,
        0,
        args.iter,
        5,
    )

    if args.model:
        tsne.read_model(args.model)
        train, test = tsne.split_dataset(data, test_size=1)
        Y = tsne.transform(test)
    else:
        train, test = tsne.split_dataset(data, train_size=0.8)
        model = NeuralNetwork()
        classifier = Classifier(tsne)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            log_every_n_steps=1,
            max_epochs=args.iter,
        )
        trainer.fit(classifier, train)
        # tsne.fit(train)
        tsne.save_model("model.pth")
        Y = trainer.predict(classifier, test)

    with open(args.o, "w") as f:
        f.writelines(f"{args.step}\n")
        for i, (X, y) in tqdm(enumerate(Y), units="samples", total=len(Y)):
            zip_data = zip(X, y)
            for x, y in zip_data:
                f.writelines(f"{x}\t{y}\n")
    # plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    # plt.show()
