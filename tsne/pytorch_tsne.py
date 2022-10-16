import torch
from torch import nn
import math


def Hbeta(D: torch.Tensor, beta: float) -> tuple:
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p_job(data: tuple, tolerance: float, max_iterations: int = 50):
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
            if math.isinf(beta_max):
                beta *= 2
            else:
                beta = (beta + beta_max) / 2
        else:
            beta_max = beta
            if math.isinf(beta_min):
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
    logU = torch.log(torch.tensor(perplexity))

    sum_X = torch.sum(torch.square(X), 1)
    D = sum_X + (sum_X.reshape(-1, 1) - 2 * torch.matmul(X, X.T))

    idx = (1 - torch.eye(n)).type(torch.bool)
    D = D[idx].reshape(n, -1)

    P = torch.zeros((n, n))

    for i in range(n):
        P[i, idx[i]] = x2p_job((i, D[i], logU), tolerance)[1]
    return P


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(0, 0),  # TODO: insert correct dimensions
            nn.ReLU(),
            nn.Linear(0, 0),
            nn.ReLU(),
            nn.Linear(0, 0),
        )

    def forward(self):
        x = self.flatten()
        logits = self.linear_relu_stack()
        return logits


class ParametericTSNE:
    def __init__(
        self,
        loss_fn,
        perplexity: int,
        early_exaggeration: int,
        batch_size: int,
        early_exaggeration_epochs: int,
        early_exaggeration_value: float,
    ):
        self.__device = device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = NeuralNetwork().to(device)

        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.batch_size = batch_size
        self.early_exaggeration_epochs = early_exaggeration_epochs
        self.early_exaggeration_value = early_exaggeration_value
        self.loss_fn = loss_fn

    def save_model(self, filename: str):
        torch.save(self.model.state_dict(), filename)

    def read_model(self, filename: str):
        self.model.load_state_dict(torch.load(filename))
        
    
