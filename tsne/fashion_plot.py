import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns


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
    parser = argparse.ArgumentParser(description="Plotter")
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument(
        "-labels",
        type=FileTypeWithExtensionCheck(valid_extensions=("txt", "data")),
        help="Labels file",
        required=False,
    )
    parser.add_argument(
        "-col", type=int, help="Column from label file", required=False, default=None
    )
    parser.add_argument("-step", type=int, help="Step", required=False, default=1)
    parser.add_argument(
        "-alpha", type=float, help="Alpha of point", required=False, default=1
    )

    args = parser.parse_args()

    data = None
    step = None

    with open(args.input_file, "r") as f:
        step = int(f.readline())
        data = np.loadtxt(f)

    dims = len(data.shape)

    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels)
        args.labels.close()

    fig, ax = plt.subplots()

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    classes = [i for i in range(10)]

    # plt.scatter(
    #     data[:, 0], data[:, 1], 20, alpha=args.alpha, marker="."
    # ) if labels is None else plt.scatter(
    #     data[:, 0],
    #     data[:, 1],
    #     20,
    #     labels[:len(data)],
    #     alpha=args.alpha,
    #     marker=".",
    # )

    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels[: len(data)], palette="Set2")
    plt.legend(classes)
    plt.show()
