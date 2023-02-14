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
    parser = argparse.ArgumentParser(
        description="Plotter for t-SNE results from MNIST datasets"
    )
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
    parser.add_argument("-f", action="store_true", help="Fashion MNIST")

    args = parser.parse_args()

    data = None
    step = None

    with open(args.input_file, "r") as f:
        step = int(f.readline())
        data = np.loadtxt(f)

    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels)
        args.labels.close()

    if args.f:
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
    else:
        classes = [i for i in range(10)]

    fig, ax = plt.subplots(1, 1)

    sns.scatterplot(
        x=data[:, 0], y=data[:, 1], hue=labels[: len(data)], palette="Paired"
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(classes)

    new_name = args.input_file.rsplit(".", 1)[0] + ".png"

    plt.savefig(new_name)
    plt.show()
