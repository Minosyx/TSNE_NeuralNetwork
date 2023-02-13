import matplotlib.pyplot as plt
import numpy as np
import argparse


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
    parser.add_argument("-neural_labels", action="store_true", help="Neural labels")

    args = parser.parse_args()

    data = None
    step = None

    with open(args.input_file, "r") as f:
        step = int(f.readline())
        data = np.loadtxt(f)

    dims = len(data.shape)

    fig, ax = plt.subplots(1, 1)

    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels, usecols=args.col)
        args.labels.close()
        data = data[: len(labels)]

    plt.scatter(
        data[:: args.step, 0],
        data[:: args.step, 1],
        15,
        alpha=args.alpha,
        marker=".",
    ) if labels is None else plt.scatter(
        data[:: args.step, 0],
        data[:: args.step, 1],
        15,
        labels[:: step * args.step] if not args.neural_labels else labels[:: args.step],
        alpha=args.alpha,
        marker=".",
    )

    plt.ylabel("t-SNE 2")
    plt.xlabel("t-SNE 1")

    new_name = args.input_file.rsplit(".", 1)[0] + ".png"
    plt.savefig(new_name, dpi=500)
    plt.show()
