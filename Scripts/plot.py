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
    parser.add_argument(
        "-neural_labels", action="store_true", help="Labels obtained from neural t-SNE"
    )
    parser.add_argument("-marker_size", type=int, help="Marker size", default=15)

    args = parser.parse_args()

    data = None
    step = None

    with open(args.input_file, "r") as f:
        step = int(f.readline())
        data = np.loadtxt(f)

    fig, ax = plt.subplots(1, 1)

    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels, usecols=args.col)
        args.labels.close()
        data = data[: len(labels)]

    plt.scatter(
        data[:: args.step, 0],
        data[:: args.step, 1],
        args.marker_size,
        alpha=args.alpha,
        marker=".",
    ) if labels is None else plt.scatter(
        data[:: args.step, 0],
        data[:: args.step, 1],
        args.marker_size,
        labels[:: step * args.step] if not args.neural_labels else labels[:: args.step],
        alpha=args.alpha,
        marker=".",
    )

    plt.ylabel("t-SNE 2")
    plt.xlabel("t-SNE 1")

    new_name = args.input_file.rsplit(".", 1)[0] + ".png"
    plt.savefig(new_name)
    plt.show()