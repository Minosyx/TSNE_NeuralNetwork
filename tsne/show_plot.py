import matplotlib.pyplot as plt
import numpy as np
import argparse

class FileTypeWithExtensionCheck(argparse.FileType):
    def __init__(self, mode='r', valid_extensions=None, **kwargs):
        super().__init__(mode, **kwargs)
        self.valid_extensions = valid_extensions

    def __call__(self, string):
        if self.valid_extensions:
            if not string.endswith(self.valid_extensions):
                raise argparse.ArgumentTypeError(
                    'Not a valid filename extension!')
        return super().__call__(string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotter')
    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('-labels', type=FileTypeWithExtensionCheck(
        valid_extensions=('txt', 'data')), help='Labels file', required=False)
    parser.add_argument('-col', type=int, help='Column from label file', required=False, default=None)
    parser.add_argument('-step', type=int, help='Step', required=False, default=1)
    parser.add_argument('-alpha', type=float, help='Alpha of point', required=False, default=1)
    
    args = parser.parse_args()
    
    data = None
    step = None
    
    with open(args.input_file, 'r') as f:
        step = int(f.readline())
        data = np.loadtxt(f)

    dims = len(data.shape)
    
    labels = None
    if args.labels:
        labels = np.loadtxt( args.labels, usecols=args.col)
        args.labels.close()
    
    plt.scatter(data[::args.step, 0], data[::args.step, 1], 20, alpha=args.alpha, marker='.') if labels is None else plt.scatter(data[::args.step, 0], data[::args.step, 1], 20, labels[::step * args.step], alpha=args.alpha, marker='.')
    plt.show()