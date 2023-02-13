import pstats
import argparse

if __name__ == '__main__':
    sort_default = [-1]
    
    parser = argparse.ArgumentParser(description='Stats printer')
    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('-sort', nargs='+', choices=['cumulative', 'time', 'filename', 'calls'], help='Sort by', required=False, default=sort_default)
    parser.add_argument('-filter', type=str, help='Filter by', required=False, default=None)
    
    args = parser.parse_args()
    
    p = pstats.Stats(args.input_file)
    sort_by = None
    if args.sort != sort_default:
        options = {
            'cumulative': pstats.SortKey.CUMULATIVE,
            'time': pstats.SortKey.TIME,
            'filename': pstats.SortKey.FILENAME,
            'calls': pstats.SortKey.CALLS
        }
        sort_by = [options[x] for x in args.sort]
    else: sort_by = sort_default
    
    res = p.strip_dirs().sort_stats(*sort_by).print_stats(args.filter)
    print(res)