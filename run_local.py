import argparse
from runner import *

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['prepare',
                                       'evaluate',
                                       'evaluate-linear-naive',
                                       'evaluate-net',
                                       'evaluate-rf',
                                       'summarize',
                                       'summarize-net',
                                       'summarize-rf'])
parser.add_argument('--net-data')
parser.add_argument('--ignore-max-hindex', action='store_false')
parser.add_argument('--i', type=int, metavar='crossvalidation-i')

args = parser.parse_args()
if args.action.startswith('evaluate') and args.i is None:
    parser.error('"prepare" required --i')
elif not args.action.startswith('evaluate') and args.i is not None:
    parser.error('--i only possible for "prepare"')


def prepare_net(net):
    net.ignore_max_hindex_before = args.ignore_max_hindex
    if args.net_data is not None:
        net.data_dir_xy = args.net_data


runner = TimeSeriesRunner(net_prepare_func=prepare_net)

if args.action == 'prepare':
    runner.prepare()
elif args.action == 'evaluate':
    runner.evaluate_one(args.i)
elif args.action == 'evaluate-net':
    runner.evaluate_net(args.i)
elif args.action == 'evaluate-rf':
    runner.evaluate_rf(args.i)
elif args.action == 'evaluate-linear-naive':
    runner.evaluate_linear_naive(args.i)
elif args.action == 'summarize':
    runner.summarize()
elif args.action == 'summarize-net':
    runner.summarize_net()
elif args.action == 'summarize-rf':
    runner.summarize_rf()
else:
    raise Exception("Action %s not implemented!" % args.action)
