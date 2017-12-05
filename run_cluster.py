import argparse
from runner import *

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['train',
                                       'train-net',
                                       'train-rf'])
parser.add_argument('i', type=int, metavar='crossvalidation-i')
parser.add_argument('--net-data')
parser.add_argument('--epochs', type=int)

args = parser.parse_args()


def prepare_net(net):
    if args.epochs is not None:
        net.epochs = args.epochs
    if args.net_data is not None:
        net.data_dir_xy = args.net_data


runner = TimeSeriesRunner(net_prepare_func=prepare_net)
if args.action == 'train':
    runner.train_one(args.i)
elif args.action == 'train-net':
    runner.train_net(args.i)
elif args.action == 'train-rf':
    runner.train_rf(args.i)
else:
    raise Exception("Action %s not implemented!" % args.action)

# def test():
#       from net import Net
#       n = Net(cutoff='single', target='hindex_cumulative')
#       n.epochs = 1
#       n.suffix += '-cv-0'
#       n.suffix_author_ids += '-cv-0'
#       n.data_dir_xy = '/scratch/fias/mistele/'
#       n.train(live_validate=False)
#       n.train(live_validate=False)
#
#
# from memory_profiler import memory_usage
# mem = max(memory_usage(proc=test))
# print("Maximum memory used: {0} MiB".format(str(mem)))
# RESULT: Maximum memory used: 8165.67578125 MiB
