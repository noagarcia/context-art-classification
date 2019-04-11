import warnings
warnings.filterwarnings("ignore")

from params import get_parser
from train import run_train
from test import run_test


if __name__ == "__main__":

    # Load parameters
    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    assert args_dict.att in ['type', 'school', 'time', 'author'], \
        'Incorrect classifier. Please select type, school, time, or author.'

    if args_dict.model == 'mtl':
        args_dict.att = 'all'
    args_dict.name = '{}-{}'.format(args_dict.model, args_dict.att)

    opts = vars(args_dict)
    print('------------ Options -------------')
    for k, v in sorted(opts.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-----------------------------------')

    # Check mode and model are correct
    assert args_dict.mode in ['train', 'test'], 'Incorrect mode. Please select either train or test.'
    assert args_dict.model in ['mtl', 'kgm'], 'Incorrect model. Please select either mlt or kgm.'

    # Run process
    if args_dict.mode == 'train':
        run_train(args_dict)
    elif args_dict.mode == 'test':
        run_test(args_dict)
