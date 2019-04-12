import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='test', type=str, help='Mode (train | test)')
    parser.add_argument('--model', default='kgm', type=str, help='Model (mtl | kgm). mlt for multitask learning model. kgm for knowledge graph model.' )
    parser.add_argument('--att', default='time', type=str, help='Attribute classifier (type | school | time | author) (only kgm model).')

    # Directories
    parser.add_argument('--dir_data', default='Data')
    parser.add_argument('--dir_dataset', default='')
    parser.add_argument('--dir_images', default='Images/')
    parser.add_argument('--dir_model', default='Models/')

    # Files
    parser.add_argument('--csvtrain', default='semart_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='semart_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='semart_test.csv', help='Dataset test data file')
    parser.add_argument('--vocab_type', default='type2ind.pckl', help='Type classes file')
    parser.add_argument('--vocab_school', default='school2ind.pckl', help='Author classes file')
    parser.add_argument('--vocab_time', default='time2ind.pckl', help='Timeframe classes file')
    parser.add_argument('--vocab_author', default='author2ind.pckl', help='Author classes file')

    # Training opts
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--nepochs', default=300, type=int)

    # KGM model
    parser.add_argument('--graph_embs', default='semart-artgraph-node2vec.model')
    parser.add_argument('--lambda_c', default=0.9, type=float)
    parser.add_argument('--lambda_e', default=0.1, type=float)

    # Test
    parser.add_argument('--model_path', default='Models/best-kgm-time-model.pth.tar', type=str)
    parser.add_argument('--no_cuda', action='store_true')

    return parser