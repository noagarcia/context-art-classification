import torch
import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from gensim.models import Word2Vec


class ArtDatasetKGM(data.Dataset):

    def __init__(self, args_dict, set, att2i, att_name, transform = None):
        """
        Args:
            args_dict: parameters dictionary
            set: 'train', 'val', 'test'
            att2i: attribute vocabulary
            att_name: either 'type', 'school', 'time', or 'author'
            transform: data transform
        """

        self.args_dict = args_dict
        self.set = set

        # Load Data + Graph Embeddings
        self.graphEmb = []
        if self.set == 'train':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
            self.graphEm = Word2Vec.load(os.path.join(args_dict.dir_data, args_dict.graph_embs))
        elif self.set == 'val':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
        elif self.set == 'test':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
        df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')

        self.imagefolder = os.path.join(args_dict.dir_dataset, args_dict.dir_images)
        self.transform = transform
        self.att2i = att2i
        self.imageurls = list(df['IMAGE_FILE'])
        if att_name == 'type':
            self.att = list(df['TYPE'])
        elif att_name == 'school':
            self.att = list(df['SCHOOL'])
        elif att_name == 'time':
            self.att = list(df['TIMEFRAME'])
        elif att_name == 'author':
            self.att = list(df['AUTHOR'])


    def __len__(self):
        return len(self.imageurls)


    def class_from_name(self, vocab, name):

        if vocab.has_key(name):
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']

        return idclass


    def __getitem__(self, index):
        """Returns data sample as a pair (image, description)."""

        # Load image & apply transformation
        imagepath = self.imagefolder + self.imageurls[index]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Attribute class
        idclass = self.class_from_name(self.att2i, self.att[index])

        # Graph embedding (only training samples)
        if self.set == 'train':
            graph_emb = self.graphEm.wv[self.imageurls[index]]
            graph_emb = torch.FloatTensor(graph_emb)
            return [image], [idclass, graph_emb]
        else:
            return [image], [idclass]