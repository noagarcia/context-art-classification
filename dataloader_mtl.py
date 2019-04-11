import torch.utils.data as data
import pandas as pd
import os

from PIL import Image


class ArtDatasetMTL(data.Dataset):

    def __init__(self, args_dict, set, att2i, transform = None):
        """
        Args:
            args_dict: parameters dictionary
            set: 'train', 'val', 'test'
            att2i: list of attribute vocabularies as [type2idx, school2idx, time2idx, author2idx]
            transform: data transform
        """

        self.args_dict = args_dict
        self.set = set

        # Load data
        if self.set == 'train':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
        elif self.set == 'val':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
        elif self.set == 'test':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
        df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')

        self.imagefolder = os.path.join(args_dict.dir_dataset, args_dict.dir_images)
        self.transform = transform
        self.type_vocab = att2i[0]
        self.school_vocab = att2i[1]
        self.time_vocab = att2i[2]
        self.author_vocab = att2i[3]

        self.imageurls = list(df['IMAGE_FILE'])
        self.type = list(df['TYPE'])
        self.school = list(df['SCHOOL'])
        self.time = list(df['TIMEFRAME'])
        self.author = list(df['AUTHOR'])


    def __len__(self):
        return len(self.imageurls)


    def class_from_name(self, vocab, name):

        if vocab.has_key(name):
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']

        return idclass


    def __getitem__(self, index):

        # Load image & apply transformation
        imagepath = self.imagefolder + self.imageurls[index]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Attribute class
        type_idclass = self.class_from_name(self.type_vocab, self.type[index])
        school_idclass = self.class_from_name(self.school_vocab, self.school[index])
        time_idclass = self.class_from_name(self.time_vocab, self.time[index])
        author_idclass = self.class_from_name(self.author_vocab, self.author[index])

        return [image], [type_idclass, school_idclass, time_idclass, author_idclass]
