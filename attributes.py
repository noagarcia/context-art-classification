import os
import utils

# Load attribute vocabularies, which are found in directory `Data/`

def load_att_class(args_dict):

    type_file = os.path.join(args_dict.dir_data, args_dict.vocab_type)
    school_file = os.path.join(args_dict.dir_data, args_dict.vocab_school)
    time_file = os.path.join(args_dict.dir_data, args_dict.vocab_time)
    author_file = os.path.join(args_dict.dir_data, args_dict.vocab_author)

    assert os.path.isfile(type_file), 'File %s not found.' % type_file
    assert os.path.isfile(school_file), 'File %s not found.' % school_file
    assert os.path.isfile(time_file), 'File %s not found.' % time_file
    assert os.path.isfile(author_file), 'File %s not found.' % author_file

    type2idx = utils.load_obj(type_file)
    school2idx = utils.load_obj(school_file)
    time2idx = utils.load_obj(time_file)
    author2idx = utils.load_obj(author_file)

    return type2idx, school2idx, time2idx, author2idx
