## Context Embeddings for Art Classification

Pytorch code for the classification part of our ICMR 2019 paper [Context-Aware Embeddings for Automatic Art Analysis](https://arxiv.org/abs/1904.04985). For the retrieval part, check [this other repository](https://github.com/noagarcia/context-art-retrieval). 


### Setup

1. Download dataset from [here](http://noagarciad.com/SemArt/).

2. Clone the repository: 
    
    `git clone https://github.com/noagarcia/context-art-classification.git`

3. Install dependencies:
    - Python 2.7
    - pytorch (`conda install pytorch=0.4.1 cuda90 -c pytorch`) 
    - torchvision (`conda install torchvision`)
    - visdom (check tutorial [here](https://github.com/noagarcia/visdom-tutorial))
    - pandas (`conda install -c anaconda pandas`)
    - gensim (`conda install -c anaconda gensim`)

4. For the KGM model, download the pre-computed graph embeddings from [here](http://noagarciad.com/data/ICMR2019/semart-artgraph-node2vec.model), and save the file into the `Data/` directory.

### Train

- To train MTL multi-classifier run:
    
    `python main.py --mode train --model mtl --dir_dataset $semart`
    
- To train KGM classifier run:
    
    `python main.py --mode train --model kgm --att $attribute --dir_dataset $semart`

Where `$semart` is the path to SemArt dataset and `$attribute` is the classifier type (i.e. `type`, `school`, `time`, or `author`).

### Test

- To test MTL multi-classifier run:
    
    `python main.py --mode test --model mtl --dir_dataset $semart`
    
- To test KGM classifier run:
    
    `python main.py --mode test --model kgm --att $attribute --dir_dataset $semart --model_path $model-file`

Where `$semart` is the path to SemArt dataset, `$attribute` is the classifier type (i.e. `type`, `school`, `time`, or `author`), and `$model-file` is the path to the trained model.

You can download our pre-trained models from:
- [MTL](http://noagarciad.com/data/ICMR2019/best-mtl-model.pth.tar)
- [KGM Type](http://noagarciad.com/data/ICMR2019/best-kgm-type-model.pth.tar)
- [KGM School](http://noagarciad.com/data/ICMR2019/best-kgm-school-model.pth.tar)
- [KGM Timeframe](http://noagarciad.com/data/ICMR2019/best-kgm-time-model.pth.tar)
- [KGM Author](http://noagarciad.com/data/ICMR2019/best-kgm-author-model.pth.tar)

### Results
 
Classification results on SemArt:

| Model        | Type           | School  |    Timeframe    | Author |
| ------------- |:-------------:| -----:|---------:|--------:|
| VGG16 pre-trained | 0.706 | 0.502 | 0.418 | 0.482 |
| ResNet50 pre-trained | 0.726 | 0.557 | 0.456 | 0.500 | 
| ResNet152 pre-trained | 0.740 | 0.540 | 0.454 | 0.489 |
| VGG16 fine-tuned | 0.768 | 0.616 | 0.559 | 0.520 |
| ResNet50 fine-tuned | 0.765 | 0.655 | 0.604 | 0.515 |
| ResNet152 fine-tuned | 0.790 | 0.653 | 0.598 | 0.573 |
| ResNet50+Attributes | 0.785 | 0.667 | 0.599 | 0.561 |
| ResNet50+Captions | 0.799 | 0.649 | 0.598 | 0.607 |
| MTL context-aware | 0.791 | **0.691** | **0.632** | 0.603 |
| KGM context-aware | **0.815** | 0.671 | 0.613 | **0.615** |  


### Examples

Paintings with the highest scores for each class:

![example](https://github.com/noagarcia/context-art-classification/blob/master/examples/examples_type.png?raw=true
)
![example](https://github.com/noagarcia/context-art-classification/blob/master/examples/examples_school.png?raw=true
)

### Citation


```
@InProceedings{Garcia2017Context,
   author    = {Noa Garcia and Benjamin Renoust and Yuta Nakashima},
   title     = {Context-Aware Embeddings for Automatic Art Analysis},
   booktitle = {Proceedings of the ACM International Conference on Multimedia Retrieval},
   year      = {2019},
}
``` 

[1]: http://researchdata.aston.ac.uk/380/
[2]: https://github.com/facebookresearch/visdom
