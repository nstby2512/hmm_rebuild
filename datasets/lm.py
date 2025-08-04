import torch
import io

from torchtext import data
from torchtext.data.dataset import Dataset

def process_lines(path, encoding, text_field, newline_eos, fields):
    examples = []
    with io.open(path, encoding= encoding) as f:
        for i, line in enumerate(f):
            text = text_field.preprocess(line)
            if newline_eos:
                text.append('<eos>')
            example = data.Example.fromlist([text],fields)
            example.idx = i
            examples.append(example)
    return examples

def process_articles(path, encoding, text_field, newline_eos, fields):
    examples = []
    cur_example = []
    with io.open(path, encoding = encoding) as f:
        for i, line in enumerate(f):
            text = text_field.preprocess(line)
            if newline_eos:
                text.append('<eos>')
            cur_example.append(text)

            if cur_example[-1][0] == '=' and cur_example[-1][1] != '=':
                example = data.Example.fromlist([[
                    token for tokens in cur_example[:-1] for token in tokens
                ]],fields
                )
                example.idx = i
                examples.append(example)
                cur_example = [cur_example[-1]]
        example = data.Example.fromlist([[
            token for tokens in cur_example[:-1] for token in tokens
        ]], fields)
        example.idx = i
        examples.append(example)
    return examples[1:]


class LanguageModelingDataset(data.Dataset):
    def __init__(
            self, 
            path, 
            text_field, 
            newline_eos = True, 
            feature_path = None, 
            encoding = 'utf-8', 
            articles = False, 
            **kwargs):
        #定义数据集的处理方式并提供处理好的数据集
        fields = [('text', text_field)]
        process = process_articles if articles else process_lines
        examples = process(path, encoding, text_field, newline_eos, fields)

        #继承data.Dataset的init
        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)
        
class PennTreebank(LanguageModelingDataset):
    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    name = 'penn-treebank'
    dirname = ''

    @classmethod
    def splits(cls, text_field, root = '.data', train = 'ptb.train.txt', 
               validation = 'ptb.valid.txt', test = 'ptb.test.txt', **kwargs ):
        return super(PennTreebank, cls).splits(root = root, train = train, 
                validation = validation, test = test, text_field = text_field, 
                articles = False, **kwargs)
    
    @classmethod
    def iters(cls, batch_size = 32, bptt_len = 35, device = 0, 
              root = '.data', vectors = None, **kwargs):
        TEXT = data.Field()
        train, val, test = cls.splits(TEXT, root = root, **kwargs)
        TEXT.build_vocab(train, vectors=vectors)
        
        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)
    
class WikiText2(LanguageModelingDataset):

    urls = ['https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip']
    name = 'wikitext-2'
    dirname = 'wikitext-2'

    @classmethod
    def splits(cls, text_field, root='.data', train='wiki.train.tokens',
               validation='wiki.valid.tokens', test='wiki.test.tokens',
               **kwargs):
        return super(WikiText2, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, articles=True, **kwargs)
    
    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
                  vectors=None, **kwargs):
        TEXT = data.Field()
        train, val, test = cls.splits(TEXT, root=root, **kwargs)
        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)
                







    
