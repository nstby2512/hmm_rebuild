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





    
