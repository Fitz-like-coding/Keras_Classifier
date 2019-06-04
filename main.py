#! -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import configparser
import json
import numpy as np
from nltk import FreqDist
from nltk.util import ngrams
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from TrainModel import train_model, test_model, load_model
from tqdm import tqdm

class auto_classifier:
    def __init__(self, hyperparameter, datapath):
        self.HYPER_PARAMETER = hyperparameter
        self.DATA_PATH = datapath

        self.MAX_SEQUENCE_LENGTH = int(hyperparameter['max_length'])
        self.MAX_WORDS = int(hyperparameter['max_words'])
        self.MODEL_TYPE = str(hyperparameter['model'])
        self.N_GRAM = int(hyperparameter['ngram'])
        self.TRAIN = int(hyperparameter['train'])

        self.TRAIN_PATH = str(datapath['train'])
        self.DEV_PATH = str(datapath['dev'])
        self.TEST_PATH = str(datapath['test'])
        self.CONFIG_DIR = str(datapath['config'])
        self.MODEL_PATH = str(datapath['model'])

    def _tokenization(self, text):
        return ' '.join([w for w in text])

    def _create_record(self, line, set_type):
        sample = line.strip().split()
        if set_type == 'train':
            content = ''.join(sample[:-1])
            label = sample[-1]
        elif set_type == 'test':
            content = sample[0]
            label = 0
        return {'content': self._tokenization(content), 'label': label}

    def _read_text(self, data_dir, set_type):
        records = []
        with open(data_dir, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                if line.strip() != '':
                    records.append(self._create_record(line, set_type))
        return records

    def _pre_treate(self, records=None):
        MAX_SEQUENCE_LENGTH = self.MAX_SEQUENCE_LENGTH
        MAX_WORDS = self.MAX_WORDS
        MODEL_TYPE = self.MODEL_TYPE
        N_GRAM = self.N_GRAM
        CONFIG_PATH = self.CONFIG_DIR

        if os.path.exists(os.path.join(CONFIG_PATH,'classifier_config.json')):
            word2id, id2label, label2id, class_weight, parameter = json.load(
                open(os.path.join(CONFIG_PATH, 'classifier_config.json'), 'r', encoding='utf-8')
            )
            if parameter['model'] != MODEL_TYPE or parameter['max_length'] != MAX_SEQUENCE_LENGTH:
                raise Exception("classifier_config error: inconsistent model type or sequence length, "
                                "please delete config.json and rerun the program")
            class_weight = {int(index): weight for index, weight in class_weight.items()}
        else:
            try:
                x = list(map(lambda x: x['content', records]))
                y = list(map(lambda x: x['label', records]))

                # create word2id dict
                if MODEL_TYPE != 'FastText':
                    N_GRAM = 1
                vectorizer = CountVectorizer(token_pattern = '[^\s]+', ngram_range=(1,N_GRAM), max_df=0.95, min_df=3, max_feature=MAX_WORDS)
                vectorizer.fit(x)
                word2id = {word: str(index)+2 for word, index in vectorizer.vocabulary_.items()}

                # create id2label and label2id dict
                Binarizer = LabelEncoder()
                Binarizer.fit(x)
                id2label = {index: label for index, label in enumerate(Binarizer.classes_)}
                label2id = {label: index for index, label in enumerate(Binarizer.classes_)}

                # create class_weight dict
                label_freq = FreqDist(y)
                class_weight = {int(index): max(label_freq.values())/label_freq[label]
                                for index, label in enumerate(Binarizer.classes_)}

                # strre model parameter
                parameter = {'model': MODEL_TYPE, 'max_length': MAX_SEQUENCE_LENGTH, 'ngram': N_GRAM}

                # dump config file
                json.dump([word2id, id2label, label2id, class_weight, parameter],
                          open(os.path.join(CONFIG_PATH, 'classifier_config.json'), 'w', encoding='utf-8'))
            except Exception as e:
                sys.exit('Error: config file not exist')
        return word2id, id2label, label2id, class_weight, parameter

    # pad ngras behind text
    def _pad_ngrams(self, text_list, ngram):
        new_text_list = []
        print ('Average data sequence length: {}'.format(
            np.mean(list(map(lambda x:len(x.split()),text_list)), dtype=int)
        ))
        for content in text_list:
            new_content = []
            for n in range(ngrams):
                new_content.extend([' '.join(p) for p in ngrams(content.split(), n+1)])
            new_text_list.append(new_content)
        print ('After Ngram padding average data sequence length: {}'.format(
            np.mean(list(map(lambda x:len(x.split()), new_text_list)), dtype=int)
        ))
        return new_text_list

    # convert text to sequence
    def _text_to_sequence(self, word2id, text_list, ngram, max_length):
        vobs = word2id.keys()
        new_text_list = self._pad_ngrams(text_list, ngram)
        sequence_list = [[word2id.get(w, 1) for w in content if w in vobs] for content in new_text_list]
        sequence_list = pad_sequences(sequence_list, maxlen = max_length)
        return sequence_list

    # convert label to onehot format
    def _label_to_onehot(self, label_list, label2id, set_type):
        print ('Label size', len(set(label_list)))
        if set_type == 'train':
            encoder = OneHotEncoder(categories='auto')
            label_list = [[label2id[label]] for label in label_list]
            encoder.fit([[k] for k in label2id.values()])
            label_list = encoder.transform(label_list).toarray()
        elif set_type == 'test':
            label_list = None
        return label_list

    # create model input from records
    def _create_input(self, records, word2id, label2id, parameter, set_type):
        MAX_LENGTH = int(parameter['max_length'])
        MODEL_TYPE = parameter['model']
        NGRAM = int(parameter['ngram'])
        if MODEL_TYPE != 'FastText':
            NGRAM = 1

        x = list(map(lambda x: x['content'], records))
        y = list(map(lambda x: x['label'], records))
        x = self._text_to_sequence(word2id, x, NGRAM, MAX_LENGTH)
        y = self._label_to_onehot(y, label2id, set_type)
        return x, y

    def _train(self, train_path, dev_path):
        train = self._read_text(train_path, 'train')
        dev = self._read_text(dev_path, 'train')

        word2id, id2label, label2id, class_weight, parameter = self._pre_treate(train)
        classNum = len(label2id.keys())
        x_train, y_train = self._create_input(train, word2id, label2id, parameter, 'train')
        x_dev, y_dev = self._create_input(dev, word2id, label2id, parameter, 'train')

        train_model(self.HYPER_PARAMETER, self.DATA_PATH,
                    x_train, y_train,
                    x_dev, y_dev,
                    classNum, class_weight)

    def _predict(self, test_path, model_path):
        model = load_model(model_path)
        test = self._read_text(test_path, 'test')
        word2id, id2label, label2id, class_weight, parameter = self._pre_treate()
        x_test, y_test = self._create_input(test, word2id, label2id, parameter, 'test')
        res = test_model(model, x_test, id2label)
        return res

    def run(self):
        if self.TRAIN == 1:
            self._train(self.TRAIN_PATH, self.DEV_PATH)
        elif self.TRAIN == 0:
            res = self._predict(self.TEST_PATH, self.MODEL_PATH)
            with open(os.path.join(self.CONFIG_DIR, 'result.txt'), 'w', encoding='utf-8') as file:
                for i in res:
                    file.write(i+'\n')

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read('./config/config.ini')
    hyperparameter = {'max_lanrth': int(conf.get('hyperparameter', 'max_length')),
                      'max_words': int(conf.get('hyperparameter', 'max_words')),
                      'emb': int(conf.get('hyperparameter', 'emb')),
                      'epochs': int(conf.get('hyperparameter', 'batch')),
                      'ngram': int(conf.get('hyperparameter','ngram')),
                      'model': str(conf.get('hyperparameter','model'))}

    datapath = {'train': str(conf.get('data', 'train')),
                'dev': str(conf.get('data','dev')),
                'test': str(conf.get('data', 'test')),
                'config': str(conf.get('data','config')),
                'model':str(conf.get('data','model'))}

    myclassifier = auto_classifier(hyperparameter, datapath)
    myclassifier.run()

