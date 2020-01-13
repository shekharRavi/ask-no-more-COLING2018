import io
import os
import gzip
import h5py
import json
import wget
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer

import utils

url = {
    'train': 'https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.train.jsonl.gz',
    'valid': 'https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.valid.jsonl.gz',
    'test': 'https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.test.jsonl.gz'
}

class GuessWhatDataset(Dataset):

    def __init__(self, split, data_dir, model, coco_dir=None, vgg_on_the_fly=False, h5File=None,
                 mapping_file=None, successful_only=True, min_occ=3, max_sequence_length=100,
                 max_question_length=20):

        assert split in url.keys()
        self.split = split
        self.successful_only = successful_only
        model = model.lower()
        assert model in ['qgen', 'guesser', 'oracle', 'dm1', 'dm2', 'inference', 'image_only']
        self.model = model

        self.vgg_on_the_fly = vgg_on_the_fly

        # create data directory and download data
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self.file = os.path.join(data_dir, 'guesswhat.%s.jsonl.gz'%self.split)

        if not os.path.exists(self.file):
            print("%s not found. Downloading..."%self.file)
            wget.download(url[split], out=self.file)
            print()

        # create / load vocablurary
        self.vocab_file = os.path.join(data_dir, 'vocab_%i.json'%(min_occ))
        if not os.path.exists(self.vocab_file):
            print("Vocablurary not found at %s. Creating new."%self.vocab_file)
            word_vocab = self._create_vocab(min_occ)
        else:
            word_vocab = self._load_vocab(self.vocab_file)

        self.w2i, self.i2w = word_vocab['w2i'], word_vocab['i2w']

        self.a2t = {
            'Yes': '<yes>',
            'No': '<no>',
            'N/A': '<n/a>',
        }

        # create dataset
        if self.model in ['qgen', 'dm1', 'dm2', 'inference', 'image_only']:

            if self.model == 'qgen':
                self.data = self._create_dataset_qgen(max_sequence_length)
            elif self.model == 'dm1':
                self.data, self.sample_weights = self._create_dataset_dm1(max_sequence_length)
            elif self.model == 'image_only':
                self.data = self._create_dataset_image_only()


            if self.vgg_on_the_fly:
                assert coco_dir is not None
                self.train_coco = os.path.join(coco_dir, 'train2014')
                self.valid_coco = os.path.join(coco_dir, 'val2014')

                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(( 0.485, 0.456, 0.406 ),
                                         ( 0.229, 0.224, 0.225 ))])
            else:
                assert h5File is not None and mapping_file is not None
                self.features = np.asarray(h5py.File(os.path.join(data_dir, h5File), 'r')['vgg_fc8'])
                self.mapping = json.load(open(os.path.join(data_dir, mapping_file)))

        if self.model in ['guesser', 'oracle', 'dm2', 'inference']:
            self.category_file = os.path.join(data_dir, 'category.json')
            if split == 'train' and not os.path.exists(self.category_file):
                print("Category-Vocablurary not found at %s. Creating new."%self.category_file)
                category_vocab = self._create_category_vocab()
            else:
                category_vocab = self._load_vocab(self.category_file)
            self.c2i, self.i2c = category_vocab['c2i'], category_vocab['i2c']

            if self.model == 'guesser':
                self.data = self._create_dataset_guesser(max_sequence_length)

            elif self.model == 'oracle':
                self.data = self._create_dataset_oracle(max_question_length)

            elif self.model == 'dm2':
                self.data, self.sample_weights = self._create_dataset_dm2(max_question_length)

            elif self.model == 'inference':
                self.data = self._create_dataset_inference()

        print("%s dataset with %i examples created."%(self.split.upper(), self.__len__()))

    @property
    def vocab_size(self):
        return len(self.w2i)
    @property
    def pad(self):
        return self.w2i['<pad>']
    @property
    def num_categories(self):
        return len(self.c2i)

    def __getitem__(self, idx):
        if self.model == 'qgen':
            return {
                'input': np.asarray(self.data[idx]['input']),
                'target': np.asarray(self.data[idx]['target']),
                'length': self.data[idx]['length'],
                'image': self._image_transforms(self.data[idx]['img_file_name'])
                }
        elif self.model == 'dm1':
            return {
                'input': np.asarray(self.data[idx]['input']),
                'length': self.data[idx]['length'],
                'image': self._image_transforms(self.data[idx]['img_file_name']),
                'decision_label': self.data[idx]['decision_label']
            }
        elif self.model == 'guesser':
            return {
                'input': np.asarray(self.data[idx]['input']),
                'length': self.data[idx]['length'],
                'target': self.data[idx]['target'],
                'categories': np.asarray(self.data[idx]['categories']),
                'bboxes': np.asarray(self.data[idx]['bboxes'], dtype=np.float32),
                'num_objects': self.data[idx]['num_objects']
            }
        elif self.model == 'dm2':
            return {
                'input': np.asarray(self.data[idx]['input']),
                'length': self.data[idx]['length'],
                'target': self.data[idx]['target'],
                'categories': np.asarray(self.data[idx]['categories']),
                'bboxes': np.asarray(self.data[idx]['bboxes'], dtype=np.float32),
                'num_objects': self.data[idx]['num_objects'],
                'image': self._image_transforms(self.data[idx]['img_file_name']),
            }
        elif self.model == 'oracle':
            return {
                'question': np.asarray(self.data[idx]['question']),
                'length': self.data[idx]['length'],
                'target_category': self.data[idx]['target_category'],
                'target_spatial': np.asarray(self.data[idx]['target_bounding_boxes'], dtype=np.float32),
                'answer': self.data[idx]['answer']
            }
        elif self.model == 'inference':
            return {
                'input': np.asarray(self.data[idx]['input']),
                'image': self._image_transforms(self.data[idx]['img_file_name']),
                'target': self.data[idx]['target'],
                'target_category': self.data[idx]['target_category'],
                'target_spatial': np.asarray(self.data[idx]['target_bounding_boxes'], dtype=np.float32),
                'categories': np.asarray(self.data[idx]['categories']),
                'bboxes': np.asarray(self.data[idx]['bboxes'], dtype=np.float32),
                'num_objects': self.data[idx]['num_objects']
            }
        elif self.model == 'image_only':
            return {
                'file_name': self.data[idx],
                'image': self._image_transforms(self.data[idx])
            }

    def __len__(self):
        return len(self.data)

    def _image_transforms(self, image_file_name):

        if self.vgg_on_the_fly:
            p = os.path.join(self.train_coco if 'train' in image_file_name else self.valid_coco, image_file_name)
            i = Image.open(p).convert('RGB')
            x = self.transform(i)
            return x
        else:
            i = self.mapping[image_file_name]
            return self.features[i]

    def _create_vocab(self, min_occ):
        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = utils.OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<eoq>', '<sos>', '<eos>', '<yes>', '<no>', '<n/a>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with gzip.open(self.file, 'r') as file:

            for json_game in file:
                game = json.loads(json_game.decode("utf-8"))

                if successful_only and game['status'] != 'success':
                    continue

                for qa in game['qas']:
                    words = tokenizer.tokenize(qa['question'])
                    w2c.update(words)

        for w, c in w2c.items():
            if c >= min_occ and w.count('.') <= 1:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(self.vocab_file, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        return self._load_vocab(self.vocab_file)

    def _load_vocab(self, file):
        with open(file, 'r', encoding='utf8') as vocab_file:
            vocab = json.load(vocab_file)

        return vocab

    def _create_category_vocab(self):

        assert self.split == 'train',  "Vocablurary can only be created for training file."

        c2i = {'<pad>': 0}
        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                for oi, obj in enumerate(game['objects']):
                    if obj['category'] not in c2i:
                        c2i[obj['category']] = len(c2i)

        i2c = {v:k for v,k in c2i.items()}
        vocab = dict(c2i=c2i, i2c=i2c)
        with io.open(self.category_file, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        return self._load_vocab(self.category_file)

    def _create_dataset_qgen(self, max_sequence_length, max_num_questions=25):

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only and game['status'] != 'success':
                    continue

                input = ['<sos>']
                target = list()
                for j, qa in enumerate(game['qas'], 1):
                    words = tokenizer.tokenize(qa['question'])
                    input += words + [self.a2t[qa['answer']]]
                    target += words + ['<eoq>']

                input = input[:max_sequence_length]

                target = target[:max_sequence_length-1]
                #target = target + ['<eos>']
                target = target + ['<pad>'] # putting pad here to not backprop through this

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (max_sequence_length-length))
                target.extend(['<pad>'] * (max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

                data[id]['img_file_name'] = game['image']['file_name']
                data[id]['img_flickr_url'] = game['image']['flickr_url']

                # if i == 99:
                #     break

        return data

    def _create_dataset_guesser(self, max_sequence_length, max_objects=20):

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)

        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only and game['status'] != 'success':
                    continue

                sequence = list()
                for qa in game['qas']:
                    words = tokenizer.tokenize(qa['question'])
                    sequence += words
                    sequence += [self.a2t[qa['answer']]]

                sequence = sequence[:max_sequence_length]
                length = len(sequence)

                sequence.extend(['<pad>'] * (max_sequence_length-length))
                sequence = [self.w2i.get(w, self.w2i['<unk>']) for w in sequence]

                object_categories = list()
                object_bounding_boxes = list()
                target_id = -1
                for oi, obj in enumerate(game['objects']):
                    object_categories.append(self.c2i[obj['category']])
                    object_bounding_boxes.append(self.bb2feature(bbox=obj['bbox'],
                                                            im_width=game['image']['width'],
                                                            im_height=game['image']['height'])
                                                            )
                    if obj['id'] == game['object_id']:
                        assert target_id == -1
                        target_id = oi
                assert target_id != -1

                num_objects = len(object_categories)
                object_categories.extend([0] * (max_objects-num_objects))
                object_bounding_boxes.extend([[0] * 8] * (max_objects-num_objects))

                id = len(data)
                data[id]['input'] = sequence
                data[id]['length'] = length
                data[id]['target'] = target_id
                data[id]['categories'] = object_categories
                data[id]['bboxes'] = object_bounding_boxes
                data[id]['num_objects'] = num_objects

                # if i > 99:
                #     break

        return data

    def _create_dataset_oracle(self, max_question_length):

        self.a2i = {
            'Yes': 0,
            'No': 1,
            'N/A': 2
        }

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only and game['status'] != 'success':
                    continue

                for oi, obj in enumerate(game['objects']):
                    if obj['id'] == game['object_id']:
                        target_category = self.c2i[obj['category']]
                        target_bounding_box = self.bb2feature(bbox=obj['bbox'],
                                                            im_width=game['image']['width'],
                                                            im_height=game['image']['height'])
                        break

                for qa in game['qas']:
                    question = tokenizer.tokenize(qa['question'])
                    question = question[:max_question_length]
                    length = len(question)
                    question.extend(['<pad>'] * (max_question_length-length))
                    question = [self.w2i.get(w, self.w2i['<unk>']) for w in question]

                    answer = self.a2i[qa['answer']]

                    id = len(data)
                    data[id]['question'] = question
                    data[id]['length'] = length
                    data[id]['target_category'] = target_category
                    data[id]['target_bounding_boxes'] = target_bounding_box
                    data[id]['answer'] = answer

                # if i == 99:
                #     break

        return data

    def _create_dataset_dm1(self, max_sequence_length):
        tokenizer = TweetTokenizer(preserve_case=False)

        sample_weights = list()
        data = defaultdict(dict)
        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only and game['status'] != 'success':
                    continue

                sequence = ['<sos>']
                for j, qa in enumerate(game['qas'], 1):
                    words = tokenizer.tokenize(qa['question'])
                    sequence += words
                    sequence += [self.a2t[qa['answer']]]

                    sequence = sequence[:max_sequence_length]

                    dialogue = sequence[:]
                    length = len(dialogue)
                    dialogue.extend(['<pad>'] * (max_sequence_length-length))
                    dialogue = [self.w2i.get(w, self.w2i['<unk>']) for w in dialogue]

                    id = len(data)
                    data[id]['input'] = dialogue
                    data[id]['length'] = length
                    data[id]['decision_label'] = 0 if j<len(game['qas']) else 1

                    data[id]['img_file_name'] = game['image']['file_name']
                    data[id]['img_flickr_url'] = game['image']['flickr_url']

                    sample_weights.append(0.2 if j<len(game['qas']) else 0.8)

                # if i == 99:
                #     break

        return data, sample_weights

    def _create_dataset_dm2(self, max_sequence_length, max_objects=20):
        tokenizer = TweetTokenizer(preserve_case=False)

        sample_weights = list()
        data = defaultdict(dict)
        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only and  game['status'] != 'success':
                    continue

                object_categories = list()
                object_bounding_boxes = list()
                target_id = -1
                for oi, obj in enumerate(game['objects']):
                    object_categories.append(self.c2i[obj['category']])
                    object_bounding_boxes.append(self.bb2feature(bbox=obj['bbox'],
                                                            im_width=game['image']['width'],
                                                            im_height=game['image']['height'])
                                                            )
                    if obj['id'] == game['object_id']:
                        assert target_id == -1
                        target_id = oi

                num_objects = len(object_categories)
                object_categories.extend([0] * (max_objects-num_objects))
                object_bounding_boxes.extend([[0] * 8] * (max_objects-num_objects))

                sequence = list()
                for j, qa in enumerate(game['qas'], 1):
                    words = tokenizer.tokenize(qa['question'])
                    sequence += words
                    sequence += [self.a2t[qa['answer']]]

                    sequence = sequence[:max_sequence_length]

                    dialogue = sequence[:]
                    length = len(dialogue)
                    dialogue.extend(['<pad>'] * (max_sequence_length-length))
                    dialogue = [self.w2i.get(w, self.w2i['<unk>']) for w in dialogue]

                    id = len(data)
                    data[id]['input'] = dialogue
                    data[id]['length'] = length

                    data[id]['target'] = target_id
                    data[id]['categories'] = object_categories
                    data[id]['bboxes'] = object_bounding_boxes
                    data[id]['num_objects'] = num_objects

                    data[id]['img_file_name'] = game['image']['file_name']
                    data[id]['img_flickr_url'] = game['image']['flickr_url']

                    sample_weights.append(0.2 if j<len(game['qas']) else 0.8)

                # if i == 99:
                #     break

        return data, sample_weights

    def _create_dataset_inference(self, max_objects=20):
        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only and  game['status'] != 'success':
                    continue

                object_categories = list()
                object_bounding_boxes = list()
                target_id = -1
                for oi, obj in enumerate(game['objects']):
                    object_categories.append(self.c2i[obj['category']])
                    object_bounding_boxes.append(self.bb2feature(bbox=obj['bbox'],
                                                            im_width=game['image']['width'],
                                                            im_height=game['image']['height'])
                                                            )
                    if obj['id'] == game['object_id']:
                        target_id = oi
                        target_category = object_categories[-1]
                        target_bounding_box = object_bounding_boxes[-1]

                num_objects = len(object_categories)
                object_categories.extend([0] * (max_objects-num_objects))
                object_bounding_boxes.extend([[0] * 8] * (max_objects-num_objects))

                id = len(data)
                data[id]['input'] = [self.w2i['<sos>']]
                data[id]['img_file_name'] = game['image']['file_name']
                data[id]['img_flickr_url'] = game['image']['flickr_url']

                data[id]['categories'] = object_categories
                data[id]['bboxes'] = object_bounding_boxes
                data[id]['num_objects'] = num_objects

                data[id]['target_category'] = target_category
                data[id]['target_bounding_boxes'] = target_bounding_box

                data[id]['target'] = target_id

                # if i == 1000:
                #     break

        return data

    def _create_dataset_image_only(self):

        images = set()
        with gzip.open(self.file, 'r') as file:

            for i, json_game in enumerate(file):
                game = json.loads(json_game.decode("utf-8"))

                images.add(game['image']['file_name'])

        data = list(images)

        return data



    def bb2feature(self, bbox, im_width, im_height):
        x_width = bbox[2]
        y_height = bbox[3]

        x_left = bbox[0]
        y_upper = bbox[1]
        x_right = x_left+x_width
        y_lower = y_upper+y_height

        x_center = x_left + 0.5*x_width
        y_center = y_upper + 0.5*y_height

        # Rescale features fom -1 to 1

        x_left = (1.*x_left / im_width) * 2 - 1
        x_right = (1.*x_right / im_width) * 2 - 1
        x_center = (1.*x_center / im_width) * 2 - 1

        y_lower = (1.*y_lower / im_height) * 2 - 1
        y_upper = (1.*y_upper / im_height) * 2 - 1
        y_center = (1.*y_center / im_height) * 2 - 1

        x_width = (1.*x_width / im_width) * 2
        y_height = (1.*y_height / im_height) * 2

        # Concatenate features
        feat = [x_left, y_upper, x_right, y_lower, x_center, y_center, x_width, y_height]

        return feat


if __name__ == "__main__":

    gw = GuessWhatDataset('train', 'data', 'guesser')
    print(gw[0])
