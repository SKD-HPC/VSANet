import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import h5py

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.symptom_prob_file = h5py.File('/public/home/huarong/yixiulong/RM/R2Gen-AAGT/data/mimic_cxr/symptom_pro.h5','r')
        self.symptom_file = h5py.File('/public/home/huarong/yixiulong/RM/R2Gen-AAGT/data/mimic_cxr/symptom_f.h5','r')

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir  
        self.ann_path = args.ann_path  
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer  
        self.transform = transform  
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]

        self.symptom_prob_file = h5py.File('/public/home/huarong/yixiulong/RM/R2Gen-AAGT/data/iu_xray/iu_symptom_pro.h5','r')
        self.symptom_file = h5py.File('/public/home/huarong/yixiulong/RM/R2Gen-AAGT/data/iu_xray/symptom_f.h5','r')

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        symptom_prob = torch.tensor(self.symptom_prob_file[image_id][()]).squeeze(0).unsqueeze(1)
        symptom_f = torch.tensor(self.symptom_file['feature_embedding'][()])
        symptom_feature = symptom_f*symptom_prob

        sample = (image_id, image, report_ids, report_masks, seq_length, symptom_feature)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        symptom_prob = torch.tensor(self.symptom_prob_file[image_id[-48:]][()]).squeeze(0).unsqueeze(1)
        symptom_f = torch.tensor(self.symptom_file['feature_embedding'][()])
        symptom_feature = symptom_f*symptom_prob
        sample = (image_id, image, report_ids, report_masks, seq_length, symptom_feature)
        return sample
