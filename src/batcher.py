import numpy as np
import random
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import faiss
from torch.utils.tensorboard import SummaryWriter


class Batcher:
    def __init__(self, data, tokenizer, index_generator, samples_per_cui=3, batch_size=16, positives=30, negatives=30, seed=42, writer=None):        
        self.data = data
        self.tokenizer = tokenizer
        self.index_generator = index_generator
        
        self.cui2idx = {cui: idx for idx, cui in enumerate(data.keys())}
        
        self.rnd = np.random.RandomState(seed)
        self.random = random.Random(seed)
        self.dataset = self.sample_contexts(self.data, samples_per_cui)
        
        self.num_items = len(self.dataset)
        self.indices = np.arange(self.num_items)
        self.batch_size = batch_size
        self.positives = positives
        self.negatives = negatives
        
        self.rnd.shuffle(self.indices)
        self.ptr = 0
        self.bi = 0
                
        if writer is None:
            self.writer = SummaryWriter(comment="finetune bert")
        else:
            self.writer = writer
    
    def __iter__(self):
        return self
    
    # Sample n representative contexts for each CUI
    def sample_contexts(self, data, n):
        dataset = []
        for cui in data:
            joined = []
            for syn in data[cui]:
                joined += [(cui, syn, x) for x in data[cui][syn]]
            s = min(len(joined), n)
            dataset.extend(self.random.sample(joined, s))
        return dataset
        
    # Sample k positives with priority on synonyms of syn
    def sample_positives(self, cui, syn, ctx, k):
        positives = []
        if len(self.data[cui]) > 1:
            positives.extend([x for syn1 in self.data[cui] for x in self.data[cui][syn1] if syn1 != syn])
        
        if len(positives) < k:
            missing = k-len(positives)
            if len(self.data[cui][syn]) > missing:
                positives.extend([x for x in self.random.sample(self.data[cui][syn], missing) if x != ctx])
            else:
                positives.extend([x for x in self.data[cui][syn] if x != ctx])
        elif len(positives) > k:
            self.rnd.shuffle(positives)
            positives = positives[:k]
        
        return positives, [cui]*len(positives)
    
    # Get top-m similar contexts
    def sample_possible_negatives(self, cui, syn, ctx, m):
        negatives = []
        labels = []
        idx = self.data[cui][syn].index(ctx)
        for n_cui, n_syn, n_idx in self.index_generator.search(cui, syn, idx):
            negatives.append(self.data[n_cui][n_syn][n_idx])
            labels.append(n_cui)
        return negatives, labels
    
    # Build a batch with k positives and m possible negatives for each entry of data
    def build_batch(self, data, k, m):
        batch = []
        labels = []
        
        
        batch_hard_negatives = 0
        for cui, syn, ctx in data:
            batch.append(ctx)
            labels.append(cui)
            
            positives, positive_labels = self.sample_positives(cui, syn, ctx, k)
            batch.extend(positives)
            
            labels.extend(positive_labels)
            
            negatives, negative_labels = self.sample_possible_negatives(cui, syn, ctx, m)
            for n, l in zip(negatives, negative_labels):
                if n not in batch:
                    batch.append(n)
                    labels.append(l)
            batch_hard_negatives += sum(n_cui != cui for n_cui in negative_labels)
            
        self.writer.add_scalar('avg_hard_negatives_in_batch', (batch_hard_negatives/len(data)), global_step=self.bi)
        
        max_length = max(len(x) for x in batch)+2
        tokens = [self.tokenizer.encode_plus(x, padding='max_length', max_length=max_length, return_tensors='pt', is_pretokenized=True,return_attention_mask=True) for x in batch]
        
        batch = torch.cat([token['input_ids'] for token in tokens])
        attention_masks = torch.cat([token['attention_mask'] for token in tokens])
        
        labels = [self.cui2idx[cui] for cui in labels]
        labels = torch.FloatTensor(labels)
        return batch, labels, attention_masks
        
    def __next__(self):
        if self.ptr >= self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration
            
        batch_indices = list(self.indices[self.ptr:self.ptr + self.batch_size])
        batch, labels, attention_mask = self.build_batch([self.dataset[int(i)] for i in batch_indices], self.positives, self.negatives)
        
        self.ptr += self.batch_size
        self.bi += 1

        return batch, labels, attention_mask
