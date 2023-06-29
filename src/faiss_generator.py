import torch
import faiss

from tqdm import tqdm
import time
def get_embeddings(model, tokenizer, data):
    model.eval()
    batch_size = 3072

    max_length = max(len(x) for cui in data for syn in data[cui] for x in data[cui][syn])+2
    indices = {}
    global_ctx_index = 0
    all_contexts = []
    all_attention_masks = []
    for cui in tqdm(data, desc="Tokenizing"):
        for syn in data[cui]:
            syn_ctx_index = 0
            for x in data[cui][syn]:
                indices[global_ctx_index] = (cui, syn, syn_ctx_index)
                tokens = tokenizer.encode_plus(x, 
                                               padding='max_length', 
                                               max_length=max_length, 
                                               return_tensors='pt', 
                                               is_pretokenized=True,
                                               return_attention_mask=True
                                              )

                all_contexts.extend(tokens['input_ids'])
                all_attention_masks.extend(tokens['attention_mask'])
                global_ctx_index += 1
                syn_ctx_index += 1
    embeddings = torch.empty([0, 768]).cpu()
    with torch.no_grad():
        for i in tqdm(range((len(all_contexts)//batch_size)+1), desc="Calculating embeddings"):
            input_ids = torch.stack(all_contexts[i*batch_size:i*batch_size+batch_size]).to(model.device)
            attention_mask = torch.stack(all_attention_masks[i*batch_size:i*batch_size+batch_size]).to(model.device)

            embeddings = torch.cat([embeddings, model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()], 0)
            
            del input_ids
            del attention_mask
            torch.cuda.empty_cache()
            
    return embeddings, indices

class Generator:
    def __init__(self, model, tokenizer, indices={}, neighbours=None):
        self.model = model
        self.tokenizer = tokenizer
        
        self.indices = indices
        self.inverse_indices = {v: k for k, v in indices.items()}
        self.neighbours = neighbours

    def __call__(self, data, n):
        embeddings, self.indices = get_embeddings(self.model, self.tokenizer, data)
        self.inverse_indices = {v: k for k, v in self.indices.items()}
        
        d = embeddings.shape[1]
        res = faiss.StandardGpuResources()
        
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        
        e_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        e_norm = torch.div(embeddings, e_norm)
        embeddings = e_norm 
        
        gpu_index.add(embeddings.numpy())
        
        _, self.neighbours = gpu_index.search(embeddings.numpy(), n)
        
        del gpu_index
        del index
        del res
        torch.cuda.empty_cache()
        
        return embeddings, self.indices, self.neighbours
    
    def search(self, cui, syn, idx):
        i = self.inverse_indices[(cui, syn, idx)]
        
        answer = []
        for e in self.neighbours[i]:
            answer.append(self.indices[e])
        
        return answer
