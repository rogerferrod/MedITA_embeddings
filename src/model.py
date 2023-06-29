import torch
import time
from torch import nn

class BERTFineTunedModel(nn.Module):
    def __init__(self, device, model, start_token_id, end_token_id, summary_method="CLS"):
        super(BERTFineTunedModel, self).__init__()
        
        self.device = device
        self.model = model
        self.summary_method = summary_method
        self.model.to(device)
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
    
    def get_context_embeddings(self, input_ids, attention_mask):
        start_indices = (input_ids == self.start_token_id).nonzero(as_tuple=True)[1]
        end_indices = (input_ids == self.end_token_id).nonzero(as_tuple=True)[1]
        
        mask = (input_ids!=self.start_token_id) & (input_ids!=self.end_token_id)
        
        input_ids = torch.masked_select(input_ids, mask).reshape(input_ids.shape[0], -1)
        attention_mask = torch.masked_select(attention_mask, mask).reshape(attention_mask.shape[0], -1)
        
        output = self.model(input_ids=input_ids[:], attention_mask=attention_mask[:])
        if self.summary_method == "CLS":
            embeddings = output.last_hidden_state[:,0,:]
        if self.summary_method == "mean":
            embeddings = [torch.mean(output.last_hidden_state[i, s:e-1, :], dim=0) for i, (s,e) in enumerate(zip(start_indices, end_indices))]
            embeddings = torch.stack(embeddings)
        if self.summary_method == "max":
            embeddings = [torch.max(output.last_hidden_state[i, s:e-1, :], dim=0)[0] for i, (s,e) in enumerate(zip(start_indices, end_indices))]
            embeddings = torch.stack(embeddings)
            
        del start_indices
        del end_indices
        del output
        torch.cuda.empty_cache()
        
        return embeddings
    
    def forward(self, input_ids, attention_mask):
        context_embeddings = self.get_context_embeddings(input_ids, attention_mask)
        return context_embeddings
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
