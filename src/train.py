import os
import pickle
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from batcher import Batcher
from faiss_generator import Generator
from model import BERTFineTunedModel
from multi_similarity_loss import MultiSimilarityLoss, MultiSimilarityLossV2

import time

from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

def train(args, model, iterator, optimizer, criterion, scheduler, writer):
    step = args.step
    iteration = 0
    t_total = args.max_steps
        
    while True:
        model.train()
        epoch_iterator = tqdm(iterator, desc="Iteration "+str(iteration), ascii=True)
        batch_loss = 0.
        total_loss = 0.
        accumulated_loss = 0.
        
        for _, batch in enumerate(epoch_iterator):
            if step >= t_total:
                save_path = os.path.join(
                    args.output_dir, f'model_{step}')
                os.makedirs(save_path)
                model.save_pretrained(save_path)
                print("Saved model")
                return None
            
            input_ids = batch[0].to(args.device)
            attention_mask = batch[2].to(args.device)
            
            writer.add_scalar('batch_size', input_ids.size(0),
                              global_step=step)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).to(args.device)
            del input_ids
            del attention_mask
            
            labels = batch[1].to(args.device)
            loss = criterion(outputs, labels)
            
            batch_loss = float(loss.item())
            total_loss += batch_loss
            accumulated_loss += batch_loss
            
                        
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            epoch_iterator.set_description("Loss: %0.4f" % batch_loss)
            writer.add_scalar('batch_loss', batch_loss,
                              global_step=step)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.use_scheduler:
                    scheduler.step()
                grad = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(args.device) for p in model.parameters() if p.grad is not None]))
                writer.add_scalar('gradient', grad,
                              global_step=((step + 1)/args.gradient_accumulation_steps))
                writer.add_scalar('accumulated_loss', accumulated_loss/args.gradient_accumulation_steps,
                              global_step=((step + 1)/args.gradient_accumulation_steps))
                
                del grad
                accumulated_loss = 0
                optimizer.zero_grad()
                            
            
            lr = optimizer.param_groups[0]['lr']
            
            
            writer.add_scalar('learning_rate', lr,
                              global_step=step)
            
            
            del loss
            del outputs
            del labels
            torch.cuda.empty_cache()
                                  
            step += 1
            if step % args.save_step == 0 and step > 0:
                save_path = os.path.join(
                    args.output_dir, f'model_{step}')
                os.makedirs(save_path)
                model.save_pretrained(save_path)
                print("Saved model")
            if step % args.faiss_step == 0 and step > 0:
                print("Recalculating faiss")
                args.step = step
                return None
            
        print("Epoch processed, step:", step)
        iteration += 1

def run(args):
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)  
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    args.use_scheduler = args.use_scheduler == "True"
    args.preload_neighbours = args.preload_neighbours == "True"
    
    print("Loading contexts")
    with open(args.contexts, 'rb') as f:
        contexts = pickle.load(f)
        
    writer = SummaryWriter(comment="finetune bert")
    
    config = BertConfig.from_json_file(args.model_config_path)
    
    model = BertModel.from_pretrained(args.model, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    special_tokens_list = ['[M_s]', '[M_e]']
    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    special_tokens = {x: tokenizer.convert_tokens_to_ids([x])[0] for x in special_tokens_list}

    model = BERTFineTunedModel(args.device, model, special_tokens['[M_s]'], special_tokens['[M_e]'], summary_method=args.summary_method)
    
    indices = {}
    neighbours = None
    preloaded = False
    shift = 0
    
    if args.preload_neighbours:
        print("Preloading neighbours file")
        if os.path.exists(args.output_dir):
            neighbours = []
            for f in os.listdir(args.output_dir):
                if f[0:10] == "neighbours" and f[-4:] == ".pkl":
                    neighbours.append(int(f[11:-4]))
            if len(neighbours) > 0:
                shift = max(neighbours)
                print(f'Found {shift} file')
                with open(os.path.join(args.output_dir, 'neighbours_'+str(shift)+'.pkl'), 'rb') as f:
                    neighbours = pickle.load(f)
                with open(os.path.join(args.output_dir, 'indices_'+str(shift)+'.pkl'), 'rb') as f:
                    indices = pickle.load(f)
                preloaded = True
                    
    
    index_generator = Generator(model, tokenizer, indices=indices, neighbours=neighbours)
    training_generator = Batcher(contexts, tokenizer, index_generator, samples_per_cui=args.samples_per_cui, batch_size=args.train_batch_size, 
                                 positives=args.k, negatives=args.m, seed=args.seed, writer=writer
                                )
            
    t_total = args.max_steps
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                  lr = args.learning_rate,
                  eps = args.adam_epsilon
                )
    
    args.warmup_steps = int(args.warmup_steps)
    
    if t_total%args.gradient_accumulation_steps == 0:
        sched_steps = t_total/args.gradient_accumulation_steps
    else:
        sched_steps = t_total//args.gradient_accumulation_steps + 1
    
    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps//args.gradient_accumulation_steps, num_training_steps=sched_steps
        )
    if args.schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps//args.gradient_accumulation_steps
        )
    if args.schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps//args.gradient_accumulation_steps, num_training_steps=sched_steps
        )
    
    print("Resetting scheduler to ", shift//8)
    for i in range(shift//8):
        scheduler.step()
    
    if args.msloss == "v1":
        criterion = MultiSimilarityLoss(thresh=args.thresh, epsilon=args.epsilon, beta=args.beta, alpha=args.alpha, writer=writer)
    elif args.msloss == "v2":
        criterion = MultiSimilarityLossV2(thresh_p=args.thresh, thresh_n=args.thresh_n, epsilon=args.epsilon, beta=args.beta, alpha=args.alpha, writer=writer)
    args.step = shift
    i = int(shift/args.faiss_step)
    
    if preloaded==False:
        embeddings, indices, neighbours = index_generator(contexts, args.m)
            
        with open(os.path.join(args.output_dir, 'embeddings_0.pkl'), 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.output_dir, 'indices_0.pkl'), 'wb') as f:
                pickle.dump(indices, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.output_dir, 'neighbours_0.pkl'), 'wb') as f:
                pickle.dump(neighbours, f, protocol=pickle.HIGHEST_PROTOCOL)
        del embeddings
        torch.cuda.empty_cache()
    
    print("Starting at epoch ", i)
    print("Going for ", (np.ceil(args.max_steps / args.faiss_step)), " epochs")
    while i < (np.ceil(args.max_steps / args.faiss_step)):
        train(args, model, training_generator, optimizer, criterion, scheduler, writer)
        i+=1
        embeddings, indices, neighbours = index_generator(contexts, args.m)

        with open(os.path.join(args.output_dir, f'embeddings_{i}.pkl'), 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.output_dir, f'indices_{i}.pkl'), 'wb') as f:
                pickle.dump(indices, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.output_dir, f'neighbours_{i}.pkl'), 'wb') as f:
                pickle.dump(neighbours, f, protocol=pickle.HIGHEST_PROTOCOL)

        del embeddings
        torch.cuda.empty_cache()
            
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        default="",
        type=str,
        help="Model",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        type=str,
        help="Tokenizer",
    )
    parser.add_argument(
        "--contexts",
        default="../data/context_dictionary.pkl",
        type=str,
        help="Path to contexts",
    )
    parser.add_argument(
        "--output_dir",
        default="../output/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--faiss_step",
        default=75000,
        type=int,
        help="Faiss step",
    )
    parser.add_argument(
        "--save_step",
        default=25000,
        type=int,
        help="Save step",
    )
    parser.add_argument(
        "--k", default=30, type=int, help="How many contexts with same cui to sample per group.",
    )
    parser.add_argument(
        "--m", default=30, type=int, help="How many top similarity contexts to sample per group.",
    )
    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size of groups for training. A group contains 1+m+k contexts (current + top-m + k of same cui).",
    )
    parser.add_argument(
        "--samples_per_cui", default=3, type=str, help="How many contexts to sample for every CUI.",
    )
    
    parser.add_argument(
        "--embeddings", default=None, type=str, help="Path to file with precalculated embeddings.",
    )
    
    parser.add_argument("--preload_neighbours", type=str, default="False",
                        choices=["True", "False"], help="Preload last neighbours file.")                 
    
    parser.add_argument("--use_scheduler", type=str, default="True",
                        choices=["True", "False"], help="Use lr scheduler.") 

    parser.add_argument("--learning_rate", default=2e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--msloss", type=str, default="v1",
                        choices=["v1", "v2"], help="Version of the loss.") 
    parser.add_argument("--thresh", default=0.5,
                        type=float, help="MultiSimilarity Loss lambda.")
    parser.add_argument("--thresh_n", default=0.5,
                        type=float, help="MultiSimilarity Loss negative lambda for MS Loss v2.")
    parser.add_argument("--epsilon", default=0.1,
                        type=float, help="MultiSimilarity Loss epsilon.")
    parser.add_argument("--beta", default=50.,
                        type=float, help="MultiSimilarity Loss beta.")
    parser.add_argument("--alpha", default=2.,
                        type=float, help="MultiSimilarity Loss alpha.")
    parser.add_argument(
        "--max_steps",
        default=400000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    
    parser.add_argument("--gradient_accumulation_steps", default=8,
                        type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=40000,
                        help="Linear warmup over warmup_steps or a float.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--schedule", type=str, default="linear",
                        choices=["linear", "cosine", "constant"], help="Schedule.")
    parser.add_argument("--summary_method", type=str, default="mean",
                        choices=["CLS", "mean", "M_s"], help="Entity representation method.")
    
    args = parser.parse_args()
    args.model_config_path = os.path.join(args.model, 'config.json')
    
    run(args)
    
if __name__ == "__main__":
    main()
