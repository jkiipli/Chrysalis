import time
import math
import copy
import torch
import torch.nn as nn
import tblock as tb
from data import ListDataset

#training modes (affecting only base model, not derivative models)
TRAINMODE_ALL                  = 0 #cross-entropy, affects everything
TRAINMODE_ALL_EXCEPT_EMBEDS_CE = 1 #cross-entropy, affects intermediate results and out-head
TRAINMODE_ALL_EXCEPT_EMBEDS_MIX= 2 #cross-entropy on out-head, cross-entropy*vector-similarity on middle layers, affects intermediate results and out-head
TRAINMODE_ONLY_MIDDLE          = 3 #vector similarity, affects only intermediate results
TRAINMODE_ONLY_OUTHEAD         = 4 #cross-entropy, affects only out-head

#cross entropy loss, applicable only to certain kind of tensors, like the output and target of basic gpt model
#in particular, e1 needs to be logits of shape (batches, context_length, vocab_size)
#and e2 needs to be index into vocabulary with shape (batches, context_length, 1)
def cross_entropy(e1, e2):
    return torch.nn.functional.cross_entropy(e1.flatten(0, 1), e2.flatten())
        
def mse_loss(e1, e2): #substitute for nn.MSELoss(), as the latter causes strange problems if used in conjunction with torch.vmap (EmbBuf.get_logits)
    return torch.mean((e1-e2)**2)

batched_dot = [torch.dot, torch.vmap(torch.dot), torch.vmap(torch.vmap(torch.dot))]
def cos_loss(e1, e2, e1_norm=None, e2_norm=None):
    dim = len(e1.shape)-1
    if e1_norm is None:
        e1_norm = torch.nn.functional.normalize(e1, dim=dim)
    if e2_norm is None:
        e2_norm = torch.nn.functional.normalize(e2, dim=dim)
    return 1.0-torch.mean(batched_dot[dim](e1_norm, e2_norm))

#loss function used for vectors (tensors)
def mse_cos_loss(e1, e2, e1_norm=None, e2_norm=None):
    return cos_loss(e1, e2, e1_norm, e2_norm)*mse_loss(e1,e2)

vector_lossfunc = mse_cos_loss

#embedding vectors buffer containing both raw and normalized vectors
#with shape (vocab_size, embedding_size)
class EmbBuf():
    def __init__(self, emb_raw):
        self.emb_raw = emb_raw
        self.emb_norm = torch.nn.functional.normalize(emb_raw, dim=1)
        
    def match_embed_dot(self, emb, is_normalized=False, return_normalized=False):
        with torch.no_grad():
            if len(emb.shape) != 2: raise Exception("invalid shape")
            if not is_normalized: emb = torch.nn.functional.normalize(emb, dim=1)
            _,i = torch.topk(emb @ self.emb_norm.T, 1, dim=1, largest=True)
            i = i.squeeze(1)
            emb_buf = self.emb_raw
            if return_normalized: emb_buf = self.emb_norm
            return emb_buf[i], i
        
    #compare all possible embeddings and find closest match to given embedding using given loss function
    #given emb vector has shape (batch_size, embedding_size), the returned result will have the same shape
    def match_embed(self, emb, is_normalized=False, return_normalized=False, lossf=vector_lossfunc):
        logits = self.get_logits(emb, is_normalized=is_normalized, lossf=lossf)
        i = torch.argmax(logits, dim=-1)
        emb_buf = self.emb_raw
        if return_normalized: emb_buf = self.emb_norm
        return emb_buf[i], i
    
    #slow like mud, use get_logits instead, which acts equivalently
    def get_logits_iter(self, emb, is_normalized=False, lossf=vector_lossfunc):
        with torch.no_grad():
            batch_size, _ = emb.shape
            ret = []
            for b in range(batch_size):
                logits = []
                for i in range(len(self.emb_raw)):
                    e2_norm = None
                    if is_normalized: e2_norm = emb[b]
                    loss = lossf(self.emb_raw[i], emb[b], e1_norm=self.emb_norm[i], e2_norm=e2_norm)
                    logits.append(1.0/max(0.000001, loss.item()))
                ret.append(torch.tensor(logits))
            return torch.stack(ret, dim=0)
    
    #emb has shape (batch_size, embedding_size)
    #return value has shape (batch_size, vocab_size)
    def get_logits(self, emb, is_normalized=False, lossf=vector_lossfunc):
        with torch.no_grad():
            v_inner = torch.vmap(lambda e1, e2: 1.0/torch.clamp(lossf(e1, e2), min=0.000001), in_dims=(None, 0))
            v_outer = torch.vmap(v_inner, in_dims=(0, None))
            ret = v_outer(emb, self.emb_raw)
            '''
            dbg = self.get_logits_iter(emb, is_normalized, lossf).to(ret.device)
            print(f"{torch.min(ret-dbg)} {torch.max(ret-dbg)}")
            '''
            return ret
   
#wrapper around pytorch sequential module to allow recording of intermediate results
class RecordingSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        seq_record=[]
        for s in self:
            x = s(x)
            seq_record.append(x)
        return seq_record

#ai model with capability of recording intermediate results    
class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = None
        if self.has_embed(): self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = RecordingSequential(*[tb.TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        if self.has_logits_outhead():
            self.final_norm = tb.LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, inp, seq_record=None):
        shape = inp.shape
        seq_len = shape[1]
        if self.has_embed():
            x = self.tok_emb(inp)
        else:
            x = inp
        if seq_record is not None: seq_record.append(x) #without positional info
        if (seq_len != self.cfg['context_length']):
            print(f"{seq_len} {self.cfg['context_length']}")
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=inp.device))
        x = x + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        xx = self.trf_blocks(x)
        if seq_record is not None: seq_record.extend(xx)
        x = xx[-1]
        if self.has_logits_outhead():
            x = self.final_norm(x)
            x = self.out_head(x)
        return x
    
    def get_device(self):
        return next(self.parameters()).device
    
    def get_config(self):
        return self.cfg
    
    #whether the model has embeddings layer and takes in token id-s (True) or takes in already prepared vectors from embedding space (False)
    def has_embed(self):
        return self.cfg["tokenizer"] is not None
    
    def get_embed_size(self):
        return self.cfg["emb_dim"]
    
    #whether the model has output head to convert from embeddings space to logits of vocab size (True) or the model outputs just raw embedding vectors (False)
    def has_logits_outhead(self):
        return self.cfg["logits"]
    
    def get_num_tr_blocks(self):
        return self.cfg["n_layers"]
    
    def get_context_length(self):
        return self.cfg["context_length"]
    
    def set_train_embeds(self, train):
        if self.tok_emb is not None: self.tok_emb.weight.requires_grad = train
        
    def is_train_embeds(self):
        return self.tok_emb is not None and self.tok_emb.weight.requires_grad
        
    #create embedding buffer after training the model for subsequent use where needed
    def create_emb_buf(self, tokenizer, device):
        if not self.has_embed(): raise Exception("model has no embeddings")
        with torch.no_grad():
            emb_buf = []
            for i in range(self.tok_emb.num_embeddings):
                if tokenizer.convert_ids_to_tokens(i) is None: break#may happen to be None if vocabulary size is set to be too big
                t = self.tok_emb(torch.tensor(i).to(device))
                emb_buf.append(t.to(device))
            return EmbBuf(torch.stack(emb_buf, dim=0).to(device))
    
    #output logits of the last token
    #if the model uses logits outhead then the out is already in logits form
    #otherwise the vector is compared to all the vocabulary embeddings and logits are constructed based on given loss function
    def last_result_to_logits(self, out, emb_buf, lossf=vector_lossfunc):
        out = out[:, -1, :]
        if self.has_logits_outhead(): return out
        return emb_buf.get_logits(out, lossf)
    
    #project output corresponding to the last token to token embedding
    #if the model uses logits outhead then the embedding is found using argmax of the logits vector
    #otherwise the output depends wheter match is True or False
    #if match is True then the vector is compared to given vocabulary embeddings and the best match will be selected
    #if match is False then the raw vector itself will be returned
    def last_result_to_embed(self, out_batch, emb_buf, match=True, calc_inx=True, lossf=vector_lossfunc):
        if self.has_logits_outhead():
            with torch.no_grad():
                inx = torch.argmax(out_batch[:,-1,:], dim=-1, keepdim=True)
                return self.tok_emb(inx), inx.squeeze(1).tolist() #TODO: what if no self.tok_emb
        else:
            if match:
                t,i = emb_buf.match_embed(out_batch[:,-1,:], lossf=lossf)
                return t.unsqueeze(1), i.tolist()
            else:
                inx = None
                if calc_inx: _,inx = emb_buf.match_embed(out_batch[:,-1,:], lossf=lossf)
                if inx is not None: inx = inx.tolist()
                return out_batch[:,-1:,:], inx
    
    #project target corresponding to the last token for each batch row to embedding vector space
    #if the model has logits outhead then targets are in form of indexes and embedding vector is found based on index
    #otherwise the target is already in form of vector belonging to embedding vector space and no modification or mapping takes place
    def last_target_to_embed(self, target_batch, emb_buf, calc_inx=True, lossf=vector_lossfunc):
        if self.has_logits_outhead():
            with torch.no_grad():
                r = self.tok_emb(target_batch)[:, -1:, :] #TODO: what if no self.tok_emb
                return r, target_batch[:, -1].tolist()
        else:
            inx = None
            if calc_inx: _,inx = emb_buf.match_embed(target_batch[:, -1, :], lossf=lossf)
            if inx is not None: inx = inx.tolist()
            return target_batch[:, -1:, :], inx
    
    #feedbact info, uses the loss_last function over the given number of batches
    def evaluate_loader(self, data_loader, emb_buf, num_batches=None, lossf=vector_lossfunc):
        device = self.get_device()
        self.eval()
        with torch.no_grad():
            ret_len = 1
            if type(lossf) is tuple or type(lossf) is list: ret_len = len(lossf)
            else: lossf = [lossf]
            total_loss_last = torch.zeros(ret_len)
            total_loss_allin = torch.zeros(ret_len)
            if len(data_loader) == 0: return float("nan")
            elif num_batches is None: num_batches = len(data_loader)
            else: num_batches = min(num_batches, len(data_loader))
            for i, (input_batch, target_batch) in enumerate(data_loader):
                if i < num_batches:
                    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                    out_batch = self(input_batch)
                    for f in range(len(lossf)):
                        if not self.has_embed():
                            total_loss_allin[f] += lossf[f](out_batch, target_batch).item()
                        out_embeds, _    = self.last_result_to_embed(out_batch,    emb_buf, match=True, calc_inx=False, lossf=lossf[f])
                        target_embeds, _ = self.last_target_to_embed(target_batch, emb_buf, calc_inx=False, lossf=lossf[f])
                        total_loss_last[f] += lossf[f](out_embeds, target_embeds).item()
                else:
                    break
        total_loss_last = total_loss_last/num_batches
        total_loss_allin= total_loss_allin/num_batches
        return torch.sum(total_loss_last).item(), torch.sum(total_loss_allin).item(), total_loss_last.tolist(), total_loss_allin.tolist()
    
    #train
    def train_model(self, train_loader, val_loader, learning_rate, weight_decay, num_epochs,
                    trainmode=TRAINMODE_ALL,
                    eval_conf=None, emb_buf=None, modelfile=None):
        print(f"trainmode:{trainmode}")
        self.set_train_embeds(trainmode==TRAINMODE_ALL)
        device = self.get_device()
        t = time.time()
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen = 0
        epoch_loss = 0
        epoch_loss_prev = 0
        train_loss, train_loss_allin, train_loss_list, train_loss_allin_list = self.evaluate_loader(train_loader, emb_buf)
        val_loss, val_loss_allin, val_loss_list, val_loss_allin_list         = self.evaluate_loader(val_loader,   emb_buf)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Starting out: Train loss: {epoch_loss:.4f}/{train_loss:.4f}/{train_loss_allin:.4f}(batches:{len(train_loader)}) Val loss {val_loss:.4f}/{val_loss_allin:.4f}(batches:{len(val_loader)})")
        
        print(f"training for {num_epochs} epochs...")
        optimizer = None
        best = {
            "val_loss": val_loss,
            "model"   : self,
            "epoch"   : 0,
            "saved"   : True
            }
        #training cycle
        for epoch in range(1, num_epochs+1):
            #preparations for next training cycle
            self.train()  # Set model to training mode
            if optimizer == None or (0 < epoch_loss_prev and epoch_loss_prev < epoch_loss and 1e-06 < learning_rate):
                if optimizer is not None: learning_rate = learning_rate/2.0;
                params = self.parameters()
                if self.has_logits_outhead() and trainmode==TRAINMODE_ONLY_OUTHEAD:
                    params=self.out_head.parameters()
                optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
            epoch_loss_prev = epoch_loss
            bt = time.time()
            batches_processed = 0
            epoch_loss = 0
            
            #training cycle
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # reset loss gradients
                input_batch  = input_batch.to(device)
                target_batch = target_batch.to(device)
                seq_record = []
                out_batch = self(input_batch, seq_record)
                if self.has_logits_outhead():
                    if trainmode == TRAINMODE_ONLY_MIDDLE:
                        loss = vector_lossfunc(seq_record[-1], self.tok_emb(target_batch))
                    else:
                        loss = cross_entropy(out_batch, target_batch)
                        if trainmode == TRAINMODE_ALL_EXCEPT_EMBEDS_MIX:
                            loss *= vector_lossfunc(seq_record[-1], self.tok_emb(target_batch))
                else:
                    loss = vector_lossfunc(out_batch, target_batch)
                epoch_loss += loss.item()
                loss.backward()  #calc gradients
                optimizer.step()  # update weights
                tokens_seen += input_batch.numel()
                batches_processed += 1
                btt = time.time()
                if 10 < btt-bt:
                    print(f"batches processed:{batches_processed} of {len(train_loader)}")
                    bt = btt
            epoch_loss = epoch_loss / len(train_loader)
            
            #evaluation feedback and checkpoint savings
            if eval_conf is not None:
                num_batches = 0
                track_best_and_save = False
                if epoch <= 2 or epoch == num_epochs:
                    num_batches = len(val_loader)
                    track_best_and_save = True
                else:
                    for e in eval_conf:
                        if epoch % e[0] == 0 and num_batches < e[1]:
                            num_batches = e[1]
                            track_best_and_save = e[2]
                if 0 < num_batches:
                    n_train=1
                    if num_batches == len(val_loader): n_train = len(train_loader)
                    train_loss, train_loss_allin, train_loss_list, train_loss_allin_list = self.evaluate_loader(train_loader, emb_buf, num_batches=n_train)
                    val_loss, val_loss_allin, val_loss_list, val_loss_allin_list         = self.evaluate_loader(val_loader,   emb_buf, num_batches=num_batches)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    tt = time.time()
                    timetaken = tt-t
                    print(f"Ep {epoch}: lr:{learning_rate} \
Train loss: {epoch_loss:.4f}/{train_loss:.4f}/{train_loss_allin:.4f}(batches:{n_train}) \
Val loss {val_loss:.4f}/{val_loss_allin:.4f}(batches:{num_batches}) \
time taken: {int(timetaken/3600)}h {int((timetaken%3600)/60)}m {int(timetaken%60)}s")
                    if track_best_and_save:
                        if val_loss < best["val_loss"]:
                            mcopy = copy.deepcopy(self)
                            mcopy.eval()
                            best["val_loss"] = val_loss
                            best["model"] = mcopy
                            best["epoch"] = epoch
                            best["saved"] = False
                            print(f"registered as best:{val_loss}")
                        if (epoch % 10 == 0 or epoch == 1 or epoch == num_epochs) and not best["saved"] and modelfile is not None:
                            fname=modelfile+"-"+str(round(best["val_loss"], 3))
                            torch.save(best["model"].state_dict(), fname)
                            best["saved"] = True
                            print(f"model saved:{fname} (epoch:{best['epoch']})")
        print(f"best model: {best['val_loss']} (epoch:{best['epoch']})")
        self.eval()
        fname = modelfile+"-"+str(round(val_loss, 3))
        torch.save(self.state_dict(), fname)
        print(f"final model saved:{fname}")     
        return train_losses, val_losses, track_tokens_seen, best
    
    
    ########################## model derivation related funcs #########################
    
    def derived_model_context_length(self, n_rows, stride):
        return math.ceil(n_rows*self.get_chain_len()/stride)
    
    def derive_model(self, n_rows=1, stride=1, num_tr_blocks=None):
        conf = copy.copy(self.cfg)
        conf["tokenizer"] = None
        conf["logits"] = False #TODO: make it configurable
        conf["context_length"] = self.derived_model_context_length(n_rows, stride)
        if num_tr_blocks is not None: conf["n_layers"] = num_tr_blocks
        return TransformerModel(conf)
    
    def get_chain_len(self):
        return 1+self.get_num_tr_blocks()
    
    def derive_chain_data(self, inp, target, rows=1, stride=1, n_steps=1): #inp and target are lists
        self.eval()
        device = self.get_device()
        device_cpu = torch.device("cpu")
        embed = self.has_embed()
        inp_list = []
        target_list = []
        with torch.no_grad():
            seq_record = []
            inp_len = len(inp)
            inp = inp.unsqueeze(0).to(device)
            self(inp, seq_record=seq_record) #seq_rcord will be list of tensors of shape (batch, n_tokens, embed_size)
            _inp  = []
            clen = self.get_chain_len()
            new_context_len = self.derived_model_context_length(1, stride)
            new_context_len = self.derived_model_context_length(rows, stride)
            start = max(0, inp_len-rows-stride*math.ceil((n_steps-1)/clen))
            for j in range(start, inp_len): #each original sequence token accounts for entire new sequence
                for x in seq_record:
                    _inp.append(x[0][j])
                if j == inp_len-1:
                    _targ = _inp[1:] #new target sequence
                    t = target[j].to(device)
                    #print(f"...........................{t}")
                    if embed: t = self.tok_emb(t)
                    #print(f"...........................{t}")
                    _targ.append(t)
                    _inp_len = len(_inp)
                    __inp = []
                    __targ= []
                    for s in range(0, len(_inp), stride):
                        __inp .append( _inp[_inp_len-s-1])
                        __targ.append(_targ[_inp_len-s-1])
                    __inp.reverse()
                    __targ.reverse()
                    _inp = __inp
                    _targ= __targ
                    _inp_len = len(_inp)
                    for s in range(0, n_steps):
                        end = _inp_len-s
                        begin=end-new_context_len
                        if begin < 0: break
                        inp_list   .append(torch.stack(_inp [begin:end], dim=0).to(device_cpu))
                        target_list.append(torch.stack(_targ[begin:end], dim=0).to(device_cpu))
                    _inp = []
        return inp_list, target_list
    
    def derive_chain_dataset(self, dataset, rows=1, stride=1, n_steps=1):
        inp_list = []
        target_list = []
        for inp,target in dataset:
            i,t = self.derive_chain_data(inp, target, rows, stride, n_steps)
            inp_list.extend(i)
            target_list.extend(t)
        return ListDataset(inp_list, target_list)
    
    #model size information
    def print_size(self, long="True"):
        if long: print("------------------")
        n_tensors = 0
        n_param = 0
        param_size = 0
        i = 0
        for param in self.parameters():
            n_tensors += 1
            n = param.nelement()
            n_param += n
            param_size += n * param.element_size()
            if long: print(f"tensor:{i} element size:{param.element_size()} elements:{n}")
            i += 1
        if long: print("------------------")
        n_buffers = 0
        buffer_size = 0
        i = 0
        for buffer in self.buffers():
            n_buffers += 1
            n = buffer.nelement()
            buffer_size += n * buffer.element_size()
            if long: print(f"buffer:{i} element size:{buffer.element_size()} elements:{n}")
            i += 1
        size_all_mb = (param_size + buffer_size) / 1024**2
        print("------------------")
        print(f"Number of parameter tensors: {n_tensors}")
        print(f"Number of buffers: {n_buffers}")
        print(f"Number of parameters: {n_param}")
        print('model size: {:.3f}MB'.format(size_all_mb))
        print("------------------")

