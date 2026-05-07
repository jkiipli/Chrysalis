import copy
import torch
from torch.utils.data import DataLoader
import sys
import tracemalloc
from pathlib import Path
from tokenizer import get_tokenizer
import conf
from model import TransformerModel, vector_lossfunc, mse_loss, cos_loss, mse_cos_loss, \
TRAINMODE_ALL,\
TRAINMODE_ALL_EXCEPT_EMBEDS_CE,\
TRAINMODE_ALL_EXCEPT_EMBEDS_MIX,\
TRAINMODE_ONLY_MIDDLE,\
TRAINMODE_ONLY_OUTHEAD
from data import ListDataset, MyDataset

device_cpu = torch.device("cpu")

def main(config, settings, modelfiles=None, 
         modeldir="models/work", tmpdir="tmp", 
         trainmode=TRAINMODE_ALL,
         train_aggregate=True):
    lang = config["lang"]
    trainfile = "data/large/train_1.9.dataw"+lang
    testfile = "data/test_1.9.dataw"+lang
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = device_cpu
    tokenizer = get_tokenizer(config, dirr=modeldir, trainfile=trainfile)
              
    #print(f"dbg0 mem:{tracemalloc.get_traced_memory()}")
    tracemalloc.start()
    with open(trainfile, "r", encoding="utf-8") as file:
        train_data = file.read()
    with open(testfile, "r", encoding="utf-8") as file:
        test_data = file.read()
    #initial datasets construction
    tail_size=3
    stride=1
    nlines = 1000000
    train_set = MyDataset(train_data, tokenizer, config["context_length"], 
                                tail_size=tail_size, stride=stride, nlines=nlines)
    train_set_org = train_set
    val_set   = MyDataset(test_data,  tokenizer, config["context_length"], 
                                tail_size=tail_size, stride=stride, nlines=nlines)
    val_set_org = val_set
    
    #print(f"dbg1 mem:{tracemalloc.get_traced_memory()}")
    tracemalloc.stop()
    
    model = TransformerModel(config)
    emb_buf = None
    n_derivatives=1
    if modelfiles is not None: n_derivatives = len(modelfiles)
    models = []
    for i in range(n_derivatives):
        model.print_size(long=False)
        model.to(device)
        tmp_modelpath = None
        new_model = True
        if modelfiles is not None and i < len(modelfiles):
            modelfile = modelfiles[i]
            modelpath = modeldir+"/"+modelfile
            tmp_modelpath = tmpdir+"/"+modelfile
            if Path(modelpath).exists():
                model.load_state_dict(torch.load(modelpath, weights_only=True, map_location=device))
                new_model = False
                print(f"model loaded:{modelpath}")
                model.eval() #set to eval mode because it is loaded with train mode on
                if model.has_embed():
                    if emb_buf is None: emb_buf = model.create_emb_buf(tokenizer, device)
            else: print(f"{modelpath} does not exist")
        if new_model: print("new model")
        
        if emb_buf is not None and not new_model and (0 == settings["num_epochs"] or i == n_derivatives-1):
            print("check model, train set:")
            check_model(model, tokenizer, train_set, emb_buf)
            print("check model, val set:")
            check_model(model, tokenizer, val_set, emb_buf, drop_last=False)
        if new_model or (i == n_derivatives-1 and not train_aggregate):
            if 0 < settings["num_epochs"]:
                print(f"training model {i}, save path: {tmp_modelpath}")
                best = train(model, train_set, val_set, settings, emb_buf, tmp_modelpath, True, trainmode=trainmode)
                model = best["model"]
                model.to(device)
                model.eval()
            elif new_model: return #cut short
        models.append((model, modelfile))
        if i == n_derivatives-1: break
        
        #prepare data and model for next derivation cycle
        if emb_buf is None and model.has_embed():
            emb_buf = model.create_emb_buf(tokenizer, device)
        #datasets construction for derived model
        n_rows = model.get_context_length()
        stride=1
        if modelfiles is not None and i+1 < len(modelfiles):
            modelfile = modelfiles[i+1]
            n_rows=parse_n_rows(modelfile)
            stride=parse_stride(modelfile)
        n_steps = 1 #model.get_chain_len()
        train_set, val_set = derive_data(model, train_set, val_set, n_rows, stride, n_steps)
        num_rows = 1
        num_rows = n_rows
        #derive model to train in the next cycle
        model = model.derive_model(num_rows, stride, model.get_num_tr_blocks())
        
    if train_aggregate: train_aggregate_model(device, config, settings, tokenizer, 
                  train_set_org, val_set_org, models, emb_buf, modeldir, tmpdir)
        
def train_aggregate_model(device, config, settings, tokenizer, 
                  train_set_org, val_set_org, models, emb_buf, modeldir, tmpdir):    
    d_list = []
    for dset in (train_set_org, val_set_org):
        print(f"deriving data for aggregate model...")
        x_list = []
        y_list = []
        for _inp,_targ in dset:
            inp = _inp
            targ= _targ
            x = []
            with torch.no_grad():
                for i in range(len(models)):
                    model, modelfile = models[i]
                    model.eval()
                    seq_record = []
                    model(inp.unsqueeze(0).to(device), seq_record=seq_record)
                    start = 1
                    if i == 0: start = 0
                    for k in range(start, len(seq_record)): x.append(seq_record[k][0][-1])
                    if i == len(models)-1: break
                    nextfile = models[i+1][1]
                    inp,targ = model.derive_chain_data(inp, targ, rows=parse_n_rows(nextfile), stride=parse_stride(nextfile), glue_rows=True, n_steps=1)
                    inp = inp[-1]
                    targ = targ[-1]
                y = x[1:]
                y.append(emb_buf.emb_raw[_targ[-1]])
                x_list.append(torch.stack(x, dim=0).to(device_cpu))
                y_list.append(torch.stack(y, dim=0).to(device_cpu))
            if len(x_list) % 1000 == 0: print(f"{len(x_list)}")
        print(f"size={len(x_list)}")
        d_list.append(ListDataset(x_list, y_list))  
    conf = copy.copy(config)
    conf["tokenizer"] = None
    conf["logits"] = False #TODO: make it configurable
    conf["context_length"] = len(d_list[0][0][0])
    print(f"context len: {conf['context_length']}")
    model = TransformerModel(conf)
    model.to(device)
    modelpath = modeldir+"/aggregate_"+modelfiles[0]
    tmp_modelpath = tmpdir+"/aggregate_"+modelfiles[0]
    if Path(modelpath).exists():
        model.load_state_dict(torch.load(modelpath, weights_only=True, map_location=device))
        print(f"model loaded:{modelpath} trainmode:{model.training}")
        model.eval()
    else:
        print(f"{modelpath} does not exist")
        best = train(model, d_list[0], d_list[1], settings, emb_buf, tmp_modelpath, pin_memory=True)
        model = best["model"]
        model.to(device)
        model.eval()
    print("check model, train set:")
    check_model(model, tokenizer, d_list[0], emb_buf)
    print("check model, val set:")
    check_model(model, tokenizer, d_list[1], emb_buf, drop_last=False)      

def derive_data(model, train_set, val_set, n_rows, stride, n_steps):
    print(f"deriving train data...")
    t_set = model.derive_chain_dataset(train_set, n_rows, stride, n_steps)
    print(f"deriving val data...")
    v_set = model.derive_chain_dataset(val_set, n_rows, stride, n_steps)
    print(f"derived data: train={t_set.__len__()} val={v_set.__len__()}")
    return t_set, v_set
    
def train(model, train_set, val_set, settings, emb_buf, modelfile, pin_memory, trainmode=TRAINMODE_ALL):
    num_workers = 0
    batch_size=settings["batch_size"]
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, persistent_workers=(0 < num_workers), pin_memory=pin_memory)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, 
        num_workers=num_workers, persistent_workers=(0 < num_workers), pin_memory=pin_memory)    
    print("training model...")
    train_losses, val_losses, tokens_seen, best = model.train_model(
        train_loader, val_loader, settings["learning_rate"], settings["weight_decay"], settings["num_epochs"],
        trainmode=trainmode,
        eval_conf=[(1,1,False), (5,len(val_loader),True)],
        emb_buf=emb_buf,
        modelfile=modelfile
    )
    return best

def check_model(model, tokenizer, data_set, emb_buf, drop_last=True):
    device = model.get_device()
    model.eval()
    with torch.no_grad():
        data_loader = DataLoader(
            data_set, batch_size=20, shuffle=False, drop_last=drop_last,
            num_workers=0, persistent_workers=False, pin_memory=True)#(device==device_cpu))
        loss_last_matched, loss_allin_raw, loss_last_matched_list, loss_allin_raw_list, = model.evaluate_loader(data_loader, emb_buf, lossf=(mse_loss, cos_loss, mse_cos_loss))
        lm = [f"{n:.4f}" for n in loss_last_matched_list]
        ar = [f"{n:.4f}" for n in loss_allin_raw_list]
        print(f"loss: last matched(mse,cos,mse_cos) {lm}")
        print(f"loss: all-in raw  (mse,cos,mse_cos) {ar}")       
        print(f"avg embedding vector length: {torch.mean(torch.linalg.norm(emb_buf.emb_raw, dim=1)).item()}")
        print(f"avg cos  from other vectors: {torch.mean(emb_buf.emb_norm @ emb_buf.emb_norm.T).item()}")
        
        data_loader = DataLoader(
            data_set, batch_size=1, shuffle=False, drop_last=drop_last,
            num_workers=0, persistent_workers=False, pin_memory=True)#(device==device_cpu))
        
        s_loss0 = 0
        s_loss1 = 0
        s_loss2 = 0
        n = 0
        for i, (input_batch, target_batch) in enumerate(data_loader):
            #if i % 3 != 2: continue
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            #print(f"{input_batch.shape} {target_batch.shape}") #if embed then (batches, context_len) else (batches, context_len, embed_size)
            '''
            if model.has_embed():
                #print(f"{input_batch}")
                print(f"{tokenizer.decode(input_batch)}")
            else:
                x = []
                for b in range(0, input_batch.size()[0]):
                    y = []
                    for j in range(0, model.get_context_length()):
                        e,inx = match_embed(input_batch[b][j].unsqueeze(0), emb_buf, vector_lossfunc)
                        y.append(inx[0])
                    x.append(y)
                x = torch.tensor(x).to(device)
                #print(f"{x}")
                print(f"{tokenizer.decode(x)}")
            '''
            out_batch = model(input_batch)
            out_last_matched, out_id     = model.last_result_to_embed(
                out_batch, emb_buf, match=True, calc_inx=True)
            out_last_raw, _     = model.last_result_to_embed(
                out_batch, emb_buf, match=False, calc_inx=False)
            target_last, targ_id = model.last_target_to_embed(
                target_batch, emb_buf, calc_inx=True)
            
            loss0 = vector_lossfunc(out_last_matched, target_last).item()
            loss1 = vector_lossfunc(out_last_raw,     target_last).item()
            loss2 = 0
            if not model.has_logits_outhead():
                loss2 = vector_lossfunc(out_batch, target_batch).item()
            s_loss0 += loss0
            s_loss1 += loss1
            s_loss2 += loss2
            if i % 10 == 0 or i == len(data_loader)-1:
                print(f"{i}: out:{out_id} target:{targ_id}\t {loss0:.4f}({s_loss0/(n+1):.4f}) {loss1:.4f}({s_loss1/(n+1):.4f}) {loss2:.4f}({s_loss2/(n+1):.4f})")
            if 200 <= i: break
            n += 1

def parse_n_rows(s):
    i = s.rfind("/")
    if 0 <= i: s = s[i+1:]
    i = 1
    if s.startswith("ddd"): i = 3
    elif s.startswith("dd"): i = 2
    return int(s[i:s.find("-")])
    
def parse_stride(s): #TODO
    i = s.rfind("/")
    if 0 <= i: s = s[i+1:]
    return int(s[s.find("-")+1:s.find("_")])

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    config = conf.GPT_CONFIG_V13 #configuration for models
    settings = {
        "learning_rate": 5e-3,
        "num_epochs": 10, #set to 0 for evaluation mode
        "batch_size": 20,
        "weight_decay": 0.01
    }
    #first is base model, second is derivative, third is second derivative, etc
    #derivative prefixes follow certain pattern that are parsed subsequently
    #derivative models are structurally same as base model they just lack embeddings layer and out-head, and they take different (derived) input
    modelprefixes = ["",   "d60-3_"]
    modelsuffixes = ["-0.13", "-0.13-0.116"]
    modelfiles = []
    for i in range(len(modelprefixes)):
        modelfiles.append(modelprefixes[i]+"model_v"+str(config["version"])+modelsuffixes[i])
    main(config,settings,modelfiles,
         modeldir="models", #working directory where model files are loaded from.
         tmpdir="tmp", #The output files are written here
         trainmode=TRAINMODE_ALL, #layer training specifics, affects only base model training
         train_aggregate=False #build an aggregate model on top of last token flow in base model and derivatives
         )
    