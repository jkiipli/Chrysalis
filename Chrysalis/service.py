import time
import socket
import sys
import torch
import copy
from model import TransformerModel
from train import parse_n_rows, parse_stride
from tokenizer import get_tokenizer
import conf
import urllib.parse
import math
import threading

CMD_SHUTDOWN = 1
CMD_PING     = 2
CMD_APPLY    = 10

PORT = 5555
lock = threading.Lock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}
sys.stdout.reconfigure(encoding='utf-8')

#model and tokenizer bundeled together
#the modelfile needs to follow certain naming convention, 
#in particular it needs to contain version number in form like "_v3_" for example in it's name
#as this is used to identify and load the appropriate gpt config object
#also the modelfile should not contain whitespaces nor commas in it's path for command parsing is done based on space character
class Modelwrap():
    def __init__(self, modelfile, base_modelwrap=None):
        self.modelfile = modelfile
        if base_modelwrap is None:
            gpt_config = conf.get_conf(modelfile)
            if gpt_config is None: raise Exception("unable to load gpt configuration")
            self.model = TransformerModel(gpt_config)
            slashinx = modelfile.rfind("/")
            dirr = ""
            if 0 <= slashinx: dirr = modelfile[0:slashinx]
            self.tokenizer = get_tokenizer(gpt_config, dirr=dirr)
        else:
            self.model = base_modelwrap.model.derive_model(n_rows=parse_n_rows(self.modelfile), stride=parse_stride(self.modelfile), num_tr_blocks=None) #TODO: make n_rows and blocks configurable
        self.model.load_state_dict(torch.load(modelfile, weights_only=True, map_location=device))
        self.model.eval()
        self.model.to(device)
        print(f"model loaded:{modelfile}", flush=True)
        self.emb_buf = None
        if base_modelwrap is None:
            self.emb_buf = self.model.create_emb_buf(self.tokenizer, device)
        
    #deep copy of the model and shallow copy of the other fluff
    def copy(self):
        mw = copy.copy(self)
        mw.model = copy.deepcopy(self.model)
        return mw

#several models bundeled together with first being the base model and others derivatives of the immediately preceding model
class Modelpack():   
    def __init__(self, modelfiles):
        self.modelwraps = []
        prev = None
        for modelfile in modelfiles:
            if not modelfile in models:
                models[modelfile] = Modelwrap(modelfile, prev)
            m = models[modelfile]
            self.modelwraps.append(m)
            prev = m
            
    #copy the content, so that subsequent finetuning of the models do not affect originals
    def copy(self):
        mp = copy.copy(self)
        mp.modelwraps = []
        for m in self.modelwraps: mp.modelwraps.append(m.copy())
        return mp
    
    def get_wrap(self, i):
        return self.modelwraps[i]
    
    def __len__(self):
        return len(self.modelwraps)
    
def apply(modelpack, encoded, max_tokens, max_len, prob_info_cnf, parent_node_info=None):
    if max_tokens <= 0 or max_len <= 0: return parent_node_info, ""
    with torch.no_grad():
        tokenizer = modelpack.get_wrap(0).tokenizer
        emb_buf = modelpack.get_wrap(0).emb_buf
        model_input = encoded
        for i in range(len(modelpack)):
            modelwrap = modelpack.get_wrap(i)
            model = modelwrap.model
            trainmode = model.training
            if trainmode: model.eval()
            seq_record = []
            inp_len = model.get_context_length()
            #print(f"dbg0...{i} {inp_len}", flush=True)
            model_input  = model_input[:, -inp_len:]
            #print(f"dbg1...{i} {model_input.shape}", flush=True)
            model_output = model(model_input, seq_record=seq_record)
            #print(f"dbg2...{i} {model_output.shape}", flush=True)
            if trainmode: model.train()
            if i < len(modelpack)-1:
                model_input = []
                rows = parse_n_rows(modelpack.get_wrap(i+1).modelfile)
                stride=parse_stride(modelpack.get_wrap(i+1).modelfile)
                for j in range(inp_len-rows, inp_len): #each original sequence token accounts for entire new sequence
                    for x in seq_record:
                        model_input.append(x[0][j])
                __inp = []
                for j in range(0, len(model_input), stride):
                    __inp.append(model_input[len(model_input)-j-1])
                __inp.reverse()
                model_input = __inp
                model_input = torch.stack(model_input, dim=0).unsqueeze(0).to(device) #prepate input for next cycle
                #print(f"dbg4...{model_input.shape}", flush=True)
                continue
            logits = model.last_result_to_logits(model_output, emb_buf)
            probs = torch.nn.functional.softmax(logits, dim=1)
            values, indices = torch.sort(probs, descending=True)
            idx_next = indices[:,0:1]
            
        if prob_info_cnf is not None:
            distr_step_exp_base  = float(prob_info_cnf[0]);
            distr_step_size_base = float(prob_info_cnf[1]);
            prob_step            = float(prob_info_cnf[2]);
            max_num_alts         =   int(prob_info_cnf[3]);
            num_alts_prob_limit  = float(prob_info_cnf[4]);
            max_spawn            =   int(prob_info_cnf[5]);
            sep = "_"
            prec = 6 #precision for probabilities
            #cut out batch dimensin
            values = values[0].tolist()
            indices = indices[0].tolist()
            i = 0
            v = 0
            vv = []
            upper_bound = distr_step_size_base
            for ix in range (0, len(values)):
                if upper_bound <= ix:
                    vv.append(round(v,prec))
                    i += 1
                    upper_bound = int(distr_step_size_base*(distr_step_exp_base ** i))
                v += values[ix]
            vv.append(1.0)#set last element explicitly to 1.0 because of precicion errors round(v,prec))
            v = 0
            ss = []
            steps_taken = 0
            for i in range(0, len(values)):
                v += values[i]
                while (len(ss)+1)*prob_step <= v:
                    step = i-steps_taken
                    ss.append(step)
                    steps_taken += step
            ss[-1] += len(values)-steps_taken #adjust result because of rounding errors to make steps taken to sum up to len(values)
            #set up root node info if not present
            if parent_node_info is None:
                logval = math.log(len(values)/distr_step_size_base, distr_step_exp_base)
                vvv = [0] * (2+math.floor(logval))
                vvv[0] = 1.0
                sss = [0] * int(1.0/prob_step)
                sss[-1] = len(values)
                idx_prev = encoded[:, -1:]
                iteminfo = str(idx_prev[0][0].item())+"="+tokenizer.decode(idx_prev)[0]
                parent_node_info = str(len(iteminfo))+sep+iteminfo+"0"+sep+"1.0"+sep+str(len(values))+sep+str(vvv).replace(" ", "")+sep+str(sss).replace(" ", "")
                parent_node_info = str(len(parent_node_info))+sep+parent_node_info
                #print(f"{parent_node_info}", flush=True)
            #add generated alternatives
            v = 0
            main_path_str = ""
            for inx in range(0, max_num_alts):
                idx_next = torch.Tensor([[indices[inx]]])
                idx_next = idx_next.type(torch.int32)
                token = tokenizer.decode(idx_next)[0]
                if inx == 0: main_path_str = token
                iteminfo = str(indices[inx])+"="+token
                nodeinfo = str(len(iteminfo))+sep+iteminfo+str(inx)+sep+str(round(values[inx], prec))+sep+str(len(values))+sep+str(vv).replace(" ", "")+sep+str(ss).replace(" ", "")
                nodeinfo = str(len(nodeinfo))+sep+nodeinfo
                #print(f"{nodeinfo}", flush=True)
                
                #add max_spawn subtrees before wrap and also add tokens to main path string
                if inx < max_spawn and 1 < max_tokens and 0 < max_len-len(token):
                    nodeinfo, next_str = apply(modelpack, torch.cat((encoded, idx_next), dim=1), max_tokens-1, max_len-len(token), prob_info_cnf, nodeinfo)
                    if inx == 0: main_path_str += next_str
                else: nodeinfo = str(len(nodeinfo))+sep+nodeinfo #wrap
                parent_node_info += nodeinfo #add wrapped nodeinfo to parent
                v += values[inx]
                if num_alts_prob_limit < v: break
            #print(f"dbg0", flush=True)
            parent_node_info = str(len(parent_node_info))+sep+parent_node_info #final wrap
            s = main_path_str
        else:
            token = tokenizer.decode(idx_next)[0]
            _, next_tokens = apply(modelpack, torch.cat((encoded, idx_next), dim=1), max_tokens-1, max_len-len(token), None, None)
            s = token + next_tokens
        return parent_node_info, s

def send(conn, s):
    bs = s.encode('utf-8')
    l = len(bs)
    blen = bytearray([(l >> 24) & 0xff, (l >> 16) & 0xff, (l >> 8) & 0xff, l & 0xff])
    try:
        conn.send(blen)
        conn.send(bs)
        #print("response sent")
        return True
    except:
        print("cannot send response")
        return False

def processCmd(conn):
    length = (conn.recv(1)[0] << 24) | (conn.recv(1)[0] << 16) | (conn.recv(1)[0] << 8) | (conn.recv(1)[0])
    #print("bytes:",length, flush=True)
    b = conn.recv(length)
    #print("b=", b)
    s = b.decode('utf8')
    x = s.find(" ")
    if x == -1: x = len(s)
    cmd = int(s[0:x]);
    #print("cmd=",cmd, flush=True)
    with lock:
        try:
            if cmd == CMD_SHUTDOWN:
                exit();
            elif cmd == CMD_PING:
                send(conn, str(cmd)+" 0")
            elif cmd == CMD_APPLY:
                s = s[x+1:len(s)]
                x = s.find(" ")
                modelfiles = s[0:x]; #model file name
                if modelfiles.startswith("["):
                    modelfiles = modelfiles[1:-1]
                    modelfiles = modelfiles.split(",")
                else:
                    modelfiles = [modelfiles]
                s = s[x+1:len(s)]
                x = s.find(" ")
                probinfo = s[0:x]; #logits probabilities format
                s = s[x+1:len(s)]
                x = s.find(" ")
                max_tokens = int(s[0:x]);
                s = s[x+1:len(s)]
                x = s.find(" ")
                max_len = int(s[0:x]);
                s = s[x+1:len(s)]
                modelpack = Modelpack(modelfiles)
                tokenizer = modelpack.get_wrap(0).tokenizer
                encoded = tokenizer.encode(s)
                '''
                print(f"{encoded}", flush=True)
                toks = ""
                for e in encoded: toks += ","+str(e)+":"+tokenizer.decode(e)
                print(f"{toks}", flush=True)
                '''
                encoded = torch.tensor(encoded).unsqueeze(0).to(device)  # add batch dimension
                   
                prob_info_cnf = None
                if probinfo.startswith("1"):
                    prob_info_cnf = probinfo.split(":")[1:]
                #print(f"prob_info_cnf={prob_info_cnf}", flush=True)
                #apply
                probs, s = apply(modelpack, encoded, max_tokens, max_len, prob_info_cnf)
                encprobs = "None"
                if probs is not None: encprobs = urllib.parse.quote(probs)
                #print(f"encprobs={encprobs}", flush=True)
                send(conn, str(cmd)+" 0 "+encprobs+" "+s)
            else:
                raise Exception("unknown command")
        except Exception as e:
            send(conn, str(cmd)+" 1 "+str(e))

def run_connection(conn):
    with conn:
        try:
            while True: processCmd(conn)
        except:
            print("run_connection exception: ", sys.exc_info()[0], flush=True)
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python "+sys.argv[0]+" port", flush=True)
    else:
        PORT = int(sys.argv[1])
    print("PORT="+str(PORT), flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', PORT))
        s.listen()
        accept = True
        while accept:
            print('listening network port', PORT, '...', flush=True)
            conn, addr = s.accept()
            print('connected by', addr, flush=True)
            thread = threading.Thread(target=run_connection, args=(conn,))
            thread.start()
    
            