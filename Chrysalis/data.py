

import torch
from torch.utils.data import Dataset, DataLoader

class ListDataset(Dataset):
    def __init__(self, input_list, target_list):
        self.input_ids = input_list
        self.target_ids = target_list

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class MyDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, tail_size=1, stride=1, shift=0, nlines=1000000000, lineshift=0):
        self.input_ids = []
        self.target_ids = []
        self.firstline = None
        lines = txt.splitlines()
        lcount = 0
        for line in lines:
            if len(line) < 1: continue
            lcount += 1
            if lcount <= lineshift:
                continue
            #print(line)
            while 0 < shift:
                lastspace = line.rfind(" ")
                line = line[0:lastspace]
                shift = shift-1
            ids = tokenizer.encode(line, allowed_special={"<|endoftext|>"})
            #if 0 < shift: ids = ids[0:len(ids)-shift]
            l = len(ids)
            if l < max_length+1: raise Exception("")
            for i in range(0, tail_size, stride):
                end = l-i-1
                start = end-max_length
                if start < 0: break
                input_chunk = ids[start:end]
                target_chunk = ids[start+1:end+1]
                if len(input_chunk) != max_length or len(target_chunk) != max_length:
                    raise Exception("input:"+str(len(input_chunk))+" target:"+str(len(target_chunk)))
                ic = torch.tensor(input_chunk)
                tc = torch.tensor(target_chunk)
                
                self.input_ids.append(ic)
                self.target_ids.append(tc)
                #print(f"{len(input_chunk)}:{input_chunk}")
                #print(f"{len(target_chunk)}:{target_chunk}")
            if self.firstline is None: self.firstline = line
            if nlines <= lcount-lineshift: break
        print(f"dataset size:{len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

