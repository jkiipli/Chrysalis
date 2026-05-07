#adapted code from
#https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb#scrollTo=HxIPN_Kq_Zir

import sys
from pathlib import Path
from transformers import AutoTokenizer

def batch_iterator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]
        
def get_dataset(train_file):
    with open(train_file, "r", encoding="utf-8") as file:
        text = file.read()
        print(f"text length: {len(text)}")
        return text.splitlines()
       
def get_tokenizer_gpt2(tokenizer_file, train_file=None, vocab_size=25000):
    if tokenizer_file is not None and Path(tokenizer_file).exists():
        print(f"loading tokenizer from {tokenizer_file}...")
        t = AutoTokenizer.from_pretrained(tokenizer_file)
        print(f"tokenizer loaded")
        return t
    elif train_file is not None:
        print("training new gpt2 tokenizer...")
        batch_size = 10 #1000
        dataset = get_dataset(train_file)
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        '''
        Make sure that the tokenizer you picked as a fast version (backed by the Tokenizers library) otherwise the rest of the notebook will not run:
        '''
        print(f"is_fast={tokenizer.is_fast}")
        new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(dataset, batch_size), vocab_size=vocab_size)
        '''
        And that's all there is to it! The training goes very fast thanks to Tokenizers library, backed by Rust.
        You now have a new tokenizer ready to preprocess your data and train a language model. You can feed it input texts as usual:
        
        print(dataset[0])
        print(new_tokenizer(dataset[0])[])
        print(dataset[:5])
        print(new_tokenizer(dataset[:5]))
        
        You can save it locally with the save_pretrained method:
        '''
        new_tokenizer.save_pretrained(tokenizer_file)
        print(f"new tokenizer saved:{tokenizer_file}")
        return new_tokenizer
    else:
        raise Exception("file not found:"+str(tokenizer_file))


from transformers import GPT2TokenizerFast
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer    
'''
Building your tokenizer from scratch
'''
def get_tokenizer_bpe(tokenizer_file, train_file=None, vocab_size=25000):
    if tokenizer_file is not None and Path(tokenizer_file).exists():
        print(f"loading tokenizer from {tokenizer_file}...")
        t = AutoTokenizer.from_pretrained(tokenizer_file)
        print(f"tokenizer loaded")
        return t
    elif train_file is not None:
        print("training new bpe tokenizer...")
        batch_size = 10 #1000
        dataset = get_dataset(train_file)
            
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        '''
        If we want to have a quick look at how it preprocesses the inputs, we can call the pre_tokenize_str method:
        tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!")
        We used the same default as for GPT-2 for the prefix space, so you can see that each word gets an initial 'G' added at the beginning, except the first one.
        We can now train our tokenizer! This time we use a BpeTrainer.
        '''
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
        tokenizer.train_from_iterator(batch_iterator(dataset, batch_size), trainer=trainer)
        '''
        To finish the whole pipeline, we have to include the post-processor and decoder:
        '''
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        '''
        And like before, we finish by wrapping this in a Transformers tokenizer object:
        '''
        new_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
        new_tokenizer.save_pretrained(tokenizer_file)
        
        print(f"new tokenizer saved:{tokenizer_file}")
        return new_tokenizer
    else:
        raise Exception("file not found:"+str(tokenizer_file))

def get_tokenizer(gpt_config, dirr="", trainfile=None):
    lang = gpt_config["lang"]
    vocab_size=gpt_config["vocab_size"]
    if not dirr == "" and not dirr.endswith("/"): dirr = dirr + "/"
    if "gpt2" == gpt_config["tokenizer"]:
        return get_tokenizer_gpt2(dirr+"tokenizer-gpt2-"+lang+"-"+str(vocab_size), trainfile, vocab_size=vocab_size)
    elif "bpe" == gpt_config["tokenizer"]:
        return get_tokenizer_bpe(dirr+"tokenizer-bpe-"+lang+"-"+str(vocab_size), trainfile, vocab_size=vocab_size)
    else:
        raise Exception("unknown tokenizer type:"+str(gpt_config["tokenizer"]))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    s = ["mida m-i LLM-i!", "jakm kell bamj", "0c20 2b30 0c10"]
    vocab_size=10000
    tokenizer1 = get_tokenizer_gpt2("tokenizer-gpt2-"+str(vocab_size), "../JavaScreener/tmp/train_1.9.dataw", vocab_size=vocab_size)
    tokenizer2 = get_tokenizer_bpe("tokenizer-bpe-"+str(vocab_size), "../JavaScreener/tmp/train_1.9.dataw", vocab_size=vocab_size)
    for x in s:
        print("---------------")
        print(x)
        e1 = tokenizer1.encode(x)
        e2 = tokenizer2.encode(x)
        print(e1)
        print(e2)
        print(tokenizer1.decode(e1))
        print(tokenizer2.decode(e2))
    print("done")
    