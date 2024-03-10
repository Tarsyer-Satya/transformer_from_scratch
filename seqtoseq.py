import random
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import vocab
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


multi30k = load_dataset("bentrevett/multi30k")
train, test = multi30k['train'], multi30k['test']



ger_counter = Counter()
eng_counter = Counter()
for data in tqdm(train):
    ger_counter.update(tokenizer_ger(data['de'].lower()))
    eng_counter.update(tokenizer_eng(data['en'].lower()))   

ger_vocab = vocab(ger_counter, min_freq=2, specials=("<unk>", "<pad>", "<sos>", "<eos>"))
eng_vocab = vocab(eng_counter, min_freq=2, specials=("<unk>", "<pad>", "<sos>", "<eos>"))
ger_vocab.set_default_index(ger_vocab["<unk>"])
eng_vocab.set_default_index(eng_vocab["<unk>"])
print(f"Size of German Vocab : {len(ger_vocab)}\n Size of English Vocab : {len(eng_vocab)}")
print(type(ger_vocab))

