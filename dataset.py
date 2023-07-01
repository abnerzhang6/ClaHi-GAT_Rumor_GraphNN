import json, os, re, tqdm
import torch
from torch import nn
from torch.utils import data
from torchtext.data import get_tokenizer
import torchtext.vocab as vocab

cache_dir = r'D:\pycharmProjects\PHEME\.vector_cache'
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)


def clean_text(text):
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    return text


def seg_and_encode(text, obj_len):
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(text)
    while len(tokens) < obj_len:
        tokens.append('unk')
    l = []
    for t in tokens:
        try:
            l.append(glove.stoi[t])
        except KeyError:
            l.append(glove.stoi['unk'])
    return l


def read_tree(idx, struct, main_dict, adj):
    l = [0] * len(main_dict)
    if type(struct) != list:
        for id in list(struct.keys()):
            l[main_dict[id]] = 1
            read_tree(id, struct[id], main_dict, adj)
        adj[main_dict[idx]] = l
    else:
        adj[main_dict[idx]] = l
    return adj


class PHEME_dataset(data.Dataset):
    def __init__(self, event_name=None):
        if event_name is None:
            self.event_name = os.listdir(r"D:\pycharmProjects\PHEME\data")
        else:
            self.event_name = [event_name]
        obj_sentence_length = 66
        obj_num_respond = 346
        self.seg_text = []
        self.label = []
        self.adj = []

        for en in self.event_name:
            r_nr = os.listdir(r"D:\pycharmProjects\PHEME\data//" + en)
            for rnr in r_nr:
                eids = os.listdir(r"D:\pycharmProjects\PHEME\data//" + en + "//" + rnr)
                for eid in eids:
                    text = []
                    re_filenames = os.listdir(
                        r"D:\pycharmProjects\PHEME\data//" + en + "//" + rnr + "//" + eid + "//reactions")
                    with open(
                            r"D:\pycharmProjects\PHEME\data//" + en + "//" + rnr + "//" + eid + "//source-tweets//" + eid + ".json",
                            'r') as f:
                        j_file = json.load(f)
                        content = j_file['text']
                        content = clean_text(content).strip()
                    se_content = seg_and_encode(content, obj_sentence_length)
                    text.append(se_content)

                    for re_filename in re_filenames:
                        with open(
                                r"D:\pycharmProjects\PHEME\data//" + en + "//" + rnr + "//" + eid + "//reactions//" + re_filename,
                                'r') as f:
                            j_file = json.load(f)
                            content = j_file['text']
                            content = clean_text(content).strip()
                        se_content = seg_and_encode(content, obj_sentence_length)
                        text.append(se_content)

                    while len(text) < obj_num_respond:
                        text.append([glove.stoi['unk']] * obj_sentence_length)

                    text = torch.tensor(text)
                    self.seg_text.append(text)

                    if rnr == "non-rumours":
                        self.label.append(torch.tensor([0, 1]))
                    elif rnr == "rumours":
                        self.label.append(torch.tensor([1, 0]))

                    with open(r"D:\pycharmProjects\PHEME\data//" + en + "//" + rnr + "//" + eid + "//structure.json",
                              'rb') as f:
                        j_file = json.load(f)
                        struct = j_file
                    rumours_ids = re_filenames
                    rumours_ids = [i[:-5] for i in rumours_ids]
                    tweet_id = os.listdir(
                        r"D:\pycharmProjects\PHEME\data//" + en + "//" + rnr + "//" + eid + "//source-tweets")
                    rumours_ids.insert(0, tweet_id[0][:-5])
                    main_dict = {}

                    new_rumours_ids = list(set(rumours_ids))
                    new_rumours_ids.sort(key=rumours_ids.index)
                    for idx, element in enumerate(new_rumours_ids):
                        main_dict[element] = idx

                    adj = [-1] * len(main_dict)
                    id = list(main_dict.keys())[0]
                    try:
                        adj = read_tree(id, struct[id], main_dict, adj)
                    except KeyError:
                        pass

                    for i, item in enumerate(adj):
                        if isinstance(item, int):
                            adj[i] = [0] * len(main_dict)

                    adj = torch.tensor(adj)
                    pad = nn.ZeroPad2d(padding=(0, 346 - adj.shape[0], 0, 346 - adj.shape[1]))
                    adj = pad(adj)
                    diag_ele = torch.tensor([1] * len(main_dict) + [0] * (346 - len(main_dict)))
                    diag_mat = torch.diag_embed(diag_ele)
                    adj = adj + diag_mat + adj.t()
                    self.adj.append(adj)

    def __getitem__(self, index):
        text = self.seg_text[index]
        adj = self.adj[index]
        label = self.label[index]
        return text, adj, label

    def __len__(self):
        return len(self.label)

# dataset = PHEME_dataset()
