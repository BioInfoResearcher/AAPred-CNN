import re
from transformers import T5Tokenizer


def get_tokenizer(path_tokenizer=None):
    if path_tokenizer is None:
        path_tokenizer = '../data/meta_data'
    tokenizer = T5Tokenizer.from_pretrained(path_tokenizer, do_lower_case=False)
    return tokenizer


def get_map_dict():
    tokenizer = get_tokenizer()
    raw_map_dict = tokenizer.get_vocab()
    residue2id = {}
    for key, value in raw_map_dict.items():
        if 'extra_id' not in key:
            if '<' in key:
                residue2id[key] = value
            else:
                residue2id[key[-1]] = value
    id2residue = {}
    for key, value in residue2id.items():
        id2residue[value] = key
    return {'raw_map_dict': raw_map_dict, 'residue2id': residue2id, 'id2residue': id2residue}


def get_sequence_from_id(id_list):
    id2residue = get_map_dict()['id2residue']
    raw_seq = [id2residue[id] for id in id_list]
    seq_list = [s for s in raw_seq if '<' not in s]
    seq = ''.join(seq_list)
    return seq


def tokenize(data_list, path_tokenizer=None):
    tokenizer = get_tokenizer(path_tokenizer)
    pattern = re.compile('.{1}')
    seqs = [' '.join(pattern.findall(seq)) for seq in data_list]
    seqs_std = [re.sub(r"[UZOB]", "X", seq) for seq in seqs]
    ids = tokenizer.batch_encode_plus(seqs_std, add_special_tokens=True, padding=True)
    return ids


def get_std_residue():
    id2residue = get_map_dict()['id2residue']
    non_std_list = ['<pad>', '</s>', '<unk>', '</eos>', 'U', 'Z', 'O', 'B', 'X']
    residue_list = id2residue.values()
    std_residue_ids = [i for i, residue in enumerate(residue_list) if residue not in non_std_list]
    std_residue_tokens = [id2residue[i] for i in std_residue_ids]
    return std_residue_tokens, std_residue_ids


if __name__ == '__main__':
    raw_map_dict = get_map_dict()['raw_map_dict']
    print('raw_map_dict', raw_map_dict)
    id2residue = get_map_dict()['id2residue']
    residue2id = get_map_dict()['residue2id']
    print('id2residue', id2residue)
    print('residue2id', residue2id)
    id_list = [3, 4, 5, 6, 7, 8, 1, 0, 0, 0, 0]
    seq = get_sequence_from_id(id_list)
    print('seq', seq)
    std_residue_tokens, std_residue_ids = get_std_residue()
    print('std_residue_tokens', std_residue_tokens)
    print('std_residue_ids', std_residue_ids)
