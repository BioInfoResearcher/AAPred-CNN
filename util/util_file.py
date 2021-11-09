import re


def read_tsv_data(filename, skip_first=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))
    return [sequences, labels]


def read_txt_data(filename, skip_first=False):
    sequences = []
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        for line in file:
            seq = re.match('[\w]+', line).group()
            sequences.append(seq)
    return sequences


def write_tsv_data(tsv_filename, labels, sequences):
    if len(labels) == len(sequences):
        with open(tsv_filename, 'w') as file:
            file.write('index\tlabel\tsequence\n')
            for i in range(len(labels)):
                file.write('{}\t{}\t{}\n'.format(i, labels[i], sequences[i]))
        return True
    return False
