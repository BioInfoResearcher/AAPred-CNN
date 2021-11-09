from util import util_file

if __name__ == '__main__':
    path_main_train_pos = '../data/task_data/Anti-angiogenic Peptide/train/main_train_pos.txt'
    AAP_main_train_pos = util_file.read_txt_data(path_main_train_pos)
    print('AAP_main_train_pos:\n', len(AAP_main_train_pos), AAP_main_train_pos)

    path_main_train_neg = '../data/task_data/Anti-angiogenic Peptide/train/main_train_neg.txt'
    AAP_main_train_neg = util_file.read_txt_data(path_main_train_neg)
    print('AAP_main_train_neg:\n', len(AAP_main_train_neg), AAP_main_train_neg)

    path_main_test_pos = '../data/task_data/Anti-angiogenic Peptide/test/main_test_pos.txt'
    AAP_main_test_pos = util_file.read_txt_data(path_main_test_pos)
    print('AAP_main_test_pos:\n', len(AAP_main_test_pos), AAP_main_test_pos)

    path_main_test_neg = '../data/task_data/Anti-angiogenic Peptide/test/main_test_neg.txt'
    AAP_main_test_neg = util_file.read_txt_data(path_main_test_neg)
    print('AAP_main_test_neg:\n', len(AAP_main_test_neg), AAP_main_test_neg)

    AAP_main_train_data = AAP_main_train_pos + AAP_main_train_neg
    AAP_main_train_labels = [1] * len(AAP_main_train_pos) + [0] * len(AAP_main_train_neg)
    tsv_name_train = '../data/task_data/Anti-angiogenic Peptide/train/main.tsv'
    util_file.write_tsv_data(tsv_name_train, AAP_main_train_labels, AAP_main_train_data)

    AAP_main_test_data = AAP_main_test_pos + AAP_main_test_neg
    AAP_main_test_labels = [1] * len(AAP_main_test_pos) + [0] * len(AAP_main_test_neg)
    tsv_name_test = '../data/task_data/Anti-angiogenic Peptide/test/main.tsv'
    util_file.write_tsv_data(tsv_name_test, AAP_main_test_labels, AAP_main_test_data)
