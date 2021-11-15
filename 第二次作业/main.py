from crf import CRFModel
from evaluating import Metrics
import os, sys, pickle

CRF_MODEL_PATH = './model/crf.pkl'


def build_corpus(model, make_vocab=True, data_dir="./conll2003"):
    """读取数据"""
    assert model in ['eng.train', 'eng.testa', 'eng.testb']

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, model), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, char, char2, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def crf_train_eval(train_data, test_data, remove_O=False):
    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, CRF_MODEL_PATH)

    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    """训练模型，评估结果"""

    # 读取数据
    print('loading...')
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("eng.train")
    dev_word_lists, dev_tag_lists = build_corpus("eng.testa", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("eng.testb", make_vocab=False)

    # 训练评估CRF模型
    print('training...')
    crf_pred = crf_train_eval((train_word_lists, train_tag_lists), (test_word_lists, test_tag_lists))

    # 加载并评估CRF模型
    print('evaluating...')
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(dev_word_lists)
    metrics = Metrics(dev_tag_lists, crf_pred)
    metrics.report_scores()
    metrics.report_confusion_matrix()


if __name__ == "__main__":
    main()