import random


def split_dataset_manual(X, y = None, train_ratio=0.8, shuffle=True, random_seed=None):
    """
    手动实现数据集划分。
    
    :param X: 特征数据列表
    :param y: 标签数据列表
    :param train_ratio: 训练集所占的比例
    :param shuffle: 是否在划分前打乱数据
    :param random_seed: 随机种子，用于保证结果可重复
    :return: 分割后的训练集特征、训练集标签、测试集特征、测试集标签
    """
    if random_seed is not None:
        random.seed(random_seed)
    if y is None:
        y = [None] * len(X)
    
    dataset = list(zip(X, y))
    if shuffle:
        random.shuffle(dataset)
    
    train_size = int(len(dataset) * train_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    
    X_train, y_train = zip(*train_set)
    X_test, y_test = zip(*test_set)
    
    return list(X_train), list(X_test), list(y_train), list(y_test)