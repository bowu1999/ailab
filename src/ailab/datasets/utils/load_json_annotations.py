import json


def load_dicts_from_jsonlines(filename):
    """
    从每行一个字典的jsonl文件读取内容
    :param filename: 文件路径
    :return: 字典组成的列表
    """
    dict_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                d = json.loads(line)
                dict_list.append(d)

    return dict_list