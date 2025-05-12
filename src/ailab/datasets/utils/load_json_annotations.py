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


def load_json_file(file_path):
    """
    该函数接收一个文件路径作为参数，尝试打开并读取该路径下的JSON文件，
    然后将文件内容解析为Python对象（通常是字典或列表）。
    
    :param file_path: JSON文件的路径
    :return: 解析后的Python对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是一个有效的JSON文件")
    except Exception as e:
        print(f"发生了一个错误：{e}")


def save_dicts_to_jsonlines(dict_list, filename):
    """
    将字典列表保存为JSON Lines格式，每行一个字典。
    :param dict_list: List[dict]，要保存的字典列表
    :param filename: str，目标文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in dict_list:
            json_str = json.dumps(d, ensure_ascii=False)
            f.write(json_str + '\n')


def save_dict2json(output_path, data):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)