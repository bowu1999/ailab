def read_image_paths_from_file(file_path):
    """
    从指定的文本文件中读取图像文件路径。
    
    :param file_path: 包含图像路径的文本文件的路径。
    :return: 包含所有图像文件路径的列表。
    """
    image_paths = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行末的换行符，并添加到列表中
                path = line.strip()
                if path:  # 确保路径非空
                    image_paths.append(path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return image_paths


# 使用示例
if __name__ == "__main__":
    file_path = "path/to/your/file.txt"  # 替换为你的文件路径
    paths = read_image_paths_from_file(file_path)
    
    if paths:
        print(f"Total number of image paths: {len(paths)}")
        # 打印前5个路径作为示例
        for i, path in enumerate(paths[:5]):
            print(f"Path {i+1}: {path}")
    else:
        print("No valid image paths found.")