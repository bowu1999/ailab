import os
import argparse
import importlib.util
from typing import Any


class ConfigParser:
    def __init__(self, default_config: str = 'cfg.py'):
        """
        初始化 ConfigParser，解析命令行参数并加载配置文件

        Args:
            default_config (str): 默认配置文件路径
        示例:
            config_parser = ConfigParser()
            args = config_parser.get_args()
            A = args.A
            ...
        """
        self.args = self._parse_args(default_config)

    def _load_config(self, config_path: str) -> Any:
        """
        动态加载 .py 格式的配置文件

        Args:
            config_path (str): 配置文件路径

        Returns:
            module: 加载的模块对象
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")

        module_name = "config_module"
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        config = importlib.util.module_from_spec(spec) # type: ignore
        spec.loader.exec_module(config) # type: ignore
        return config

    def _parse_args(self, default_config: str) -> argparse.Namespace:
        """
        解析命令行参数并加载配置文件内容

        Returns:
            argparse.Namespace: 包含所有参数的对象
        """
        parser = argparse.ArgumentParser(description="加载配置文件并解析参数")
        parser.add_argument('-c', '--config', type=str, default=default_config, help='配置文件路径')
        args = parser.parse_args()

        config = self._load_config(args.config)

        # 将配置文件中的变量添加到 args 中
        for key in dir(config):
            if not key.startswith('__') and not key.endswith('__'):
                value = getattr(config, key)
                setattr(args, key.lower(), value)

        return args

    def get_args(self) -> argparse.Namespace:
        """
        获取解析后的参数对象

        Returns:
            argparse.Namespace: 参数对象
        """
        return self.args
