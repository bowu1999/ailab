import argparse
import importlib.util
import torch
from src.utils.inference import inference

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c','--config', default='configs/train.py')
    p.add_argument('-w','--ckpt', required=True)
    p.add_argument('-i','--input', required=True)
    p.add_argument('-o','--output', default=None)
    return p.parse_args()

def load_cfg(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cfg

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    outputs = inference(cfg, args.ckpt, args.input)

    if args.output:
        torch.save(outputs, args.output)
        print("Results saved to", args.output)
    else:
        print("Inference outputs:", outputs)

if __name__ == '__main__':
    main()
