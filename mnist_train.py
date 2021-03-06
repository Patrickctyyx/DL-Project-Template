# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_loaders.mnist_dl import MnistDL
from models.mnist_model import MnistModel
from trainers.mnist_trainer import MnistTrainer
from utils.config_utils import process_config, get_train_args
import numpy as np


def mnist_train():

    print('[INFO] 解析配置…')
    parser = None
    config = None
    model_path = None

    try:
        args, parser = get_train_args()
        config = process_config(args.config)
        model_path = args.pre_train
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] 参考: python main_train.py -c configs/simple_mnist_config.json')
        exit(0)

    np.random.seed(47)

    print('[INFO] 加载数据…')
    dl = MnistDL(config=config)

    print('[INFO] 构造网络…')
    if model_path != 'None':
        model = MnistModel(config=config, model_path=model_path)
    else:
        model = MnistModel(config=config)

    print('[INFO] 训练网络')
    trainer = MnistTrainer(
        model=model.model,
        data=[dl.get_train_data(), dl.get_test_data()],
        config=config
    )
    trainer.train()
    print('[INFO] 训练完成…')


if __name__ == "__main__":
    mnist_train()
