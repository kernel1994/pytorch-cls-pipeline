import shutil

import torch

import cfg
import data
import evaluate
import models
import tester
import trainer


def main():
    # data
    train_dataloader, test_dataloader = data.mnist_dataloader()

    # model
    model = models.CNN_Net().to(cfg.device)

    # train
    trainer.train(model, train_dataloader, test_dataloader)

    # test
    model.load_state_dict(torch.load(cfg.best_model_path))
    tester.test(model, test_dataloader)

    # metrics
    evaluate.eval()


if __name__ == "__main__":
    shutil.rmtree(cfg.main_path)
    cfg.main_path.mkdir()

    main()
