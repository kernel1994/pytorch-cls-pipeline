import pandas as pd
import torch
import torch.nn.functional as F

import cfg


def test(model, dataloader):
    model.eval()  # set model to evaluation mode

    # header: truth, predict, score0, score1...
    dfs = pd.DataFrame(columns=cfg.csv_header)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(cfg.device)
            labels = labels.long().to(cfg.device)

            outputs = model(images)
            outputs_prob = F.softmax(outputs, dim=1)
            outputs_label = outputs.argmax(dim=1)

            # construct DataFrame to save on disk
            truth_df = pd.DataFrame(labels.cpu().numpy(), columns=[cfg.csv_header[0]])
            predict_df = pd.DataFrame(outputs_label.detach().cpu().numpy(), columns=[cfg.csv_header[1]])
            score_df = pd.DataFrame(outputs_prob.detach().cpu().numpy(), columns=cfg.csv_header[2:])

            df = pd.concat([truth_df, predict_df, score_df], axis=1)
            dfs = dfs.append(df, ignore_index=True)

    dfs.to_csv(cfg.prediction, index=False, encoding='utf-8')
