import csv
import os
import numpy as np
import torch
import sys

sys.path.insert(0, '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/')
from model.utils import *


def save_slowfast_output(cfg, video_preds, video_labels):
    print("\nself.video_preds shape: ", video_preds.shape)

    save_path = os.path.join(cfg.OUTPUT_DIR, "y_prob_video_preds.csv")
    np.savetxt(save_path, video_preds.cpu().detach().numpy(), delimiter=',')

    print("self.video_labels shape: ", video_labels.shape)
    print(video_labels)

    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(video_preds)
    save_path = os.path.join(cfg.OUTPUT_DIR, "y_prob_output.csv")
    np.savetxt(save_path, probabilities.cpu().detach().numpy(), delimiter=',')

    print("probabilities shape: ", probabilities.shape)
    print("torch.sum(probabilities): ", torch.sum(probabilities), "\n")
    print(probabilities)


def save_slowfast_metrics(cfg, ks, topks):
    # get name and path
    run_name = os.path.basename(os.path.normpath(cfg.OUTPUT_DIR))
    output_path = os.path.dirname(os.path.normpath(cfg.OUTPUT_DIR))

    # get data
    row_data = {
        'run_name': run_name
    }
    for k, topk in zip(ks, topks):
        row_data["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)

    # create accuracy report csv file
    create_accuracy_file_at_runtime(output_path, row_data)
