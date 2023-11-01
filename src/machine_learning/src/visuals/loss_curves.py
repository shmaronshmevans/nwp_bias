# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import datetime
import os


def loss_curves(train_loss_ls, test_loss_ls, today_date, title, today_date_hr):
    fig, ax = plt.subplots(figsize=(15, 15))
    x = np.arange(0, len(train_loss_ls))
    p1 = plt.plot(x, train_loss_ls, c="blue")
    p2 = plt.plot(x, test_loss_ls, c="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}_{today_date_hr}/{title}_loss.png"
    )
