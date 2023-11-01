import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
from comet_ml import Experiment, Artifact
import datetime as dt
from datetime import date
from datetime import datetime


def plot_plotly(df_out, title, today_date, today_date_hr, experiment):
    length = len(df_out)
    pio.templates.default = "seaborn"
    plot_template = dict(
        layout=go.Layout(
            {"font_size": 18, "xaxis_title_font_size": 24, "yaxis_title_font_size": 24}
        )
    )

    fig = px.line(
        df_out,
        labels=dict(created_at="Date", value="Forecast Error"),
        title=f"{title}",
        width=1200,
        height=400,
    )

    fig.add_vline(x=(length * 0.2), line_width=4, line_dash="dash")
    fig.add_annotation(
        xref="paper",
        x=0.2,
        yref="paper",
        y=0.8,
        text="Test set start",
        showarrow=False,
    )
    fig.update_layout(
        template=plot_template, legend=dict(orientation="h", y=1.02, title_text="")
    )

    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/"
        )

    os.mkdir(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}_{today_date_hr}/"
    )
    fig.write_image(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}_{today_date_hr}/{title}.png"
    )

    artifact3 = Artifact(name="data_output", artifact_type="line plot")
    artifact3.add(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}_{today_date_hr}/{title}.png"
    )
    experiment.log_artifact(artifact3)