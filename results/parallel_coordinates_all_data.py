#!/usr/bin/env python3

import plotly.express as px
from plotly.offline import plot
import pandas as pd


def pcp(df):
    data_value = [];
    for i, value in enumerate(df["data"].unique()):
        df.replace(value, str(i), True);
        data_value.append(value);
    df['data'] = pd.to_numeric(df['data']);

    normalisation_value = [];
    for i, value in enumerate(df["normalisation"].unique()):
        df.replace(value, str(i), True);

        if value == True:
            normalisation_value.append("With");
        else:
            normalisation_value.append("Without");

    df['normalisation'] = pd.to_numeric(df['normalisation']);

    metrics_tics = [];
    metrics_value = [];
    for i, value in enumerate(df["metrics"].unique()):
        df.replace(value, str(i), True);
        metrics_tics.append(i);
        metrics_value.append(value);
    df['metrics'] = pd.to_numeric(df['metrics']);

    fig = px.parallel_coordinates(df,
        color="metrics",
        dimensions=df.head());

    fig.layout.coloraxis.colorbar['tickvals'] = metrics_tics;
    fig.layout.coloraxis.colorbar['ticktext'] = metrics_value;

    for dim in fig.data[0]['dimensions']:
        if dim['label'] == "data":
            index = [];
            values = [];

            for i, val in enumerate(data_value):
                index.append(i);
                values.append(val)

            dim['tickvals'] = index;
            dim['ticktext'] = values;

        elif dim['label'] == "metrics":
            index = [];
            values = [];

            for i, val in enumerate(metrics_value):
                index.append(i);
                values.append(val)

            dim['tickvals'] = index;
            dim['ticktext'] = values;

        elif dim['label'] == "normalisation":
            index = [];
            values = [];

            for i, val in enumerate(normalisation_value):
                index.append(i);
                values.append(val)

            dim['tickvals'] = index;
            dim['ticktext'] = values;

    return fig;



def extract_columns(df):
    small_df = pd.DataFrame();
    small_df["data"] = df["DATA"];
    small_df["normalisation"] = df["NORMALISATION"];
    small_df["metrics"] = df["METRICS"];

    small_df["ZNCC"] = df["LAPLACIAN_LSF_ZNCC"];
    small_df["Runtime (in min)"] = df["OVERALL_RUNTIME (in min)"];
    small_df["r W (in um)"] = df["LAPLACIAN1_RADIUS_CORE (in um)"].round(0).astype(int);
    small_df["r SiC (in um)"] = df["LAPLACIAN1_RADIUS_FIBRE (in um)"].round(0).astype(int);
    small_df["mu_W"] = df["MEAN_CORE_SIM"].round(0).astype(int);
    small_df["mu_SiC"] = df["MEAN_FIBRE_SIM"].round(0).astype(int);
    small_df["mu_Ti90Al6V4"] = df["MEAN_MATRIX_SIM"].round(0).astype(int);

    return small_df;


df = pd.read_csv("summary-bis.csv");

small_df_all_data = extract_columns(df);
small_df_all_data.to_csv("summary_all_data-small.csv");

fig_all_data = pcp(small_df_all_data);

fig_all_data.update_layout(
    font_size=18
)

fig_all_data.write_html("parallel_coordinates_all_data.html")
fig_all_data.show();

