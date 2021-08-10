#!/usr/bin/env python3

# Import math Library
import math

import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import numpy as np
from skimage.util import compare_images

import scipy.stats as stats
from skimage.util import compare_images

import SimpleITK as sitk

NoneType = type(None)
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
# # fake up some data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 50
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# data = np.concatenate((spread, center, flier_high, flier_low))
#
# fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
# ax1.boxplot(data)
#
#
# plt.show()
# exit();
def printZNCC(column, best_objective, equivalent_objective, p_value_set):

    print("\\begin{tabular}{c|c|c||c|c}")
    print(" & \multicolumn{2}{c||}{\\textbf{Projections (after flat-field correction)}} &  \multicolumn{2}{c}{\\textbf{Sinogram}} \\\\")
    print(" & \\textbf{With} &  \\textbf{Without} & \\textbf{With} &  \\textbf{Without} \\\\")
    print(" & \\textbf{normalisation} &  \\textbf{normalisation} & \\textbf{normalisation} & \\textbf{normalisation} \\\\")
    print(" \hline")
    print(" \hline")

    for metrics in METRICS:
        row = [];

        row.append(metrics);
        for data in DATA:
            for normalisation in NORMALISATION:
                objective = "FULL_REGISTRATION_" + data + "_" + normalisation + "_" + metrics;

                selection = df["objective"] == objective;

                if df[selection].count()[0] != 0:

                    if objective == best_objective and len(equivalent_objective) == 1:
                        row.append("\mathbf{" + "{:.2f}".format(df[selection][column].mean()) + "\% \pm " + "{:.2f}".format(df[selection][column].std(ddof=0)) + "\%}")
                    else:
                        row.append("{:.2f}".format(df[selection][column].mean()) + "\% \pm " + "{:.2f}".format(df[selection][column].std(ddof=0)) + "\%")

                        if objective == best_objective or objective in equivalent_objective:
                            row[-1] += " ^*";

                    #row.append("%.2f" % ())# + "\%\\pm" + "%.2f" % (df[selection]["MATRIX_ZNCC"].std(ddof=0)) + "\%");
                else:
                    row.append("\\text{N/A}");

        print("\\textbf{" + row[0] + "}\t&\t$" + row[1] + "$\t&\t$" + row[2] + "$\t&\t$" + row[3] + "$\t&\t$" + row[4] + "$\\\\")

    print("\end{tabular}")

df = pd.read_csv("summary.csv", index_col=False);
df_desktop = pd.read_csv("desktop/summary-desktop.csv", index_col=False);

old_colums = [];
for col in df.head():
    old_colums.append(col);



new_colums = ["DATA", "NORMALISATION", "METRICS"];
new_colums += old_colums;

test = df['objective'].str.contains("SINOGRAM");
test[test == True] = "sinogram"
test[test == False] = "projections"
df["DATA"] = test;

test = df_desktop['objective'].str.contains("SINOGRAM");
test[test == True] = "sinogram"
test[test == False] = "projections"
df_desktop["DATA"] = test;



test = ~df['objective'].str.contains("NOT_NORMALISED");
df["NORMALISATION"] = test;

test = ~df_desktop['objective'].str.contains("NOT_NORMALISED");
df_desktop["NORMALISATION"] = test;



test = df['objective'].str.contains("_MAE");
test[test == True] = "MAE";

test1 = df['objective'].str.contains("_DSSIM");
test[test1 == True] = "DSSIM";

test1 = df['objective'].str.contains("_RMSE");
test[test1 == True] = "RMSE";

test1 = df['objective'].str.contains("_ZNCC");
test[test1 == True] = "ZNCC";

df["METRICS"] = test;



test = df_desktop['objective'].str.contains("_MAE");
test[test == True] = "MAE";

test1 = df_desktop['objective'].str.contains("_DSSIM");
test[test1 == True] = "DSSIM";

test1 = df_desktop['objective'].str.contains("_RMSE");
test[test1 == True] = "RMSE";

test1 = df_desktop['objective'].str.contains("_ZNCC");
test[test1 == True] = "ZNCC";

df_desktop["METRICS"] = test;



df=df[new_colums];
df_desktop=df_desktop[new_colums];

METRICS=["MAE", "RMSE", "ZNCC", "DSSIM"];
DATA=["PROJS", "SINOGRAM"];
NORMALISATION=["PARTIAL_NORMALISED", "PARTIAL_NOT_NORMALISED"]

# df.drop("objective")
df.to_csv("summary-bis.csv", index=False);
df_desktop.to_csv("desktop/summary-desktop-bis.csv", index=False);

def printDuration(column):
    print("\\begin{tabular}{c|c|c||c|c}")
    print(" & \multicolumn{2}{c|}{\\textbf{Projections (after flat-field correction)}} &  \multicolumn{2}{c}{\\textbf{Sinogram}} \\\\")
    print(" & \\textbf{With} & \\textbf{Without} & \\textbf{With} & \\textbf{Without} \\\\")
    print(" & \\textbf{normalisation} & \\textbf{normalisation} & \\textbf{normalisation} & \\textbf{normalisation} \\\\")
    print(" \hline")
    print(" \hline")

    for metrics in METRICS:
        row = [];

        row.append(metrics);
        for data in DATA:
            for normalisation in NORMALISATION:
                objective = "FULL_REGISTRATION_" + data + "_" + normalisation + "_" + metrics;

                selection = df["objective"] == objective;

                if df[selection].count()[0] != 0:
                    row.append("{:.2f}".format(df[selection][column].mean()) + " \pm " + "{:.2f}".format(df[selection][column].std(ddof=0)))
                    #row.append("%.2f" % ())# + "\%\\pm" + "%.2f" % (df[selection]["MATRIX_ZNCC"].std(ddof=0)) + "\%");
                else:
                    row.append("\\text{N/A}");

        print("\\textbf{" + row[0] + "}\t&\t$" + row[1] + "$\t&\t$" + row[2] + "$\t&\t$" + row[3] + "$\t&\t$" + row[4] + "$\\\\")

    print("\end{tabular}")


def boxplot(column, column_order, ylabel, multiplier=1.0, limits=None):

    fig1, ax1 = plt.subplots()
    #ax1.set_title(column)

    i = 1;
    y_data = [];
    labels = [];
    ticks = [];
    median = [];

    max_value = -math.inf;
    min_value = math.inf;
    colors = []

    for metrics in METRICS:
        row = [];
        row.append(metrics);
        for data in DATA:
            for normalisation in NORMALISATION:
                objective = "FULL_REGISTRATION_" + data + "_" + normalisation + "_" + metrics;

                selection = df["objective"] == objective;

                if df[selection].count()[0] != 0:

                    if data == "PROJS":
                        text_data = "projections";
                    else:
                        text_data = "sinogram";

                    if normalisation == "NORMALISED":
                        text_normalisation = "with full normalisation";
                    elif normalisation == "PARTIAL_NORMALISED":
                        text_normalisation = "with";
                        colors.append('pink')
                    else:
                        text_normalisation = "without";
                        colors.append('lightblue')

                    min_value = min(min_value, multiplier * df[selection][column].min());
                    max_value = max(max_value, multiplier * df[selection][column].max());
                    y_data.append(multiplier * df[selection][column]);
                    labels.append(text_normalisation);
                    ticks.append(i);
                    #median.append(df[selection][column_order].median());
                    i += 1;

    order = np.argsort(median)[::-1];
    y_data = np.array(y_data);
    ticks = np.array(ticks);
    labels = np.array(labels);

    #ax1.boxplot((y_data.T)[order])
    bplot = ax1.boxplot((y_data.T),
        notch=True,
        # showfliers=True,
        # vert=True,  # vertical box alignment
        patch_artist=True)  # fill with color)

    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        print(color)
        patch.set_facecolor(color)
        patch.set_edgecolor(color)

    #plt.xticks(ticks=ticks, labels=labels[order], rotation=45, ha='right');
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha='right');
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel(ylabel);
    # plt.title(column);


    if not isinstance(limits, NoneType):
        plt.ylim(limits)

    xcoords = [4.5, 8.5, 10.5]
    for xc in xcoords:
        plt.axvline(x=xc,
            # ymin=ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.01,
            # ymax=ax1.get_ylim()[1] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.01,
            linewidth=1,
            color='gray',
            linestyle="--")

    xcoords = [2.5, 6.5, 9.5, 12.5]
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=.5, color='gray', linestyle=":")

    ax1.get_ylim()[1]

    plt.text(2.5, ax1.get_ylim()[1] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "MAE", ha='center')
    plt.text(6.5, ax1.get_ylim()[1] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "RMSE", ha='center')
    plt.text(9.5, ax1.get_ylim()[1] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "ZNCC", ha='center')
    plt.text(12.5, ax1.get_ylim()[1] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "DSSIM", ha='center')

    plt.text(1.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Proj", ha='center')
    plt.text(3.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Sino", ha='center')
    plt.text(5.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Proj", ha='center')
    plt.text(7.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Sino", ha='center')
    plt.text(9, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Proj", ha='center')
    plt.text(10, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Sino", ha='center')
    plt.text(11.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Proj", ha='center')
    plt.text(13.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015, "Sino", ha='center')

    custom_lines = [Line2D([0], [0], color="pink", lw=4),
                    Line2D([0], [0], color="lightblue", lw=4)]

    plt.legend(custom_lines,
        ['With zero-mean, unit variance normalisation', 'Without'],
        loc='lower center',
        bbox_to_anchor=(.5, -0.15))

    ax1.get_xaxis().set_visible(False)

    fname = column.replace(" ", "_");
    fname = fname.replace("(", "");
    fname = fname.replace(")", "");
    fname = fname + ".pdf";

    plt.tight_layout();
    plt.savefig(fname);

def ttest(column):
    best_zncc = -1;
    best_objective = None;
    best_zncc_set = None;

    # Find the best set of parameters
    for metrics in METRICS:
        for data in DATA:
            for normalisation in NORMALISATION:
                objective = "FULL_REGISTRATION_" + data + "_" + normalisation + "_" + metrics;

                selection = df["objective"] == objective;

                if df[selection].count()[0] != 0:

                    if best_zncc < df[selection][column].mean():
                        best_zncc = df[selection][column].mean();
                        best_objective = objective;
                        best_zncc_set = df[selection][column];

    # Use two-sample T-test to test whether two data samples have different means.
    # Here, we take the null hypothesis that both groups have equal means.
    #print(best_metrics, best_data, best_normalisation)
    equivalent_objective = [];
    p_value_set = [];

    header = [];
    row = [];

    header = best_objective;
    row.append(0);

    for metrics in METRICS:
        for data in DATA:
            for normalisation in NORMALISATION:
                objective = "FULL_REGISTRATION_" + data + "_" + normalisation + "_" + metrics;



                selection = df["objective"] == objective;

                if df[selection].count()[0] != 0:
                    p_value =  stats.ttest_ind(a=best_zncc_set, b=df[selection][column], equal_var=False)[1];

                    header += "," + objective;
                    row.append(p_value);

                    if p_value >= 0.05:
                        equivalent_objective.append(objective);
                        p_value_set.append(p_value);
                        #print("\t", metrics, data, normalisation, p_value);

    np.savetxt(column + ".csv", np.array([row]), delimiter=',', header=header);

    return best_objective, equivalent_objective, p_value_set;

def drawProfiles(objectives, colours, labels):
    fig1, ax = plt.subplots()
    ref_not_plotted = False;
    for objective, colour, label in zip(objectives, colours, labels):

        selection = df["objective"] == objective;
        idxmin = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmin();
        idxmax = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmax();

        # if df[selection]['LAPLACIAN_LSF_ZNCC'].count() != 15:
        #     print("MISSING DATA FOR", objective)
        #     exit();
        median_value = df[selection]['LAPLACIAN_LSF_ZNCC'].median();
        for row_id, row_ZNCC in zip(df[selection]['i'], df[selection]['LAPLACIAN_LSF_ZNCC']):

            if row_ZNCC == median_value:
                idxmedian = row_id;

        offset = 30;

        print(objective, df["i"][idxmin], idxmedian, df["i"][idxmax])
        if not ref_not_plotted:
            ref_CT = sitk.ReadImage("../tutorial/fbp_scipy_recons.mha")
            ground_truth = sitk.GetArrayFromImage(ref_CT)[0]

            ref = np.diag(ground_truth[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1])
            plt.plot(np.array(range(len(ref))) * 1.9, ref, "k-", label="Real CT");
            ref_not_plotted = True;

        # Load simulated slice
        simulated_CT = sitk.ReadImage(objective + "/run_SCW_" + str(df["i"][idxmin]) + "/simulated_CT_before_noise.mha")
        sim = np.diag(sitk.GetArrayFromImage(simulated_CT)[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1])
        plt.plot(np.array(range(len(sim))) * 1.9, sim, "--", label="Worse run for " + label, color="#003f5c");

        simulated_CT = sitk.ReadImage(objective + "/run_SCW_" + str(df["i"][idxmedian]) + "/simulated_CT_before_noise.mha")
        sim = np.diag(sitk.GetArrayFromImage(simulated_CT)[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1])
        plt.plot(np.array(range(len(sim))) * 1.9, sim, ":", label="Median run for " + label, color="#bc5090");

        simulated_CT = sitk.ReadImage(objective + "/run_SCW_" + str(df["i"][idxmax]) + "/simulated_CT_before_noise.mha")
        sim = np.diag(sitk.GetArrayFromImage(simulated_CT)[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1])
        plt.plot(np.array(range(len(sim))) * 1.9, sim, "-.", label="Best run for " + label, color="#ffa600");

    #plt.yscale("log")
    #plt.ylim((ref.min(), ref.max()))

    plt.xlabel("Distance (in $\mathrm{\mu}$m)");
    plt.ylabel("Linear attenuation coefficients (in cm$^{-1}$)");

    plt.legend(loc='best');

    plt.tight_layout();
    plt.savefig("profiles.pdf");

    # plt.show()

def drawScatterPlots(objectives, colours, labels):



    plt.figure(figsize=(20,10))
    plt.xlabel("ZNCC (in %)");
    plt.ylabel("Runtime (in min)");

    for i in range(3):
        objective = objectives[i]
        colour = colours[i]
        label = labels[i]

        selection = df["objective"] == objective;
        selection_desktop = df_desktop["objective"] == objective;

        plt.scatter(df[selection]["LAPLACIAN_LSF_ZNCC"], df[selection]["OVERALL_RUNTIME (in min)"], marker="o", color=colour, label=label + " (SCW)")
        plt.scatter(df_desktop[selection_desktop]["LAPLACIAN_LSF_ZNCC"], df_desktop[selection_desktop]["OVERALL_RUNTIME (in min)"], marker="^", color=colour, label=label + " (PC)")



    plt.legend(loc='upper left');


    plt.tight_layout();
    plt.savefig("scatter_plot.pdf");


drawProfiles(["FULL_REGISTRATION_SINOGRAM_PARTIAL_NORMALISED_RMSE"],
        ["g"],
        ["RMSE on $\mathbf{Sino}$\nwith normalisation"])



print("********************************************************************************")
print("Matrix")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("MATRIX_ZNCC");
boxplot("MATRIX_ZNCC", "MATRIX_ZNCC", "ZNCC in %", 1.0);
printZNCC("MATRIX_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("CUBE1_RUNTIME (in min)", "MATRIX_ZNCC", "Runtime in minutes", 1.0);
printDuration("CUBE1_RUNTIME (in min)");
print()
print()



print("********************************************************************************")
print("Radii")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("FIBRE1_ZNCC");
boxplot("FIBRE1_ZNCC", "FIBRE1_ZNCC", "ZNCC in %", 1.0);
printZNCC("FIBRE1_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("FIBRES1_RUNTIME (in min)", "FIBRE1_ZNCC", "Runtime in minutes", 1.0);
printDuration("FIBRES1_RUNTIME (in min)");
print()
print()


print("********************************************************************************")
print("Recentring")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("FIBRE2_ZNCC");
boxplot("FIBRE2_ZNCC", "FIBRE2_ZNCC", "ZNCC in %", 1.0);
printZNCC("FIBRE2_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()



print("********************************************************************************")
print("Radii again")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("FIBRE3_ZNCC");
boxplot("FIBRE3_ZNCC", "FIBRE3_ZNCC", "ZNCC in %", 1.0);
printZNCC("FIBRE3_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("FIBRES3_RUNTIME (in min)", "FIBRE3_ZNCC", "Runtime in minutes", 1.0);
printDuration("FIBRES3_RUNTIME (in min)");
print()
print()




print("********************************************************************************")
print("Harmonics")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("HARMONICS_ZNCC");
boxplot("HARMONICS_ZNCC", "HARMONICS_ZNCC", "ZNCC in %", 1.0);
printZNCC("HARMONICS_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("HARMONICS_RUNTIME (in min)", "HARMONICS_ZNCC", "Runtime in minutes", 1.0);
printDuration("HARMONICS_RUNTIME (in min)");
print()
print()

print("********************************************************************************")
print("Noise")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("NOISE_ZNCC");
boxplot("NOISE_ZNCC", "NOISE_ZNCC", "ZNCC in %", 1.0);
printZNCC("NOISE_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("NOISE_RUNTIME (in min)", "NOISE_ZNCC", "Runtime in minutes", 1.0);
printDuration("NOISE_RUNTIME (in min)");
print()
print()

print("********************************************************************************")
print("Phase contrast")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("LAPLACIAN1_ZNCC");
boxplot("LAPLACIAN1_ZNCC", "LAPLACIAN1_ZNCC", "ZNCC in %", 1.0);
printZNCC("LAPLACIAN1_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("LAPLACIAN1_RUNTIME (in min)", "LAPLACIAN1_ZNCC", "Runtime in minutes", 1.0);
printDuration("LAPLACIAN1_RUNTIME (in min)");
print()
print()



print("********************************************************************************")
print("LSF + Phase contrast")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("LAPLACIAN_LSF_ZNCC");
boxplot("LAPLACIAN_LSF_ZNCC", "LAPLACIAN_LSF_ZNCC", "ZNCC in %", 1.0, (80,95));
printZNCC("LAPLACIAN_LSF_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("LAPLACIAN_LSF_RUNTIME (in min)", "LAPLACIAN_LSF_ZNCC", "Runtime in minutes", 1.0);
printDuration("LAPLACIAN_LSF_RUNTIME (in min)");
print()
print()

boxplot("OVERALL_RUNTIME (in min)", "OVERALL_RUNTIME", "Runtime in minutes", 1.0);
printDuration("OVERALL_RUNTIME (in min)");





SMALL_SIZE = 13
MEDIUM_SIZE = 14
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# fig1, ax = plt.subplots(gridspec_kw = {'wspace':100, 'hspace': 100})

drawScatterPlots(["FULL_REGISTRATION_SINOGRAM_PARTIAL_NORMALISED_RMSE", "FULL_REGISTRATION_PROJS_PARTIAL_NORMALISED_DSSIM", "FULL_REGISTRATION_SINOGRAM_PARTIAL_NORMALISED_ZNCC"],
        ["#1b9e77", "#d95f02", "#7570b3"],
        ["RMSE on $\mathbf{Sino}$ with normalisation", "DSSIM on $\mathbf{Proj}$ with normalisation", "ZNCC on $\mathbf{Sino}$"])



ref_CT = sitk.ReadImage("../tutorial/fbp_scipy_recons.mha")
ground_truth = sitk.GetArrayFromImage(ref_CT)[0]

norm = cm.colors.Normalize(vmax=30, vmin=-20)

objective = "FULL_REGISTRATION_SINOGRAM_PARTIAL_NORMALISED_RMSE"
label = "RMSE on sinogram\nwith normalisation"

selection = df["objective"] == objective;

idxmin = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmin();
idxmax = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmax();

# if df[selection]['LAPLACIAN_LSF_ZNCC'].count() != 15:
#     print("MISSING DATA FOR", objective)
#     exit();
median_value = df[selection]['LAPLACIAN_LSF_ZNCC'].median();
for row_id, row_ZNCC in zip(df[selection]['i'], df[selection]['LAPLACIAN_LSF_ZNCC']):

    print(row_ZNCC, median_value)
    if row_ZNCC == median_value:
        idxmedian = row_id;

print(objective)
print("\t", df["i"][idxmin], df[selection]["LAPLACIAN_LSF_ZNCC"].min())
print("\t", df["i"][idxmedian], df[selection]["LAPLACIAN_LSF_ZNCC"].median())
print("\t", df["i"][idxmax], df[selection]["LAPLACIAN_LSF_ZNCC"].max())

fig, axes = plt.subplots(nrows=1, ncols=3)

SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

norm = cm.colors.Normalize(vmax=50, vmin=0)

fname = objective + "/run_SCW_" + str(df["i"][idxmin]) + "/simulated_CT_before_noise.mha";
simulated_CT = sitk.GetArrayFromImage(sitk.ReadImage(fname))
checkerboard = compare_images(ground_truth, simulated_CT, method='checkerboard');
# diff = compare_images(ground_truth, simulated_CT, method='diff');
# relative_error = 100 * (ground_truth - simulated_CT) / ground_truth

axes.flat[0].set_title('Worse run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].min()) + "%");
img0 = axes.flat[0].imshow(checkerboard, cmap='gray', norm=norm)
axes.flat[0].set_xticks([])
axes.flat[0].set_yticks([])
axes.flat[0].axis('off')

divider = make_axes_locatable(axes.flat[0])
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.set_xticks([])
ax_cb.set_yticks([])
ax_cb.axis('off')
fig.add_axes(ax_cb)

# img3 = axes.flat[3].imshow(relative_error, cmap="RdBu", vmin=-100, vmax=100)
# axes.flat[3].set_xticks([])
# axes.flat[3].set_yticks([])
# axes.flat[3].axis('off')
#
# divider = make_axes_locatable(axes.flat[3])
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# ax_cb.set_xticks([])
# ax_cb.set_yticks([])
# ax_cb.axis('off')
# fig.add_axes(ax_cb)

fname = objective + "/run_SCW_" + str(df["i"][idxmedian]) + "/simulated_CT_before_noise.mha";
simulated_CT = sitk.GetArrayFromImage(sitk.ReadImage(fname))
checkerboard = compare_images(ground_truth, simulated_CT, method='checkerboard');
# diff = compare_images(ground_truth, simulated_CT, method='diff');
# relative_error = 100 * (ground_truth - simulated_CT) / ground_truth

axes.flat[1].set_title('\nMedian run, ZNCC: ' + "{:.2f}".format(median_value) + "%");
img1 = axes.flat[1].imshow(checkerboard, cmap='gray', norm=norm)
axes.flat[1].set_xticks([])
axes.flat[1].set_yticks([])
axes.flat[1].axis('off')

divider = make_axes_locatable(axes.flat[1])
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.set_xticks([])
ax_cb.set_yticks([])
ax_cb.axis('off')
fig.add_axes(ax_cb)

# img4 = axes.flat[4].imshow(relative_error, cmap="RdBu", vmin=-100, vmax=100)
# axes.flat[4].set_xticks([])
# axes.flat[4].set_yticks([])
# axes.flat[4].axis('off')
#
# divider = make_axes_locatable(axes.flat[4])
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# ax_cb.set_xticks([])
# ax_cb.set_yticks([])
# ax_cb.axis('off')
# fig.add_axes(ax_cb)

fname = objective + "/run_SCW_" + str(df["i"][idxmax]) + "/simulated_CT_before_noise.mha";
simulated_CT = sitk.GetArrayFromImage(sitk.ReadImage(fname))
checkerboard = compare_images(ground_truth, simulated_CT, method='checkerboard');
# diff = compare_images(ground_truth, simulated_CT, method='diff');
# relative_error = 100 * (ground_truth - simulated_CT) / ground_truth

axes.flat[2].set_title('\nBest run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].max()) + "%");
img2 = axes.flat[2].imshow(checkerboard, cmap='gray', norm=norm)
axes.flat[2].set_xticks([])
axes.flat[2].set_yticks([])
axes.flat[2].axis('off')

divider = make_axes_locatable(axes.flat[2])
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
plt.colorbar(img2, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

# img5 = axes.flat[5].imshow(relative_error, cmap="RdBu", vmin=-100, vmax=100)
# axes.flat[5].set_xticks([])
# axes.flat[5].set_yticks([])
# axes.flat[5].axis('off')
#
# divider = make_axes_locatable(axes.flat[5])
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# fig.add_axes(ax_cb)
# plt.colorbar(img5, cax=ax_cb)
# ax_cb.yaxis.tick_right()
# ax_cb.yaxis.set_tick_params(labelright=True)


# Create a Rectangle patch
offset_x = int(checkerboard.shape[1] / 8)
offset_y = int(checkerboard.shape[0] / 8)

for i in range(3):
    rect1 = patches.Rectangle((0, 0), offset_x - 1, offset_y - 1, linewidth=0.1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((offset_x+1, offset_y+1), offset_x - 1, offset_y - 1, linewidth=0.1, edgecolor='r', facecolor='none')
    rect3 = patches.Rectangle((0, offset_y + 1), offset_x - 1, offset_y-1, linewidth=0.1, edgecolor='b', facecolor='none')
    rect4 = patches.Rectangle((offset_x+1, 0), offset_x - 1, offset_y, linewidth=0.1, edgecolor='b', facecolor='none')

    # Add the patches to the Axes
    axes.flat[i].add_patch(rect1)
    axes.flat[i].add_patch(rect2)
    axes.flat[i].add_patch(rect3)
    axes.flat[i].add_patch(rect4)




# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.1, hspace=0.02)
#
# cb_ax = fig.add_axes([0.83, 0.5, 0.02, 0.15])
# cbar = fig.colorbar(img0, cax=cb_ax)
#
# cb_ax = fig.add_axes([0.83, 0.0, 0.02, 0.15])
# cbar = fig.colorbar(img3, cax=axes.flat[5])

# plt.subplots_adjust(wspace=0.5, hspace=0.3)
# set the spacing between subplots
# plt.subplots_adjust(left=0.,
#                     bottom=0.,
#                     right=0.5,
#                     top=0.9,
#                     wspace=0.,
#                     hspace=0.25)

# axes.set_xticks([])
# axes.set_yticks([])
# axes.axis('off')


# plt.tight_layout();
plt.savefig("checkerboard.pdf", dpi=600);



offset = 30;
fig, axes = plt.subplots(nrows=1, ncols=3)

SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


norm = cm.colors.Normalize(vmax=50, vmin=0)

fname = objective + "/run_SCW_" + str(df["i"][idxmin]) + "/simulated_CT_before_noise.mha";
simulated_CT = sitk.GetArrayFromImage(sitk.ReadImage(fname))[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1]
checkerboard = compare_images(ground_truth[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1], simulated_CT, method='checkerboard', n_tiles=(5, 5));
# diff = compare_images(ground_truth, simulated_CT, method='diff');
# relative_error = 100 * (ground_truth - simulated_CT) / ground_truth

axes.flat[0].set_title('Worse run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].min()) + "%");
img0 = axes.flat[0].imshow(checkerboard, cmap='gray', norm=norm)
axes.flat[0].set_xticks([])
axes.flat[0].set_yticks([])
axes.flat[0].axis('off')

divider = make_axes_locatable(axes.flat[0])
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.set_xticks([])
ax_cb.set_yticks([])
ax_cb.axis('off')
fig.add_axes(ax_cb)

# img3 = axes.flat[3].imshow(relative_error, cmap="RdBu", vmin=-100, vmax=100)
# axes.flat[3].set_xticks([])
# axes.flat[3].set_yticks([])
# axes.flat[3].axis('off')
#
# divider = make_axes_locatable(axes.flat[3])
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# ax_cb.set_xticks([])
# ax_cb.set_yticks([])
# ax_cb.axis('off')
# fig.add_axes(ax_cb)

fname = objective + "/run_SCW_" + str(df["i"][idxmedian]) + "/simulated_CT_before_noise.mha";
simulated_CT = sitk.GetArrayFromImage(sitk.ReadImage(fname))[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1]
checkerboard = compare_images(ground_truth[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1], simulated_CT, method='checkerboard', n_tiles=(5, 5));
# diff = compare_images(ground_truth, simulated_CT, method='diff');
# relative_error = 100 * (ground_truth - simulated_CT) / ground_truth

axes.flat[1].set_title('\nMedian run, ZNCC: ' + "{:.2f}".format(median_value) + "%");
img1 = axes.flat[1].imshow(checkerboard, cmap='gray', norm=norm)
axes.flat[1].set_xticks([])
axes.flat[1].set_yticks([])
axes.flat[1].axis('off')

divider = make_axes_locatable(axes.flat[1])
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
ax_cb.set_xticks([])
ax_cb.set_yticks([])
ax_cb.axis('off')
fig.add_axes(ax_cb)

# img4 = axes.flat[4].imshow(relative_error, cmap="RdBu", vmin=-100, vmax=100)
# axes.flat[4].set_xticks([])
# axes.flat[4].set_yticks([])
# axes.flat[4].axis('off')
#
# divider = make_axes_locatable(axes.flat[4])
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# ax_cb.set_xticks([])
# ax_cb.set_yticks([])
# ax_cb.axis('off')
# fig.add_axes(ax_cb)

fname = objective + "/run_SCW_" + str(df["i"][idxmax]) + "/simulated_CT_before_noise.mha";
simulated_CT = sitk.GetArrayFromImage(sitk.ReadImage(fname))[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1]
checkerboard = compare_images(ground_truth[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1], simulated_CT, method='checkerboard', n_tiles=(5, 5));
# diff = compare_images(ground_truth, simulated_CT, method='diff');
# relative_error = 100 * (ground_truth - simulated_CT) / ground_truth

axes.flat[2].set_title('\nBest run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].max()) + "%");
img2 = axes.flat[2].imshow(checkerboard, cmap='gray', norm=norm)
axes.flat[2].set_xticks([])
axes.flat[2].set_yticks([])
axes.flat[2].axis('off')

divider = make_axes_locatable(axes.flat[2])
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
plt.colorbar(img2, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

# img5 = axes.flat[5].imshow(relative_error, cmap="RdBu", vmin=-100, vmax=100)
# axes.flat[5].set_xticks([])
# axes.flat[5].set_yticks([])
# axes.flat[5].axis('off')
#
# divider = make_axes_locatable(axes.flat[5])
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# fig.add_axes(ax_cb)
# plt.colorbar(img5, cax=ax_cb)
# ax_cb.yaxis.tick_right()
# ax_cb.yaxis.set_tick_params(labelright=True)


# Create a Rectangle patch
offset_x = 7
offset_y = 7
offset_x = int(checkerboard.shape[1] / 5)
offset_y = int(checkerboard.shape[0] / 5)

for i in range(3):
    rect1 = patches.Rectangle((-0.4, -0.4), offset_x -0.4, offset_y -0.4, linewidth=0.15, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((offset_x -0.4, offset_y -0.4), offset_x - 0.4, offset_y - 0.4, linewidth=0.15, edgecolor='r', facecolor='none')
    rect3 = patches.Rectangle((-0.4, offset_y -0.4), offset_x - 0.4, offset_y - 0.4, linewidth=0.15, edgecolor='b', facecolor='none')
    rect4 = patches.Rectangle((offset_x -0.4, -0.4), offset_x - 0.4, offset_y - 0.4, linewidth=0.15, edgecolor='b', facecolor='none')

    # Add the patches to the Axes
    axes.flat[i].add_patch(rect1)
    axes.flat[i].add_patch(rect2)
    axes.flat[i].add_patch(rect3)
    axes.flat[i].add_patch(rect4)




# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.1, hspace=0.02)
#
# cb_ax = fig.add_axes([0.83, 0.5, 0.02, 0.15])
# cbar = fig.colorbar(img0, cax=cb_ax)
#
# cb_ax = fig.add_axes([0.83, 0.0, 0.02, 0.15])
# cbar = fig.colorbar(img3, cax=axes.flat[5])

# plt.subplots_adjust(wspace=0.5, hspace=0.3)
# set the spacing between subplots
# plt.subplots_adjust(left=0.,
#                     bottom=0.,
#                     right=0.5,
#                     top=0.9,
#                     wspace=0.,
#                     hspace=0.25)

# axes.set_xticks([])
# axes.set_yticks([])
# axes.axis('off')

plt.savefig("checkerboard-fibre.pdf", dpi=600);





# plt.show()
