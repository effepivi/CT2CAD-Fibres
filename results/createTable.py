#!/usr/bin/env python3

# Import math Library
import math

import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

import scipy.stats as stats
from skimage.util import compare_images

import SimpleITK as sitk

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

    print("\\begin{tabular}{c|c|c|c|c|c|c}")
    print(" & \multicolumn{3}{c||}{\\textbf{Projections (after flat-field correction)}} &  \multicolumn{3}{c}{\\textbf{Sinogram}} \\\\")
    print(" & \\textbf{With} & \\textbf{With normalisation} & \\textbf{Without} & \\textbf{With} & \\textbf{With normalisation} & \\textbf{Without} \\\\")
    print(" & \\textbf{normalisation} & \\textbf{except for spectrum} & \\textbf{normalisation} & \\textbf{except for spectrum}& \\textbf{normalisation} & \\textbf{normalisation} \\\\")
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
                        row.append("\mathbf{" + "{:.2f}".format(100*df[selection][column].mean()) + "\% \pm " + "{:.2f}".format(100*df[selection][column].std(ddof=0)) + "\%}")
                    else:
                        row.append("{:.2f}".format(100*df[selection][column].mean()) + "\% \pm " + "{:.2f}".format(100*df[selection][column].std(ddof=0)) + "\%")

                        if objective == best_objective or objective in equivalent_objective:
                            row[-1] += " ^*";

                    #row.append("%.2f" % ())# + "\%\\pm" + "%.2f" % (100*df[selection]["MATRIX1_ZNCC"].std(ddof=0)) + "\%");
                else:
                    row.append("\\text{N/A}");

        print("\\textbf{" + row[0] + "}\t&\t$" + row[1] + "$\t&\t$" + row[2] + "$\t&\t$" + row[3] + "$\t&\t$" + row[4] + "$\t&\t$" + row[5] + "$\t&\t$" + row[6] + "$\\\\")

    print("\end{tabular}")

df = pd.read_csv("summary.csv");

old_colums = [];
for col in df.head():
    old_colums.append(col);

new_colums = ["DATA", "NORMALISATION", "METRICS"];
new_colums += old_colums;

test = df['objective'].str.contains("SINOGRAM");
test[test == True] = "sinogram"
test[test == False] = "projections"
df["DATA"] = test;

test = ~df['objective'].str.contains("NOT_NORMALISED");
# test[test == True] = 0;
# test[test == False] = 1;
df["NORMALISATION"] = test;

test = df['objective'].str.contains("_MAE");
test[test == True] = "MAE";

test1 = df['objective'].str.contains("_DSSIM");
test[test1 == True] = "DSSIM";

test1 = df['objective'].str.contains("_RMSE");
test[test1 == True] = "RMSE";

test1 = df['objective'].str.contains("_ZNCC");
test[test1 == True] = "ZNCC";

df["METRICS"] = test;

df=df[new_colums];

METRICS=["MAE", "RMSE", "ZNCC", "DSSIM"];
DATA=["PROJS", "SINOGRAM"];
NORMALISATION=["NORMALISED", "PARTIAL_NORMALISED", "NOT_NORMALISED"]


def printDuration(column):
    print("\\begin{tabular}{c|c|c|c|c}")
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
                    #row.append("%.2f" % ())# + "\%\\pm" + "%.2f" % (100*df[selection]["MATRIX1_ZNCC"].std(ddof=0)) + "\%");
                else:
                    row.append("\\text{N/A}");

        print("\\textbf{" + row[0] + "}\t&\t$" + row[1] + "$\t&\t$" + row[2] + "$\t&\t$" + row[3] + "$\t&\t$" + row[4] + "$\\\\")

    print("\end{tabular}")


def boxplot(column, column_order, ylabel, multiplier=1.0):

    fig1, ax1 = plt.subplots()
    #ax1.set_title(column)

    i = 1;
    y_data = [];
    labels = [];
    ticks = [];
    median = [];

    max_value = -math.inf;
    min_value = math.inf;

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
                        text_normalisation = "with partial normalisation";
                    else:
                        text_normalisation = "without normalisation";

                    min_value = min(min_value, multiplier * df[selection][column].min());
                    max_value = max(max_value, multiplier * df[selection][column].max());
                    y_data.append(multiplier * df[selection][column]);
                    labels.append(text_data + " " + text_normalisation);
                    ticks.append(i);
                    #median.append(df[selection][column_order].median());
                    i += 1;

    order = np.argsort(median)[::-1];
    y_data = np.array(y_data);
    ticks = np.array(ticks);
    labels = np.array(labels);

    #ax1.boxplot((y_data.T)[order])
    ax1.boxplot((y_data.T))
    #plt.xticks(ticks=ticks, labels=labels[order], rotation=45, ha='right');
    plt.xticks(ticks=ticks, labels=labels, rotation=45, ha='right');
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel(ylabel);
    # plt.title(column);

    xcoords = [4.5, 8.5, 10.5]
    for xc in xcoords:
        plt.axvline(x=xc)


    plt.text(2, max_value + (max_value - min_value) * 0.07, "MAE")
    plt.text(6, max_value + (max_value - min_value) * 0.07, "RMSE")
    plt.text(9, max_value + (max_value - min_value) * 0.07, "ZNCC")
    plt.text(12, max_value + (max_value - min_value) * 0.07, "DSSIM")

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




fig1, ax = plt.subplots()
ref_not_plotted = False;
for objective, colour, label in zip(["FULL_REGISTRATION_SINOGRAM_NORMALISED_RMSE", "FULL_REGISTRATION_PROJS_NORMALISED_DSSIM"],
        ["g", "r"],
        ["RMSE on sinogram\nwith normalisation", "DSSIM on projections\nwith normalisation"]):

    selection = df["objective"] == objective;
    idxmin = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmin();
    idxmax = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmax();

    if df[selection]['LAPLACIAN_LSF_ZNCC'].count() != 15:
        print("MISSING DATA FOR", objective)
        exit();
    median_value = df[selection]['LAPLACIAN_LSF_ZNCC'].median();
    for row_id, row_ZNCC in zip(df[selection]['i'], df[selection]['LAPLACIAN_LSF_ZNCC']):

        if row_ZNCC == median_value:
            idxmedian = row_id;

    print(objective, df["i"][idxmin], idxmedian, df["i"][idxmax])
    if not ref_not_plotted:
        fname = objective + "/run_SCW_" + str(df["i"][idxmin]) + "/laplacian-LSF_profile_reference_fibre_in_centre.txt";
        ref=np.loadtxt(fname);
        plt.plot(np.array(range(len(ref))) * 1.9, ref, "k-", label="Real CT");
        ref_not_plotted = True;

    fname = objective + "/run_SCW_" + str(df["i"][idxmin]) + "/laplacian-LSF_profile_simulated_fibre_in_centre.txt";
    sim=np.loadtxt(fname);
    plt.plot(np.array(range(len(sim))) * 1.9, sim, colour + "--", label="Worse run for " + label);

    fname = objective + "/run_SCW_" + str(idxmedian) + "/laplacian-LSF_profile_simulated_fibre_in_centre.txt";
    sim=np.loadtxt(fname);
    plt.plot(np.array(range(len(sim))) * 1.9, sim, colour + ":", label="Median run for " + label);

    fname = objective + "/run_SCW_" + str(df["i"][idxmax]) + "/laplacian-LSF_profile_simulated_fibre_in_centre.txt";
    sim=np.loadtxt(fname);
    plt.plot(np.array(range(len(sim))) * 1.9, sim, colour + "-.", label="Best run for " + label);


plt.xlabel("Distance (in $\mathrm{\mu}$m)");
plt.ylabel("Linear attenuation coefficients (in cm$^{-1}$)");

plt.legend(loc='best');

plt.tight_layout();
plt.savefig("profiles.pdf");

plt.show()











fig1, ax1 = plt.subplots()
plt.xlabel("ZNCC (in %)");
plt.ylabel("Runtime (in min)");

for objective, colour, label in zip(
        ["FULL_REGISTRATION_SINOGRAM_NORMALISED_MAE", "FULL_REGISTRATION_SINOGRAM_NORMALISED_RMSE", "FULL_REGISTRATION_PROJS_NORMALISED_DSSIM"],
        ["blue", "green", "red"],
        ["MAE on sinogram with normalisation", "RMSE on sinogram with normalisation", "DSSIM on projections with normalisation"]):
    selection = df["objective"] == objective;

    plt.scatter(df[selection]["LAPLACIAN_LSF_ZNCC"] * 100, df[selection]["OVERALL_RUNTIME (in min)"], color=colour, label=label)

plt.legend(loc='lower center', bbox_to_anchor=(.5, -0.5));


plt.tight_layout();
plt.savefig("scatter_plot.pdf");



print("********************************************************************************")
print("Matrix")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("MATRIX1_ZNCC");
boxplot("MATRIX1_ZNCC", "MATRIX1_ZNCC", "ZNCC in %", 100.0);
printZNCC("MATRIX1_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("CUBE1_RUNTIME (in min)", "MATRIX1_ZNCC", "Runtime in minutes", 1.0);
printDuration("CUBE1_RUNTIME (in min)");
print()
print()



print("********************************************************************************")
print("Radii")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("FIBRE1_ZNCC");
boxplot("FIBRE1_ZNCC", "FIBRE1_ZNCC", "ZNCC in %", 100.0);
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
boxplot("FIBRE2_ZNCC", "FIBRE2_ZNCC", "ZNCC in %", 100.0);
printZNCC("FIBRE2_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()



print("********************************************************************************")
print("Radii again")
print("********************************************************************************")

best_objective, equivalent_objective, p_value_set = ttest("FIBRE3_ZNCC");
boxplot("FIBRE3_ZNCC", "FIBRE3_ZNCC", "ZNCC in %", 100.0);
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
boxplot("HARMONICS_ZNCC", "HARMONICS_ZNCC", "ZNCC in %", 100.0);
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
boxplot("NOISE_ZNCC", "NOISE_ZNCC", "ZNCC in %", 100.0);
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
boxplot("LAPLACIAN1_ZNCC", "LAPLACIAN1_ZNCC", "ZNCC in %", 100.0);
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
boxplot("LAPLACIAN_LSF_ZNCC", "LAPLACIAN_LSF_ZNCC", "ZNCC in %", 100.0);
printZNCC("LAPLACIAN_LSF_ZNCC", best_objective, equivalent_objective, p_value_set);
print()
print()
boxplot("LAPLACIAN_LSF_RUNTIME (in min)", "LAPLACIAN_LSF_ZNCC", "Runtime in minutes", 1.0);
printDuration("LAPLACIAN_LSF_RUNTIME (in min)");
print()
print()

exit()


boxplot("OVERALL_RUNTIME (in min)", "LAPLACIAN_LSF_ZNCC", "Runtime in minutes", 1.0);






SMALL_SIZE = 5
MEDIUM_SIZE = 6
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig1, ax = plt.subplots(gridspec_kw = {'wspace':100, 'hspace': 100})

i = 1;
for objective, label in zip(["FULL_REGISTRATION_SINOGRAM_NORMALISED_RMSE", "FULL_REGISTRATION_PROJS_NORMALISED_DSSIM"],
        ["RMSE on sinogram\nwith normalisation", "DSSIM on projections\nwith normalisation"]):
    selection = df["objective"] == objective;

    idxmin = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmin();
    idxmax = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmax();

    if df[selection]['LAPLACIAN_LSF_ZNCC'].count() != 15:
        print("MISSING DATA FOR", objective)
        exit();
    median_value = df[selection]['LAPLACIAN_LSF_ZNCC'].median();
    for row_id, row_ZNCC in zip(df[selection]['i'], df[selection]['LAPLACIAN_LSF_ZNCC']):

        print(row_ZNCC, median_value)
        if row_ZNCC == median_value:
            idxmedian = row_id;

    print(objective)
    print("\t", df["i"][idxmin], df[selection]["LAPLACIAN_LSF_ZNCC"].min() * 100)
    print("\t", df["i"][idxmax], df[selection]["LAPLACIAN_LSF_ZNCC"].max() * 100)

    fname = objective + "/run_SCW_" + str(df["i"][idxmin]) + "/laplacian-LSF_compare_experiment_with_simulation.png";
    print(fname)
    img = plt.imread(fname);
    ax1 = fig1.add_subplot(3, 2, i);

    ax1.set_title(label + '\nWorse run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].min() * 100) + "%");
    plt.imshow(img, cmap='gray')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])





    fname = objective + "/run_SCW_" + str(df["i"][idxmedian]) + "/laplacian-LSF_compare_experiment_with_simulation.png";
    print(fname)
    img = plt.imread(fname);
    ax1 = fig1.add_subplot(3, 2, 2 + i);

    ax1.set_title('\nMedian run, ZNCC: ' + "{:.2f}".format(median_value * 100) + "%");
    plt.imshow(img, cmap='gray')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])



    fname = objective + "/run_SCW_" + str(df["i"][idxmax]) + "/laplacian-LSF_compare_experiment_with_simulation.png";
    print(fname)
    img = plt.imread(fname);
    ax1 = fig1.add_subplot(3, 2, 4 + i);

    ax1.set_title('\nBest run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].max() * 100) + "%");
    plt.imshow(img, cmap='gray')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])

    i += 1;


# plt.subplots_adjust(wspace=0.5, hspace=0.3)
# set the spacing between subplots
plt.subplots_adjust(left=0.,
                    bottom=0.,
                    right=0.5,
                    top=0.9,
                    wspace=0.,
                    hspace=0.25)

ax.set_xticks([], [])
ax.set_yticks([], [])
ax.axis('off')


# plt.tight_layout();
plt.savefig("checkerboard.pdf", dpi=600);




fig1, ax = plt.subplots(gridspec_kw = {'wspace':100, 'hspace': 100})

i = 1;
for objective, label in zip(["FULL_REGISTRATION_SINOGRAM_NORMALISED_RMSE", "FULL_REGISTRATION_PROJS_NORMALISED_DSSIM"],
        ["RMSE on sinogram\nwith normalisation", "DSSIM on projections\nwith normalisation"]):
    selection = df["objective"] == objective;

    idxmin = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmin();
    idxmax = df[selection]["LAPLACIAN_LSF_ZNCC"].idxmax();

    if df[selection]['LAPLACIAN_LSF_ZNCC'].count() != 15:
        print("MISSING DATA FOR", objective)
        exit();
    median_value = df[selection]['LAPLACIAN_LSF_ZNCC'].median();
    for row_id, row_ZNCC in zip(df[selection]['i'], df[selection]['LAPLACIAN_LSF_ZNCC']):

        print(row_ZNCC, median_value)
        if row_ZNCC == median_value:
            idxmedian = row_id;

    print(objective)
    print("\t", df["i"][idxmin], df[selection]["LAPLACIAN_LSF_ZNCC"].min() * 100)
    print("\t", df["i"][idxmax], df[selection]["LAPLACIAN_LSF_ZNCC"].max() * 100)

    fname_ref = objective + "/run_SCW_" + str(df["i"][idxmin]) + "/laplacian-LSF_reference_fibre_in_centre.mha";
    fname_sim = objective + "/run_SCW_" + str(df["i"][idxmin]) + "/laplacian-LSF_simulated_fibre_in_centre.mha";

    ref_itk_image = sitk.ReadImage(fname_ref);
    ref_np_image = sitk.GetArrayFromImage(ref_itk_image);

    sim_itk_image = sitk.ReadImage(fname_sim);
    sim_np_image = sitk.GetArrayFromImage(sim_itk_image);
    comp_equalized = compare_images(ref_np_image, sim_np_image, method='checkerboard')

    ax1 = fig1.add_subplot(3, 2, i);

    ax1.set_title(label + '\nWorse run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].min() * 100) + "%");
    plt.imshow(comp_equalized, cmap='gray')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])





    fname_ref = objective + "/run_SCW_" + str(idxmedian) + "/laplacian-LSF_reference_fibre_in_centre.mha";
    fname_sim = objective + "/run_SCW_" + str(idxmedian) + "/laplacian-LSF_simulated_fibre_in_centre.mha";

    ref_itk_image = sitk.ReadImage(fname_ref);
    ref_np_image = sitk.GetArrayFromImage(ref_itk_image);

    sim_itk_image = sitk.ReadImage(fname_sim);
    sim_np_image = sitk.GetArrayFromImage(sim_itk_image);
    comp_equalized = compare_images(ref_np_image, sim_np_image, method='checkerboard')

    ax1 = fig1.add_subplot(3, 2, 2 + i);

    ax1.set_title('\nMedian run, ZNCC: ' + "{:.2f}".format(median_value * 100) + "%");
    plt.imshow(comp_equalized, cmap='gray')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])



    fname_ref = objective + "/run_SCW_" + str(df["i"][idxmax]) + "/laplacian-LSF_reference_fibre_in_centre.mha";
    fname_sim = objective + "/run_SCW_" + str(df["i"][idxmax]) + "/laplacian-LSF_simulated_fibre_in_centre.mha";

    ref_itk_image = sitk.ReadImage(fname_ref);
    ref_np_image = sitk.GetArrayFromImage(ref_itk_image);

    sim_itk_image = sitk.ReadImage(fname_sim);
    sim_np_image = sitk.GetArrayFromImage(sim_itk_image);
    comp_equalized = compare_images(ref_np_image, sim_np_image, method='checkerboard')

    ax1 = fig1.add_subplot(3, 2, 4 + i);

    ax1.set_title('\nBest run, ZNCC: ' + "{:.2f}".format(df[selection]["LAPLACIAN_LSF_ZNCC"].max() * 100) + "%");
    plt.imshow(comp_equalized, cmap='gray')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])

    i += 1;


# plt.subplots_adjust(wspace=0.5, hspace=0.3)
# set the spacing between subplots
plt.subplots_adjust(left=0.,
                    bottom=0.,
                    right=0.5,
                    top=0.9,
                    wspace=0.,
                    hspace=0.25)

ax.set_xticks([], [])
ax.set_yticks([], [])
ax.axis('off')


# plt.tight_layout();
plt.savefig("checkerboard-fibre.pdf", dpi=600);




# df.drop("objective")
df.to_csv("summary-bis.csv", index=False);

plt.show()