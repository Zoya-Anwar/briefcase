import random
import sys

import numpy as np
from matplotlib import pyplot as plt

from big_dataset_utility.experiments.utils import test_powers, filter_data_by_power, shuffle_data, \
    experiment_admit_bf_incons
from briefcase.case import Case
from briefcase.case_base import CaseBase


def case_powers_histogram_experiment(data, name="Mushroom"):
    """
    EXPERIMENT (1 IN MEASUREMENT PAPER)
    Creates a histogram displaying the distribution of the Information Content (case power)
    for each case in a dataset.
    Labels the averages, and quintiles.
    Uses a log scale on the x-axis (case powers).
    """

    cases = [Case.from_dict(c) for c in data]
    cb = CaseBase(cases)

    powers = test_powers(cb)
    unique_powers, power_counts = np.unique(powers, return_counts=True)

    # Calculate quintiles
    power_quintiles = np.percentile(unique_powers, [20, 40, 60, 80])

    # Calculate bin edges
    bin_edges = np.logspace(np.log10(min(unique_powers)), np.log10(max(unique_powers)), 50)
    bin_edges[-1] *= 1.05  # Extend the last bin to ensure it includes the maximum power value

    # Plotting the histogram with a log scale on the x-axis
    plt.figure(figsize=(10, 5))
    plt.hist(powers, bins=bin_edges, color='lightgreen', edgecolor='black')
    plt.xscale('log')
    plt.title(f'Distribution of Case Powers, {name} Dataset')
    plt.xlabel('Case Power (log10)')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y')  # Grid on y-axis only for better visibility of frequencies

    # Define tick positions for the x-axis
    num_logs = int(np.log10(max(unique_powers)) - np.log10(min(unique_powers))) + 1
    min_power = min(unique_powers)
    max_power = max(unique_powers)
    tick_positions = np.logspace(np.log10(min_power), np.log10(max_power), num=num_logs)

    # Set custom ticks and labels for the x-axis
    plt.xticks(tick_positions, [f'{int(np.log10(tick))}' for tick in tick_positions])

    # Mark quintiles and their original power numbers
    for quintile in power_quintiles:
        plt.axvline(quintile, color='blue', linestyle='dashed', linewidth=2)
        plt.text(quintile, 100, f' Q{np.where(power_quintiles == quintile)[0][0] + 1} \n {np.log10(quintile):.2f}',
                 verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    # Mark the average
    power_average = np.mean(unique_powers)
    plt.axvline(power_average, color='red', linestyle='dashed', linewidth=2)
    plt.text(power_average, max(power_counts) + 10, f' Average \n {np.log10(power_average):.2f}',
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Calculate median
    power_median = np.median(unique_powers)
    plt.axvline(power_median, color='purple', linestyle='dashed', linewidth=2)
    plt.text(power_median, max(power_counts) + 10, f' Median \n {np.log10(power_median):.2f}',
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Mark the maximum power possible
    max_power = max(unique_powers)
    plt.axvline(max_power, color='orange', linestyle='dashed', linewidth=2)
    plt.text(max_power, max(power_counts) + 10, f' Max Power \n {np.log10(max_power):.2f}', verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    # Mark the minimum power possible
    min_power = min(unique_powers)
    plt.axvline(min_power, color='orange', linestyle='dashed', linewidth=2)
    plt.text(min_power, max(power_counts) + 10, f' Min Power \n {np.log10(min_power):.2f}', verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    plt.show()

    print(f"Median is {(power_median / max_power) * 100}% of max dataset power")
    print(f"Q1 is {(power_quintiles[0] / max_power) * 100}% of max dataset power")
    print(f"Q2 is {(power_quintiles[1] / max_power) * 100}% of max dataset power")
    print(f"Q3 is {(power_quintiles[2] / max_power) * 100}% of max dataset power")
    print(f"Q4 is {(power_quintiles[3] / max_power) * 100}% of max dataset power")
    print(f"Mean is {(power_average / max_power) * 100}% of max dataset power")

    max_pi, max_delta = cb.order.PD.max_edges()

    print()
    print(f"Median is {(power_median / max_pi) * 100}% of max theoretical pi winning power")
    print(f"Q1 is {(power_quintiles[0] / max_pi) * 100}% of max theoretical pi winning power")
    print(f"Q2 is {(power_quintiles[1] / max_pi) * 100}% of max theoretical pi winning power")
    print(f"Q3 is {(power_quintiles[2] / max_pi) * 100}% of max theoretical pi winning power")
    print(f"Q4 is {(power_quintiles[3] / max_pi) * 100}% of max theoretical pi winning power")
    print(f"Mean is {(power_average / max_pi) * 100}% of max theoretical pi winning power")
    print()
    print(f"Median is {(power_median / max_delta) * 100}% of max theoretical delta winning power")
    print(f"Q1 is {(power_quintiles[0] / max_delta) * 100}% of max theoretical delta winning power")
    print(f"Q2 is {(power_quintiles[1] / max_delta) * 100}% of max theoretical delta winning power")
    print(f"Q3 is {(power_quintiles[2] / max_delta) * 100}% of max theoretical delta winning power")
    print(f"Q4 is {(power_quintiles[3] / max_delta) * 100}% of max theoretical delta winning power")
    print(f"Mean is {(power_average / max_delta) * 100}% of max theoretical max_delta winning power")


def power_filter_experiment(data, repeats=100):
    """
    EXPERIMENT
    Filters dataset to remove cases with power over a threshold (removing widely deciding cases).
    (A) On filtered data, tests the number of admits before the first inconsistency (100 trials)
        and plots an error bar
    (B) On filtered data, counts the number of inconsistent cases (i.e. tainted cases). Plots a scatter plot.
    """
    percentages = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avgs = []
    stds = []
    cb_incons = []

    for perc in percentages:
        power_reduced_data = filter_data_by_power(data[:], perc)
        shuffles = shuffle_data(power_reduced_data, repeats)
        print(f"\nPercentage inconsistency of factors {perc}")
        _, _, avg, std, _, _ = experiment_admit_bf_incons(shuffles)
        avgs.append(avg)
        stds.append(std)

        cb1 = CaseBase(Case.from_dict(c) for c in power_reduced_data)
        cb_incons.append(cb1.count_tainted_cases() / len(cb1.cases))

    # Visualize the results
    plt.errorbar(percentages, avgs, yerr=stds, fmt='o', capsize=5)
    plt.xlabel(f'Percentage power of cases over max power')
    plt.ylabel(f'Average cases before inconsistency - {repeats} trials')
    plt.title('Effect of filtering cases by percentage power on number of cases before inconsistency')
    plt.grid(True)
    plt.show()

    # Visualize the results
    plt.scatter(percentages, cb_incons, color='blue', label='Data Points')
    plt.xlabel('Percentage power of cases over max power')
    plt.ylabel('Percentage inconsistent cases in Case Base')
    plt.title('Scatter Plot of Percentage Inconsistent Cases with Case Base Filtered by Power')
    plt.xlim(0, max(percentages) + 1)
    plt.ylim(0, max(cb_incons) + 0.1)


def cb_power_before_incons_experiment(data, repeats=100):
    """
    EXPERIMENT

    For n (default 100) trials, shuffle the dataset and count how many cases there are before an inconsistency

    (a) Plot the distribution of the number of cases before inconsistency
    (b) Plot the distribution of the casebase power before inconsistency

    """
    scores, powers, _, _, _, _ = experiment_admit_bf_incons(shuffle_data(data, repeats))

    # Calculating quintiles
    score_quintiles = np.percentile(scores, [20, 40, 60, 80])
    power_quintiles = np.percentile(powers, [20, 40, 60, 80])

    # Calculate averages
    score_average = np.mean(scores)
    power_average = np.mean(powers)

    # Plotting inconsistency scores with quintiles
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=30, color='skyblue', edgecolor='black')

    # (A)
    # Mark quintiles and their original power numbers
    for quintile in score_quintiles:
        plt.axvline(quintile, color='blue', linestyle='dashed', linewidth=2)
        plt.text(quintile, 8, f' Q{np.where(score_quintiles == quintile)[0][0] + 1} \n {quintile:.0f}',
                 verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.axvline(score_average, color='red', linestyle='dashed', linewidth=2)
    plt.text(score_average, 6, f' Average \n {score_average:.0f}', verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    plt.title(f'Distribution of Number of Cases Before Inconsistency - {repeats} trials')
    plt.xlabel('No. Cases Before Inconsistency')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plotting power scores with quintiles
    plt.figure(figsize=(10, 5))
    plt.hist(powers, bins=30, color='lightgreen', edgecolor='black')

    # (B)
    # Mark quintiles and their original power numbers
    for quintile in power_quintiles:
        plt.axvline(quintile, color='blue', linestyle='dashed', linewidth=2)
        plt.text(quintile, 8, f' Q{np.where(power_quintiles == quintile)[0][0] + 1} \n {quintile:.0f}',
                 verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.axvline(power_average, color='red', linestyle='dashed', linewidth=2)
    plt.text(power_average, 6, f' Average \n {power_average:.0f}', verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    plt.title(f'Distribution of Case Base Power Before Inconsistency - {repeats} trials')
    plt.xlabel('Case Base Power Before Inconsistency')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def max_consistent_subset(data, repeats=10):
    """
    EXPERIMENT
    For n (default 10) trials, shuffle the data, and try to add all cases under the no_inconsistency regime
    Generate a percentage of the case base which is consistent
    """
    shuffles = shuffle_data(data, repeats)
    sizes_max_cons_subsets = []
    for data in shuffles:
        print("new shuffle")
        print(f"Adding {len(data)} cases")
        cb = CaseBase()
        for c in data:
            cb.add_case(Case.from_dict(c), "NO_INCONSISTENCY")
        size = cb.size()
        print(f"Accepted {size} cases")
        sizes_max_cons_subsets.append(size)

    print(sizes_max_cons_subsets)

def divergence_experiment(data, repeats=10):
    """
    EXPERIMENT
    For n (default 10) trials, shuffle the data, try to create consistent subsets of the data (100 entries)
    Add a new case which causes inconsistency, measure how much inconsistency it induces (case power vs cases tainted)
    """
    shuffles = shuffle_data(data, repeats)
    case_bases = []
    incons_adds = []

    while shuffles:
        cb = CaseBase()
        incons_with = []

        for c in shuffles.pop():
            case = Case.from_dict(c)
            if not incons_with:
                if not cb.order.is_cb_consistent_with(case):
                    incons_with.append(case)
            if cb.size() < 100:
                cb.add_case(case, "NO_INCONSISTENCY")
            elif cb.size() == 100 and incons_with:
                print("A case base complete")
                case_bases.append(cb)
                incons_adds.append(incons_with)
                break
        else:
            shuffles.extend(shuffle_data(data, 1))

    print("finished with shuffles")

    # Find inconsistent case with this case
    for i in range(len(case_bases)):
        cb = case_bases[i]
        incons_with = incons_adds[i]
        for case in incons_with:
            if not cb.is_consistent_with(case):
                print("")
                print(cb.metrics())
                print(f"Previous (consistent) CB power is {cb.order.PD.cb_power()}")
                print(f"Previous (consistent) CB tainted cases are {cb.count_tainted_cases()}")
                print(f"Inconsistent case power is {cb.order.PD.case_power(case)}")
                print("Adding case")
                cb.add_case(case)
                print(f"New CB power is {cb.order.PD.cb_power()}")
                print(f"New CB tainted cases are {cb.count_tainted_cases()}")
                break
        else:
            print("WARNING: no cases inconsistent with this case base")



