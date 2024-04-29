import math
import statistics
from collections import defaultdict
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from briefcase.case import Case
from briefcase.case_base import CaseBase
from briefcase.enums import decision_enum

import seaborn as sns

import statistics


def filter_data_by_power(data, perc_max_power=1):
    cases = [Case.from_dict(c) for c in data]
    cb1 = CaseBase(cases)
    pi, delta = cb1.order.PD.max_edges()
    max_powers = {decision_enum.pi: pi * (perc_max_power / 100),
                  decision_enum.delta: delta * (perc_max_power / 100)}
    filtered_data = []

    for i, case in enumerate(cases):
        if cb1.order.PD.case_power(case) <= max_powers[case.decision]:
            filtered_data.append(data[i])

    return filtered_data


def test_powers(cb):
    filtered_cases = cb.cases  # filtered out the empty side cases
    powers = []
    for case in filtered_cases:
        power = cb.order.PD.case_power(case)
        powers.append(power)
        if power == 0:
            print(case)

    return powers


def display_case_powers(data, name="Mushroom"):
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


def test_admit_bf_incons(shuffles):
    # Set the seed for reproducibility
    scores = []
    powers = []
    cases_sizes = []

    def add_dict(dic, factors):
        for factor in factors:
            dic[factor.name] += 1
        return dic

    pi_df = {}
    delta_df = {}

    for k, temp_data in enumerate(shuffles):
        cb = CaseBase([])
        score = 0
        pi_factors = defaultdict(int)
        delta_factors = defaultdict(int)
        cases_sizes.append(len(temp_data))
        for i in range(len(temp_data)):
            new_case = Case.from_dict(temp_data[i])
            polarity = new_case.decision
            if not cb.add_case(new_case, "NO_INCONSISTENCY"):
                score = i
                cb_power = cb.order.PD.cb_power()
                if polarity == decision_enum.pi:
                    pi_factors = add_dict(pi_factors, new_case.reason)
                    delta_factors = add_dict(delta_factors, new_case.defeated())
                else:
                    pi_factors = add_dict(pi_factors, new_case.defeated())
                    delta_factors = add_dict(delta_factors, new_case.reason)
                break
        powers.append(cb_power)
        scores.append(score)
        pi_df[k] = pi_factors
        delta_df[k] = delta_factors
        print(f"{k}: The number of cases we can admit before we create an inconsistency {score}")
        print(f"{k}: The power of case base before we create an inconsistency {cb_power}")
        print(f"{k}: The pi factors on the first inconsistent case {len(pi_factors)}")
        print(f"{k}: The delta factors on the first inconsistent case {len(delta_factors)}")
        print()

    avg = round(sum(scores) / len(scores))
    avg_cases = round(sum(cases_sizes) / len(cases_sizes))
    std_dev = round(statistics.stdev(scores), 2)
    print(f"Average number of cases in case base: {avg_cases}")

    repeats = len(shuffles)
    avg = round(sum(scores) / repeats)
    std_dev = round(statistics.stdev(scores), 2)
    print(f"Average cases before inconsistency ({repeats} trials): {avg}")
    print(f"Standard deviation: {std_dev}")

    avg_power = round(sum(powers) / repeats)
    std_dev_power = round(statistics.stdev(powers), 2)
    print(f"Average power score ({repeats} trials): {avg_power}")
    print(f"Standard deviation: {std_dev_power}")

    return scores, powers, avg, std_dev, pd.DataFrame(pi_df), pd.DataFrame(delta_df)


def partition_data(data):
    pi_data = [entry for entry in data if entry.get("decision") == "pi"]
    delta_data = [entry for entry in data if entry.get("decision") == "delta"]
    return pi_data, delta_data

def balanced_shuffle_data(data, repeats=50):
    random.seed(42)
    pi_data, delta_data = partition_data(data[:])
    samples = []
    for _ in range(repeats):
        min_data = min(len(pi_data), len(delta_data))
        pi_samples = random.sample(pi_data, math.ceil(min_data / 2))
        delta_samples = random.sample(delta_data, min_data // 2)
        joint_samples = []
        for i in range(min_data // 2):
            joint_samples.append(pi_samples[i])
            joint_samples.append(delta_samples[i])

        if min_data % 2 == 1:
            joint_samples.append(pi_samples[-1])
        samples.append(joint_samples)

    return samples

def shuffle_data(data, repeats=51):
    shuffled_data = []
    random.seed(42)
    for _ in range(repeats):
        temp_data = data[:]  # Make a copy of the data to avoid modifying the original
        random.shuffle(temp_data)
        shuffled_data.append(temp_data)

    return shuffled_data


def factors_descriptive_stats(data):
    cases = [Case.from_dict(c) for c in data]
    cb = CaseBase(cases)
    filtered_cases = cb.cases  # filtered out the empty side cases

    print(f"Number of cases with an empty factorset side {len(cases) - len(filtered_cases)} which are filtered out")

    pi_counts = []  # Store the number of pi factors for each case
    delta_counts = []  # Store the number of delta factors for each case

    for case in filtered_cases:
        count_pi = len(case.pi_factors)
        count_delta = len(case.delta_factors)
        pi_counts.append(count_pi)
        delta_counts.append(count_delta)

    # Plot histogram with stacked bars
    plt.figure(figsize=(10, 5))
    plt.hist([pi_counts, delta_counts], bins=range(max(max(pi_counts), max(delta_counts)) + 2),
             color=['skyblue', 'lightgreen'], label=['Pi Factors', 'Delta Factors'], edgecolor='black', align='left',
             stacked=False)

    plt.xlabel('Number of Factors')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Pi and Delta Factors')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(max(max(pi_counts), max(delta_counts)) + 1))
    plt.show()

    # Calculate total number of cases
    total_cases = len(pi_counts)

    # Calculate co-occurrence matrix
    cooccurrence_matrix = np.zeros((max(pi_counts) + 1, max(delta_counts) + 1))
    for pi, delta in zip(pi_counts, delta_counts):
        cooccurrence_matrix[pi, delta] += 1

    # Divide each element in the co-occurrence matrix by the total number of cases to get percentages
    cooccurrence_matrix_percentage = (cooccurrence_matrix / total_cases) * 100

    # Flip the co-occurrence matrix
    cooccurrence_matrix_percentage = np.flipud(cooccurrence_matrix_percentage)

    # Create heatmap
    plt.figure(figsize=(10, 5))
    plt.imshow(cooccurrence_matrix_percentage, cmap='Blues', vmin=0, vmax=8, interpolation='nearest')
    plt.colorbar(label='Percentage')
    plt.xlabel('Number of Delta Factors')
    plt.ylabel('Number of Pi Factors')
    plt.title('Percentage of Cases with A Combined Number of Pi and Delta Factors')
    plt.grid(True)

    # Set the extent of the y-axis
    plt.ylim(0, max(pi_counts))

    plt.show()

    # Calculate averages
    avg_pi_factors = np.mean(pi_counts)
    avg_delta_factors = np.mean(delta_counts)
    avg_combined_factors = np.mean([pi + delta for pi, delta in zip(pi_counts, delta_counts)])

    # Calculate medians
    median_pi_factors = np.median(pi_counts)
    median_delta_factors = np.median(delta_counts)
    median_combined_factors = np.median([pi + delta for pi, delta in zip(pi_counts, delta_counts)])

    print("Average number of Pi factors:", avg_pi_factors)
    print("Median number of Pi factors:", median_pi_factors)
    print("Average number of Delta factors:", avg_delta_factors)
    print("Median number of Delta factors:", median_delta_factors)
    print("Average combined number of factors:", avg_combined_factors)
    print("Median combined number of factors:", median_combined_factors)


def incons_percentage(pi_df, delta_df):
    # Count the number of inconsistent factors for each factor name
    inconsistent_counts_pi = pi_df.transpose().apply(pd.Series.value_counts).fillna(0).astype(int).transpose()
    inconsistent_counts_delta = delta_df.transpose().apply(pd.Series.value_counts).fillna(0).astype(int).transpose()

    # Calculate the percentage of inconsistency for each factor
    total_cases = len(pi_df.transpose())
    inconsistent_percentage_pi = (inconsistent_counts_pi[1.0] / total_cases) * 100
    inconsistent_percentage_delta = (inconsistent_counts_delta[1.0] / total_cases) * 100

    return inconsistent_percentage_pi, inconsistent_percentage_delta


def test_factor_incons(data, repeats):
    _, _, _, _, pi_df, delta_df = test_admit_bf_incons(balanced_shuffle_data(data, repeats))

    # 1. Factors More Likely to be Inconsistent
    inconsistent_percentage_pi, inconsistent_percentage_delta = incons_percentage(pi_df, delta_df)
    return inconsistent_percentage_pi, inconsistent_percentage_delta, pi_df, delta_df


def experiment_factor_incons(data, repeats):
    inconsistent_percentage_pi, inconsistent_percentage_delta, pi_df, delta_df = test_factor_incons(data, repeats)

    # Visualize the distribution of inconsistency across factors
    plt.figure(figsize=(10, 5))
    sns.barplot(x=inconsistent_percentage_pi.index, y=inconsistent_percentage_pi.values)
    plt.title(f"Percentage of First Inconsistent Cases Featuring Pi Factor ({repeats} trials)")
    plt.xlabel("Factor")
    plt.ylabel("Percentage of Inconsistent Cases")
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Visualize the distribution of inconsistency across factors
    plt.figure(figsize=(10, 5))
    sns.barplot(x=inconsistent_percentage_delta.index, y=inconsistent_percentage_delta.values)
    plt.title(f"Percentage of First Inconsistent Cases Featuring Delta Factor ({repeats} trials)")
    plt.xlabel("Factor")
    plt.ylabel("Percentage of Inconsistent Cases")
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Compute the frequency of co-occurrence of inconsistencies between pairs of factors
    cooccurrence_matrix_pi = pi_df.fillna(0).transpose().corr()

    # Visualize the co-occurrence matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cooccurrence_matrix_pi, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
    plt.title(f"Heatmap of Co-occurence of Pi Factors in First Inconsistent Cases ({repeats} trials)")
    plt.xlabel("Factors")
    plt.ylabel("Factors")
    plt.show()

    # Compute the frequency of co-occurrence of inconsistencies between pairs of factors
    cooccurrence_matrix_delta = delta_df.fillna(0).transpose().corr()

    # Visualize the co-occurrence matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cooccurrence_matrix_delta, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
    plt.title(f"Heatmap of Co-occurence of Delta Factors in First Inconsistent Cases ({repeats} trials)")
    plt.xlabel("Factors")
    plt.ylabel("Factors")
    plt.show()


def remove_factors_data(train_data, pi_factors, delta_factors):
    df = pd.DataFrame(train_data)
    # Remove factors from PI_factors column
    df['pi'] = df['pi'].apply(lambda x: [factor for factor in x if factor in pi_factors])
    # Remove factors from Delta_factors column
    df['delta'] = df['delta'].apply(lambda x: [factor for factor in x if factor in delta_factors])

    df['reason'] = df['reason'].apply(
        lambda x: [factor for factor in x if factor in delta_factors or factor in pi_factors])
    return df


def remove_factors_incons(train_data, inconsistent_percentage_pi, inconsistent_percentage_delta, perc):
    # Filter factors with inconsistency occurrence over %
    filtered_factors_pi = inconsistent_percentage_pi[inconsistent_percentage_pi <= perc]
    filtered_factors_delta = inconsistent_percentage_delta[inconsistent_percentage_delta <= perc]

    # Get the names of factors to keep
    factors_to_keep_pi = filtered_factors_pi.index.tolist()
    factors_to_keep_delta = filtered_factors_delta.index.tolist()

    new_data = remove_factors_data(train_data, factors_to_keep_pi, factors_to_keep_delta).to_dict(orient='records')
    return new_data


def remove_factors_experiment(data):
    percentages = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)
    repeats_train = 300
    repeats_test = 300
    incons_perc_pi_tr, incons_perc_delta_tr, _, _ = test_factor_incons(train_data, repeats_train)
    test_data_shuffles = shuffle_data(test_data, repeats_test)
    avgs = []
    stds = []
    for perc in percentages:
        print(f"\n\nPercentage inconsistency of factors {perc}")
        filtered_shuffles = []
        for shuffle in test_data_shuffles:
            filtered_shuffles.append(remove_factors_incons(shuffle, incons_perc_pi_tr, incons_perc_delta_tr, perc))
        _, _, avg, std, _, _ = test_admit_bf_incons(filtered_shuffles)
        avgs.append(avg)
        stds.append(std)

    # Visualize the results
    plt.errorbar(percentages, avgs, yerr=stds, fmt='o', capsize=5)
    plt.xlabel(f'Percentage of inconsistent training cases with factor - {repeats_train} trials')
    plt.ylabel(f'Average cases before inconsistency - {repeats_test} trials')
    plt.title('Effect of removing factors which are more likely to be associated with inconsistency')
    plt.grid(True)
    plt.show()


def power_remover_incons_experiment(data, repeats=100):
    percentages = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avgs = []
    stds = []
    cb_incons = []
    for perc in percentages:
        power_reduced_data = filter_data_by_power(data[:], perc)
        shuffles = shuffle_data(power_reduced_data, repeats)
        print(f"\nPercentage inconsistency of factors {perc}")
        _, _, avg, std, _, _ = test_admit_bf_incons(shuffles)
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
    scores, powers, _, _, _, _ = test_admit_bf_incons(shuffle_data(data, repeats))

    # Calculating quintiles
    score_quintiles = np.percentile(scores, [20, 40, 60, 80])
    power_quintiles = np.percentile(powers, [20, 40, 60, 80])

    # Calculate averages
    score_average = np.mean(scores)
    power_average = np.mean(powers)

    # Plotting inconsistency scores with quintiles
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=30, color='skyblue', edgecolor='black')

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