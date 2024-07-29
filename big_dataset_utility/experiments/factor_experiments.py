import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from big_dataset_utility.experiments.utils import test_factor_incons, shuffle_data, remove_factors_incons, \
    experiment_admit_bf_incons

from briefcase.case import Case
from briefcase.case_base import CaseBase


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
        _, _, avg, std, _, _ = experiment_admit_bf_incons(filtered_shuffles)
        avgs.append(avg)
        stds.append(std)

    # Visualize the results
    plt.errorbar(percentages, avgs, yerr=stds, fmt='o', capsize=5)
    plt.xlabel(f'Percentage of inconsistent training cases with factor - {repeats_train} trials')
    plt.ylabel(f'Average cases before inconsistency - {repeats_test} trials')
    plt.title('Effect of removing factors which are more likely to be associated with inconsistency')
    plt.grid(True)
    plt.show()


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
