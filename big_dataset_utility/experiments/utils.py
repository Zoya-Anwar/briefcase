import math
import random
import statistics
from collections import defaultdict

import pandas as pd
from briefcase.case import Case
from briefcase.case_base import CaseBase
from briefcase.enums import decision_enum


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
    _, _, _, _, pi_df, delta_df = experiment_admit_bf_incons(balanced_shuffle_data(data, repeats))

    # 1. Factors More Likely to be Inconsistent
    inconsistent_percentage_pi, inconsistent_percentage_delta = incons_percentage(pi_df, delta_df)
    return inconsistent_percentage_pi, inconsistent_percentage_delta, pi_df, delta_df


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


def experiment_admit_bf_incons(shuffles):
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

    avg_cases = round(sum(cases_sizes) / len(cases_sizes))
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
