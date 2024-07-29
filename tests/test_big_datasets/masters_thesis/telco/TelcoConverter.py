from briefcase.enums import decision_enum
from briefcase.case_base import CaseBase
from briefcase.case import Case
import pandas as pd
from big_dataset_utility.feature_labeling.cluster_binary_factors import cluster_factors_voting, cluster_factors_rand_un, \
    cluster_factors_corr, cluster_factors_rand, reduce_df
import yaml
import math
from collections import defaultdict
import statistics
import random

"""
Collection of functions for experiments with the Telecoms Dataset
"""

def get_df():
    # Load the dataset
    df = pd.read_csv('data/telco.csv')

    # Replace "Yes" and "No" with 1 and 0 respectively for binary columns
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

    print(f"{len(binary_columns)} binary columns")
    for col in binary_columns:
        df[col] = df[col].map({'Yes': True, 'No': False})
        df[col] = df[col].astype(bool)

    datatype_counts = df.dtypes.value_counts()
    print(datatype_counts)

    # Drop numerical columns
    df = df.select_dtypes(exclude=['int64', 'float64'])
    df = df.drop('TotalCharges', axis=1)
    df = df.drop('customerID', axis=1)

    print("2 numerical columns - dropped")

    df_encoded = pd.get_dummies(df)

    return df_encoded, 'Churn'


def convert_to_dict(test_df, y_name, factors):
    # create a dictionary
    cases = []
    for i, row in test_df.iterrows():
        case = {
            "pi": [],
            "delta": [],
            "reason": [],
            "decision": None
        }

        for col, value in row.items():  # iterate through all factor candidates in case
            if value and col != y_name:  # only add factor if True
                if factors[col] == "pi" and value:  # pi (True) factor
                    case["pi"].append(col)
                elif factors[col] == "delta" and value:  # delta (False) factor
                    case["delta"].append(col)
                else:  # todo: neutral/undecided case
                    pass

        # for the mushroom dataset the reason is the same as the decision
        if row[y_name]:
            case["reason"] = case["pi"]
            case["decision"] = "pi"
        elif not row[y_name]:
            case["reason"] = case["delta"]
            case["decision"] = "delta"

        cases.append(case)

    return cases


def display_facotrs(factors):
    # Initialize counters for each factor type
    pis = []
    deltas = []
    uns = []

    # Count the occurrences of each factor type
    for factor, factor_type in factors.items():
        if factor_type == 'pi':
            pis.append(factor)
        elif factor_type == 'delta':
            deltas.append(factor)
        else:
            uns.append(factor)

    # Print the counts for each factor type
    print("Number of factors by type:")
    print(f"PI factors: {len(pis)} - {pis}")
    print(f"Delta factors: {len(deltas)} - {deltas}")
    print(f"UN factors: {len(uns)} - {uns}")


def dataset_to_yaml(cluster_type="corr", test_split=0.5, size_filter=False):
    # import data, name of the outcome variables
    df, y_name = get_df()

    # Set the seed for reproducibility
    df = df.sample(frac=1, random_state=42)

    # Determine the split point
    # Training is for the factor clustering
    split_point = int(test_split * len(df))

    # Split the DataFrame
    test_df = df.iloc[:split_point]
    train_df = df.iloc[split_point:]

    # Constrain the dataframe to attempt to remove very high powered cases before clustering
    if size_filter:
        train_df = reduce_df(train_df)

    # get factor clusters using chosen clustering method
    if cluster_type == "corr":
        factors = cluster_factors_corr(train_df, y_name)
    elif cluster_type == "vote":
        factors = cluster_factors_voting(train_df, y_name)
    elif cluster_type == "rand_un":
        factors = cluster_factors_rand_un(train_df, y_name)
    else:
        factors = cluster_factors_rand(train_df, y_name)

    display_facotrs(factors)

    # get dict format of cases
    cases_dict = convert_to_dict(test_df, y_name, factors)
    filename = f"data/telco-{cluster_type}-test-{test_split}-{size_filter}.yaml"

    # Write the dictionary to a new YAML file
    with open(filename, "w") as yaml_file:
        yaml.dump(cases_dict, yaml_file, default_flow_style=False)

    return cases_dict


def get_existing_test_data(cluster="corr", test_split="0.25", size_filter=False):
    filename = f"data/telco-{cluster}-test-{test_split}-{size_filter}.yaml"

    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        print(f"Loaded data successfully from '{filename}'")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        print(f"Creating new file")
        data = dataset_to_yaml(cluster, float(test_split), size_filter)
    except Exception as e:
        print("An error occurred:", str(e))

    return data


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


def partition_data(data):
    pi_data = [entry for entry in data if entry.get("decision") == "pi"]
    delta_data = [entry for entry in data if entry.get("decision") == "delta"]
    return pi_data, delta_data


def shuffle_data(data, repeats=51):
    shuffled_data = []
    random.seed(42)
    for _ in range(repeats):
        temp_data = data[:]  # Make a copy of the data to avoid modifying the original
        random.shuffle(temp_data)
        shuffled_data.append(temp_data)

    return shuffled_data


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
            if not cb.add_case(new_case, "NO"):
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
    print(f"Average cases before inconsistency: {avg}")
    print(f"Standard deviation: {std_dev}")
    print()

    return scores, powers, avg, std_dev, pd.DataFrame(pi_df), pd.DataFrame(delta_df)