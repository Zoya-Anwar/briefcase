import pandas as pd
import import_ipynb
from big_dataset_utility.feature_labeling.cluster_binary_factors import cluster_factors_voting, cluster_factors_rand_un, \
    cluster_factors_corr, cluster_factors_rand
import yaml


def get_df():
    df = pd.read_csv('data/mushrooms.csv')
    y_name = 'class_p'

    datatype_counts = df.dtypes.value_counts()
    print(datatype_counts)

    # delete feature for which all instances have the same value
    del df['veil-type']

    # convert categorical values into dummies
    df = pd.get_dummies(df)

    # delete extra outcome column
    del df['class_e']

    return df, y_name


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


def mushroom_to_yaml(cluster_type="corr", test_split=0.5):
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

    # get factor clusters using chosen clustering method
    if cluster_type == "corr":
        factors = cluster_factors_corr(train_df, y_name)
    elif cluster_type == "vote":
        factors = cluster_factors_voting(train_df, y_name)
    elif cluster_type == "rand_un":
        factors = cluster_factors_rand_un(train_df, y_name)
    else:
        factors = cluster_factors_rand(train_df, y_name)

    # get dict format of cases
    cases_dict = convert_to_dict(test_df, y_name, factors)
    filename = f"data/mushrooms-{cluster_type}-test-{test_split}.yaml"

    # Write the dictionary to a new YAML file
    with open(filename, "w") as yaml_file:
        yaml.dump(cases_dict, yaml_file, default_flow_style=False)

    return cases_dict


def get_existing_test_data(cluster="corr", test_split="0.25"):
    filename = f"data/mushrooms-{cluster}-test-{test_split}.yaml"

    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        print(f"Loaded data successfully from '{filename}'")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        print(f"Creating new file")
        data = mushroom_to_yaml(cluster, float(test_split))
    except Exception as e:
        print("An error occurred:", str(e))

    return data