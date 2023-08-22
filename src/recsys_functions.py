
import argparse
import datetime
import time

import numpy as np
import pandas as pd



# input: raw df of customer data
# output: list containing a dictionary of two dfs per split, one containing the
# training data and one containing the inference data (both have rows of single orders)

def split_product_data(df, n_days,n_splits, start_date):
    start_date = pd.to_datetime(start_date)
    end_date = start_date + pd.Timedelta(f"{n_days + n_splits}D")

    difference = (end_date - start_date).days

    if difference < n_days + n_splits:
        print(f"combination of {n_days} days and {n_splits} splits gives too few days")
        return None
    df_mask = (df['ORDER_DATE'] >= start_date) & (df['ORDER_DATE'] <= end_date)
    df = df[df_mask]

    return df

# thesis is set to true here
def split_data(
    df: pd.DataFrame, n_days=0, n_splits=0, start_date=None, labels=False, encode=False, thesis=True,
) -> list[dict]:
    df = df.reset_index(drop=True)
    # make sure that the df is started by date
    df = df.sort_values(by="ORDER_DATE")

    if thesis:
        thesis_mask = df["PRODUCT_GROUP_LEVEL_1_DESCRIPTION"].isin(
            ["clothing", "footwear", "accessories"]
        )
        df = df[thesis_mask]

    if not start_date:
        start_date = df.iloc[0]["ORDER_DATE"]
    else:
        start_date = pd.to_datetime(start_date)

    end_date = start_date + pd.Timedelta(f"{n_days+n_splits}D")

    difference = (end_date - start_date).days

    if difference < n_days + n_splits:
        print(f"combination of {n_days} days and {n_splits} splits gives too few days")
        return None

    if encode:
        encode_columns = [
            "CUSTOMER_NO",
            "CITY",
            "BEX_PRODUCT_NO",
            "COLOUR",
            "GENDER_DESCRIPTION",
            "BRAND",
            "COLLECTION",
            "PRODUCT_GROUP_LEVEL_1_DESCRIPTION",
            "PRODUCT_GROUP_LEVEL_2_DESCRIPTION",
            "PRODUCT_GROUP_LEVEL_3_DESCRIPTION",
        ]

        # fundera på om jag verkligen behöver encodingsen
        encodings = {}
        for encode_column in encode_columns:
            fact = pd.factorize(df[encode_column].values.tolist(), sort=True)
            df[encode_column], encodings[encode_column] = fact[0].astype('int32'), fact[1]
        
    splits = []

    for i in range(n_splits):
        start_date_i = start_date + pd.Timedelta(f"{i}D")
        end_date_i = start_date_i + pd.Timedelta(f"{n_days}D")
        # timea hur lång tid den tar
        # kanske indexera på datumet istället borde vara snabba
        df_i_train_mask = (df["ORDER_DATE"] < end_date_i) & (
            df["ORDER_DATE"] >= start_date_i
        )

        # kan det bli några fel här?
        df_i_train = df[df_i_train_mask]
        df_i_validation_mask = df["ORDER_DATE"] == end_date_i
        df_i_validation = df[df_i_validation_mask]

        if not df_i_validation.empty:
            # https://www.geeksforgeeks.org/python-pandas-dataframe-isin/
            # https://stackoverflow.com/questions/12065885/filter-dataframe-rows-if-value-in-column-is-in-a-set-list-of-values

            # mask so that only customers that are in the eval data are in the train data (not needed)
            # df_i_train_mask = df_i_train["CUSTOMER_NO"].isin(df_i_validation["CUSTOMER_NO"].tolist())
            # df_i_train = df_i_train[df_i_train_mask]

            # mask so that only customers in the train data are in the eval data
            df_i_validation_mask = df_i_validation["CUSTOMER_NO"].isin(
                df_i_train["CUSTOMER_NO"].tolist()
            )
            df_i_validation = df_i_validation[df_i_validation_mask]

            if labels:
                # mask so that only item categories in the train data is in the eval data (the opposite is not needed)
                df_i_validation_mask = (
                    df_i_validation["BEX_PRODUCT_NO"].isin(
                        df_i_train["BEX_PRODUCT_NO"].tolist()
                    )
                    & df_i_validation["LABEL"]
                    == 1
                )
            else:
                # mask so that only item categories in the train data is in the eval data (the opposite is not needed)
                df_i_validation_mask = df_i_validation["BEX_PRODUCT_NO"].isin(
                    df_i_train["BEX_PRODUCT_NO"].tolist()
                )
            df_i_validation = df_i_validation[df_i_validation_mask]

            # håll det till en dataframe om det går?

            if not (df_i_train.empty or df_i_validation.empty):
                if encode:
                    splits.append(
                    {
                        "train": df_i_train.reset_index(drop=True),
                        "validation": df_i_validation.reset_index(drop=True),
                        "encodings": encodings,
                    }
                )
                else:
                    splits.append(
                        {
                            "train": df_i_train.reset_index(drop=True),
                            "validation": df_i_validation.reset_index(drop=True),
                        }
                    )

    return splits


# input: the list of splitted data
# output: a list of a dict for each split, containing the the bex product numbers found in the training data,
# and a list of list of probabilities for each row/user in the inference data.
def generate_random_predictions(splits: list[dict], same=False) -> list[dict]:
    # rng to keep the seed consistent
    rng = np.random.default_rng(seed=42)

    def generate_random_probabiliy_space(n):
        random_prob_space = rng.random(n)
        return (random_prob_space / random_prob_space.sum()).tolist()

        # ha fortfarande args.thesis

        # fixa lower på product level description
        # bex_data = bex_data[thesis_mask]

    random_bex_probs = []

    for split in splits:
        # print(split['truth'].shape)
        # get only unique products
        bex_products_in_split = list(set(split["train"]["BEX_PRODUCT_NO"].tolist()))
        n_products_in_split = len(bex_products_in_split)

        individual_probabilities = []

        if same:
            random_bex_prob = generate_random_probabiliy_space(n_products_in_split)

            for _ in range(split["validation"].shape[0]):
                individual_probabilities.append(random_bex_prob)

            # individual_probabilities.append(
            #     random_bex_prob for _ in range(split["validation"].shape[0])
            # )

        else:
            for _ in range(split["validation"].shape[0]):
                individual_probabilities.append(
                    generate_random_probabiliy_space(n_products_in_split)
                )
            # individual_probabilities.append(
            #     generate_random_probabiliy_space(n_products_in_split)
            #     for _ in range(split["validation"].shape[0])
            # )

        random_bex_probs.append(
            {
                "BEX_PRODUCT_NO_IN_SPLIT": bex_products_in_split,
                "probabilities_in_split": individual_probabilities,
            }
        )
    # print("this is how probabilities in split looks like:")
    # print(random_bex_probs[0]["probabilities_in_split"])

    return random_bex_probs


# input: the list of splitted data
# output: a list of a dict for each split, containing the the bex product numbers found in the training data,
# and a list of list of probabilities for each row/user in the inference data.
def generate_popular_predictions(splits: list[dict]) -> list[dict]:
    popular_bex_probs = []

    for split in splits:
        # print(split['truth'].shape)
        # get only unique products
        bex_products_in_split = list(set(split["train"]["BEX_PRODUCT_NO"].tolist()))
        n_products_in_split = len(bex_products_in_split)

        # print(f'value counts:')
        # print(split["train"]["BEX_PRODUCT_NO"].value_counts(normalize=True).to_dict().values())

        bex_probs_dict = (
            split["train"]["BEX_PRODUCT_NO"].value_counts(normalize=True).to_dict()
        )

        # print("check that this is the values")
        # print([prod for prod in bex_products_in_split])

        individual_probabilities = []

        popular_probabilities = [bex_probs_dict[prod] for prod in bex_products_in_split]

        for _ in range(split["validation"].shape[0]):
            individual_probabilities.append(popular_probabilities)

        popular_bex_probs.append(
            {
                "BEX_PRODUCT_NO_IN_SPLIT": bex_products_in_split,
                "probabilities_in_split": individual_probabilities,
            }
        )

    return popular_bex_probs


# input: prediction_probs: a list of probabilities for each of the items in the split (for that row so
#  same user can appear multiple times if they purchased multiple times on a single day)
# bex_products_list: list of items in the spli
# output: 1 or 0 in the metrics for that row


def evaluate(prediction_probs: list, bex_products_list: list, truth: str, k=5):
    # get the index of the item that was bought
    truth_index = bex_products_list.index(truth)

    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    top_k_probs = np.argpartition(prediction_probs, -k)[-k:].tolist()
    metrics = {}
    if truth_index in top_k_probs:
        metrics[f"top_{k}_accuracy"] = 1
    else:
        metrics[f"top_{k}_accuracy"] = 0
    return metrics


# input: s_data: a df containing testing data for each split. s_probs: a dict containing
# the list of bex products in that split and a list of lists containing probabilities for each row in that split
# output: average metrics from that split


def evaluate_split(s_data, s_probs, k=5):
    # print(f"this is s_data: {s_data}")

    # # print(s_data['BEX_PRODUCT_NO'].tolist())
    # # print(f"this is s_probs: {s_probs}")
    # print("this is probabilities in split:")
    # print(s_probs["probabilities_in_split"])

    rows_in_split = s_data.shape[0]
    bex_products_list = s_probs["BEX_PRODUCT_NO_IN_SPLIT"]

    no_bex_products = len(bex_products_list)
    # print(f"number of bex products in split: {no_bex_products}")

    for i in range(rows_in_split):
        # kolla så att indexeringen fungerar som jag tänkt
        # get the bex product number of what a customer bought
        truth = s_data.iloc[i]["BEX_PRODUCT_NO"]

        probs_list = s_probs["probabilities_in_split"][i]
        metrics = evaluate(
            prediction_probs=probs_list,
            bex_products_list=bex_products_list,
            truth=truth,
            k=k,
        )
        if i == 0:
            split_results_dict = metrics
        else:
            for key, _ in split_results_dict.items():
                split_results_dict[key] += metrics[key]

    # for key, _ in split_results_dict.items():
    #     split_results_dict[key] /= rows_in_split
    #     # only for the case of one
    #     actual = split_results_dict[key]

    # split_results_dict["expected"] = k / no_bex_products
    # split_results_dict["deviation"] = abs(actual - (k / no_bex_products))
    split_results_dict["observations"] = rows_in_split
    split_results_dict["bex_products"] = no_bex_products

    average = split_results_dict[f"top_{k}_accuracy"] / rows_in_split

    # print(
    #     f"results from this split: {split_results_dict}, average: {average}, expected: {k / no_bex_products}, deviation: {abs(average - (k / no_bex_products))}"
    # )

    return split_results_dict

    #     metrics = evaluate(prediction_probs=)


# input: list of dataframes containing the training and inference data in the different splits,
# list of dictionaries containing the bex products and probabilities in each split
# output: average metrics across the splits


def evaluate_splits(splits_data: list[pd.DataFrame], splits_probs: list[dict], k=5):
    # print(f"this is splits data {splits_data}")
    # print(f"this is splits probs {splits_probs}")

    # print(f"this is splits probs_i {splits_probs[0]}")

    no_splits = len(splits_data)

    for i in range(len(splits_data)):
        # print(splits_data[i])
        # print(splits_data[i]["train"]["date"].value_counts().sort_index())
        # print(splits_data[i]["train"]["date"].unique())

        metrics = evaluate_split(
            s_data=splits_data[i]["validation"], s_probs=splits_probs[i], k=k
        )

        if i == 0:
            splits_results_dict = metrics
        else:
            for key, _ in splits_results_dict.items():
                splits_results_dict[key] += metrics[key]

    # for key, _ in splits_results_dict.items():
    #     splits_results_dict[key] /= splits_results_dict["observations"]

    splits_results_dict[f"top_{k}_accuracy"] /= splits_results_dict[f"observations"]

    # print(
    #     f"this is the final results: {splits_results_dict}, expected: {k*no_splits / total_bex_products_count}"
    # )
    expected = k * no_splits / splits_results_dict["bex_products"]
    deviation = abs(splits_results_dict[f"top_{k}_accuracy"] - expected)

    print(
        f"this is the final results: {splits_results_dict}, expected: {expected}, deviation: {deviation}"
    )

    return splits_results_dict

if __name__ == "__main__":

    # input parsing

    parser = argparse.ArgumentParser(description="Recsys parameters")
    parser.add_argument("-t", "--thesis", action="store_true")
    parser.add_argument("-sgmw", "--stockholm_gbg_malmo_weather", action="store_true")

    args = parser.parse_args()

    start_time = time.time()

    # read the file and create new column data of datetime type
    order_data = pd.read_csv(f"Data/Orders/user_data_thesis.csv")

    order_data["ORDER_DATE"] = pd.to_datetime(order_data["ORDER_DATE"])
    order_data["date"] = order_data["ORDER_DATE"]

    # remove rows with customers than bought less than two items
    order_data = order_data[
        order_data["CUSTOMER_NO"].map(order_data["CUSTOMER_NO"].value_counts()) >= 2
        ]

    # filtrera bort det som inte är kläder, acessories, footware osv från datan när det är thesis
    if args.thesis:
        thesis_mask = order_data["PRODUCT_GROUP_LEVEL_1_DESCRIPTION"].isin(
            ["clothing", "footwear", "accessories"]
        )
        order_data = order_data[thesis_mask]

    n_splits = 100
    splits = split_data(order_data, n_days=30, n_splits=n_splits)

    popular_predictions = generate_popular_predictions(splits)

    popular_results = evaluate_splits(splits, popular_predictions, k=10)

    print(f"total duration: {time.time() - start_time} seconds")
