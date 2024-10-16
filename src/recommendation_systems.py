import argparse
import random
import sys
import os
import time

import lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from average_precision import *

from lightgbm import early_stopping
from lightgbm.sklearn import LGBMRanker
from sklearn.model_selection import train_test_split

from recsys_functions import evaluate, split_data, split_product_data

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# function for applying date transformations
def date_transform_df(df: pd.DataFrame):
    df["dayofyear"] = df["ORDER_DATE"].dt.dayofyear
    df["dayofweek"] = df["ORDER_DATE"].dt.dayofweek
    df["month"] = df["ORDER_DATE"].dt.month
    df["week"] = df["ORDER_DATE"].dt.isocalendar().week.astype("int32")
    return df

def normalize_transform_df(df: pd.DataFrame):
    columns_to_normalize = ['1D_SALES_QTY_PRODUCT', '14D_SALES_QTY_PRODUCT', '1D_SALES_QTY_BRAND', '14D_SALES_QTY_BRAND', '1D_SALES_QTY_CAT3', '14D_SALES_QTY_CAT3']
    stored_max_values_column_and_day = {column: {} for column in columns_to_normalize}
    def normalize_row(row, column):
        date = row['ORDER_DATE'].strftime("%Y%m%d")
        d = stored_max_values_column_and_day[column]
        if date in d:
            return row[column] / stored_max_values_column_and_day[column][date]
        else:
            max_val = df[df['ORDER_DATE'] == row['ORDER_DATE']][column].max()
            stored_max_values_column_and_day[column][date] = max_val
            return row[column] / max_val

    for column in columns_to_normalize:
        df[column] = df.apply(lambda row: normalize_row(row, column), axis=1)

    return df

def top_10_and_100_hit_14D_transform_validation_df(train_df: pd.DataFrame, validation_df: pd.DataFrame):

    stored_validation_day_values_split = {}
    def calculate_14D_top_10_and_100_row(row):
        if row['ORDER_DATE'].strftime("%Y%m%d") in stored_validation_day_values_split:
            top_10_hit = 1 if row['BEX_PRODUCT_NO'] in stored_validation_day_values_split[row['ORDER_DATE'].strftime("%Y%m%d")]['TOP_10_BEX_PRODUCT_NOS'] else 0
            top_100_hit = 1 if row['BEX_PRODUCT_NO'] in stored_validation_day_values_split[row['ORDER_DATE'].strftime("%Y%m%d")]['TOP_100_BEX_PRODUCT_NOS'] else 0
            return top_10_hit, top_100_hit
        else:
            validation_df_date_mask = (train_df['ORDER_DATE'] >= (row['ORDER_DATE'] - pd.Timedelta(f"15D"))) & (train_df['ORDER_DATE'] <= (row['ORDER_DATE'] - pd.Timedelta(f"1D")))
            train_df_date_masked = train_df[validation_df_date_mask]
            top_10_items = train_df_date_masked['BEX_PRODUCT_NO'].value_counts()[:10].index.tolist()
            top_100_items = train_df_date_masked['BEX_PRODUCT_NO'].value_counts()[:100].index.tolist()
            stored_validation_day_values_split[row['ORDER_DATE'].strftime("%Y%m%d")] = {'TOP_10_BEX_PRODUCT_NOS': top_10_items, 'TOP_100_BEX_PRODUCT_NOS': top_100_items}
            top_10_hit = 1 if row['BEX_PRODUCT_NO'] in top_10_items else 0
            top_100_hit = 1 if row['BEX_PRODUCT_NO'] in top_100_items else 0
            return top_10_hit, top_100_hit

    validation_df['14_TOP_10_HIT'], validation_df['14D_TOP_100_HIT'] = zip(*validation_df.apply(lambda row: calculate_14D_top_10_and_100_row(row), axis=1))
    return validation_df

def ma_transform_weather_df(columns_for_ma: list, weather_df: pd.DataFrame):
    smas = [3,7]
    days_before = [1]
    days_before_smas = [7]
    weather_df = weather_df.set_index(pd.to_datetime(weather_df['date']))
    for column in columns_for_ma:
        for sma in smas:
            weather_df[f"{column}_SMA{sma}"] = weather_df[column].rolling(window=f"{sma}D", min_periods=sma).mean()
        # get value of the date before
        for d in days_before:
            weather_df[f"{column}_{d}D_before"] = weather_df[f"{column}"].shift(periods=d, freq='D')
            for sma in days_before_smas:
                weather_df[f"{column}_{d}D_before_SMA{sma}"] =  weather_df[f"{column}_{d}D_before"].rolling(window=f"{sma}D", min_periods=sma).mean()

        weather_df[f"{column}_1D_difference"] = weather_df[column] - weather_df[f"{column}_1D_before"]
        weather_df[f"{column}_SMA7_difference"] = weather_df[column] -  weather_df[f"{column}_1D_before_SMA7"]
        for d in days_before:
            weather_df = weather_df.drop(columns=[f"{column}_{d}D_before"])
            for sma in days_before_smas:
                weather_df = weather_df.drop(columns=[f"{column}_{d}D_before_SMA{sma}"])

    weather_df = weather_df.sort_index(axis=1)
    return weather_df

def get_weather(city_encodings, weather_files, sales_df):
    columns_for_ma = ['feelslike', 'solarenergy', 'precip', 'snow', 'snowdepth', 'windspeed']
    columns_to_keep = columns_for_ma + ['date','CITY']
    weather_dfs = {}
    for city, filepath in weather_files.items():
        weather_dfs[city] = pd.read_csv(filepath)
        weather_dfs[city]['CITY'] = city_encodings.index(city)
        weather_dfs[city] = weather_dfs[city][columns_to_keep]
        weather_dfs[city] = ma_transform_weather_df(columns_for_ma=columns_for_ma, weather_df=weather_dfs[city])
    weather_data_df = pd.concat(weather_dfs)
    weather_data_df = weather_data_df.rename(columns={'date' : 'ORDER_DATE'})
    weather_data_df['ORDER_DATE'] = pd.to_datetime(weather_data_df['ORDER_DATE'])
    sales_df = sales_df.merge(weather_data_df, on=['ORDER_DATE', 'CITY'])
    return sales_df

def print_correlated_features(correlation_matrix_df, threshold):
    highly_correlated_features_list = []
    for row_index, row in correlation_matrix_df.iterrows():
        for column_index, value in row.items():
            if value >= threshold:
                if row_index != column_index and (column_index, row_index) not in highly_correlated_features_list:
                    highly_correlated_features_list.append((row_index, column_index))
    return highly_correlated_features_list

def get_popular_products(df: pd.DataFrame, date, days=30, no_products = 2000) -> list:
    df = df[(df['ORDER_DATE'] <= (date - pd.Timedelta(f"1D"))) & (df['ORDER_DATE'] >= (date - pd.Timedelta(f"{days+1}D")))]
    return df['BEX_PRODUCT_NO'].value_counts()[:no_products].index.tolist()

if __name__ == "__main__":
    # Sklearn random state
    RANDOM_STATE = 42
    weather_files = {'STOCKHOLM': "Data/Weather/stockholm_weather_new.csv",
                     'GÖTEBORG': "Data/Weather/gbg_weather_new.csv",
                     'MALMÖ': "Data/Weather/malmo_weather_new.csv"}
    # input parsing
    parser = argparse.ArgumentParser(description="ML Recsys parameters")
    parser.add_argument("-t", "--thesis", action="store_true")
    parser.add_argument("-w", "--weather", action="store_true")
    parser.add_argument("n_days", type=int, help="number of days")
    parser.add_argument("n_splits", type=int, help="number of splits")
    parser.add_argument("comment", type=str)

    args = parser.parse_args()

    start = time.time()

    n_splits = args.n_splits
    n_days = args.n_days
    comment = args.comment
    start_date = "2021-01-01"
    NUM_PRODUCTS = 1000

    directory = f"Outputs/Ranker/Diff/Big"
    thesis_table_v2_file = "path.csv"

    thesis_item_table_file = "path.csv"
    output_filename = f'{directory}/{os.path.basename(__file__)[:-3]}_{n_days}_days_{n_splits}_splits_{"weather" if args.weather else "no_weather"}_{NUM_PRODUCTS}_products.txt'

    sales_and_products_features = pd.read_csv(
        thesis_table_v2_file
    )

    product_trends_inference_features = pd.read_csv(
        thesis_item_table_file
    )

    product_trends_inference_features = product_trends_inference_features[
        product_trends_inference_features["BEX_PRODUCT_NO"].isin(
            sales_and_products_features["BEX_PRODUCT_NO"].values.tolist()
        )
    ]

    with open(output_filename, "w") as f:
        f.write(f'filename: {sys.argv[0]}, n_splits: {n_splits}, n_days: {n_days}, comment: {comment}, thesis table v2 file: {thesis_table_v2_file}\n')
        f.close()

    sales_and_products_features["ORDER_DATE"] = pd.to_datetime(
        sales_and_products_features["ORDER_DATE"]
    )
    # country code is useless and 14D_TOP_100 cannot be used in model since its an array
    sales_and_products_features = sales_and_products_features.drop(
        columns=["14D_TOP_100", "COUNTRY_CODE"]
    )

    # apply the training feature engineering here in order to not have to repeat it per split
    sales_and_products_features = date_transform_df(sales_and_products_features)
    sales_and_products_features = normalize_transform_df(sales_and_products_features)

    # split the training data and make so only with label 1 in the validation data
    sales_and_products_features_splits = split_data(
        sales_and_products_features,
        n_days=n_days,
        n_splits=n_splits,
        start_date=start_date,
        labels=True,
        encode=True,
        thesis=args.thesis,
    )

    product_trends_inference_features["ORDER_DATE"] = pd.to_datetime(
        product_trends_inference_features["ORDER_DATE"]
    )

    product_trends_inference_features = split_product_data(
        product_trends_inference_features,
        n_days=n_days,
        n_splits=n_splits,
        start_date=start_date,
    )

    with open(output_filename, "a") as f:
        f.write(f"time to do data preprocessing: {time.time() - start}\n")
        f.close()

    # store evaluation results
    total_rows = 0
    total_users = 0
    results = {}

    random_top_10_sum = 0
    random_top_100_sum = 0

    split_count = 0
    for split in sales_and_products_features_splits:
        split_start_time = time.time()
        split_total_ranker_predict_duration_counter = 0.0
        split_total_evaluation_duration_counter = 0.0

        rows_in_split = 0
        split_results = {}
        split_inference_constants = {}

        # used for fitting ligthgbm model
        split["train"] = split["train"].sort_values(by=["ORDER_DATE", "CUSTOMER_NO"])
        split["validation"] = split["validation"].sort_values(by=["CUSTOMER_NO"])

        if args.weather:
            split['train'] = get_weather(city_encodings=split['encodings']['CITY'].tolist(), weather_files=weather_files, sales_df=split['train'])

        if split_count == 0:
            categorical_features = ["CITY",
                "COLOUR",
                "GENDER_DESCRIPTION",
                "BRAND",
                "COLLECTION",
                "PRODUCT_GROUP_LEVEL_1_DESCRIPTION",
                "PRODUCT_GROUP_LEVEL_2_DESCRIPTION",
                "PRODUCT_GROUP_LEVEL_3_DESCRIPTION",]

            drop_columns_for_corr_matrix = categorical_features + ["ORDER_DATE",
                    "LABEL",
                    "CUSTOMER_NO",
                    "BEX_PRODUCT_NO",]

            corr = split['train'].drop(columns=drop_columns_for_corr_matrix).corr().dropna(how='all', axis=1).dropna(how='all')
            # sns.heatmap(corr, annot=True, cmap=plt.cm.Reds, fmt='.2f')
            # plt.show()

            corr_features = print_correlated_features(corr, 0.75)

            with open(output_filename, "a") as f:
                f.write(f'highly correlated features: {corr_features} \n')

            fig = px.imshow(corr, text_auto=True)
            # fig.update_layout(
            #     title=dict(text=f"{output_filename[15:-4]}", automargin=True, yref='paper')
            # )
            fig.show()

        ranker = LGBMRanker(
            objective="lambdarank",
            metric="map",
            boosting_type="gbdt",
            max_depth=7,
            n_estimators=300,
            importance_type="gain",
            verbose=1000,
        )

        X_train, X_test, _, _ = train_test_split(
            split["train"],
            split["train"]["LABEL"].values.tolist(),
            test_size=0.1,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        X_train = X_train.sort_values(by=["ORDER_DATE", "CUSTOMER_NO"])
        X_test = X_test.sort_values(by=["ORDER_DATE", "CUSTOMER_NO"])

        y_train = X_train["LABEL"].values.tolist()
        y_test = X_test["LABEL"].values.tolist()

        X_train_groups = (
            X_train.copy()
            .groupby(["ORDER_DATE", "CUSTOMER_NO"])["BEX_PRODUCT_NO"]
            .count()
            .values
        )

        X_test_groups = (
            X_test.copy()
            .groupby(["ORDER_DATE", "CUSTOMER_NO"])["BEX_PRODUCT_NO"]
            .count()
            .values
        )

        X_train = X_train.drop(
            columns=[
                "ORDER_DATE",
                "LABEL",
                "CUSTOMER_NO",
                "BEX_PRODUCT_NO",
            ],
            inplace=False,
        )
        X_test = X_test.drop(
            columns=[
                "ORDER_DATE",
                "LABEL",
                "CUSTOMER_NO",
                "BEX_PRODUCT_NO",
            ],
            inplace=False,
        )

        ranker = ranker.fit(
            X=X_train,
            y=y_train,
            group=X_train_groups,
            feature_name=X_train.columns.tolist(),
            categorical_feature=[
                "CITY",
                "COLOUR",
                "GENDER_DESCRIPTION",
                "BRAND",
                "COLLECTION",
                "PRODUCT_GROUP_LEVEL_1_DESCRIPTION",
                "PRODUCT_GROUP_LEVEL_2_DESCRIPTION",
                "PRODUCT_GROUP_LEVEL_3_DESCRIPTION",
            ],
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=["binary_error", "map"],
            eval_group=[X_train_groups, X_test_groups],
            callbacks=[early_stopping(stopping_rounds=50, first_metric_only=True)]
        )

        feature_importance_df = pd.DataFrame(sorted(zip(ranker.feature_importances_,X_train.columns),reverse=True), columns=['Value', 'Feature'])

        if split_count == 0:
            with open(output_filename, "a") as f:
                f.write('-------------- feature importance: ---------------\n')
                f.write(feature_importance_df.to_string() + '\n')
                f.close()

        lightgbm.plot_importance(ranker, ylabel='Features', xlabel='Feature importance (gain)', title=None)
        plt.show()
        with open(output_filename, "a") as f:
            if split_count == 0:
                f.write(f"time to fit ranker (+ feature importance): {time.time() - split_start_time}\n")
            else:
                f.write(f"time to fit ranker: {time.time() - split_start_time}\n")
            f.close()

        inference_and_evaluation_start_time = time.time()
        # use only the top 1000 products in order to speed up the inference
        popular_products_bex_products_encoded = get_popular_products(split['train'], date=split['validation']['ORDER_DATE'].tolist()[0], no_products=NUM_PRODUCTS)

        # decoded bex product numbers needed for comparison with the bex product numbers in product features
        bex_products_list_decoded = [
            split["encodings"]["BEX_PRODUCT_NO"][i]
            for i in popular_products_bex_products_encoded
        ]

        validation_order_date = split["validation"]["ORDER_DATE"].iloc[0]

        # mask on the validation date and bex product numbers present in the training dataset
        product_trends_inference_features_mask = (
            product_trends_inference_features["BEX_PRODUCT_NO"].isin(
                bex_products_list_decoded
            )
        ) & (product_trends_inference_features["ORDER_DATE"] == validation_order_date)

        # cannot write over product features because things are different between the splits
        product_trends_inference_features_masked = (
            product_trends_inference_features.copy()[
                product_trends_inference_features_mask
            ]
        )

        product_trends_inference_features_masked_bex_decoded = product_trends_inference_features_masked["BEX_PRODUCT_NO"].tolist()

        # get the encoded bex product number for every product
        product_trends_inference_features_masked_bex_encoded = [
            split["encodings"]["BEX_PRODUCT_NO"].tolist().index(bex_product_name)
            for bex_product_name in product_trends_inference_features_masked_bex_decoded
        ]

        # set the bex product no column to the encoded values
        product_trends_inference_features_masked[
            "BEX_PRODUCT_NO"
        ] = product_trends_inference_features_masked_bex_encoded

        category_features_columns = ['BEX_PRODUCT_NO', 'COLOUR', 'GENDER_DESCRIPTION', 'BRAND', 'COLLECTION', 'PRODUCT_GROUP_LEVEL_1_DESCRIPTION', 'PRODUCT_GROUP_LEVEL_2_DESCRIPTION', 'PRODUCT_GROUP_LEVEL_3_DESCRIPTION']

        # perform the mask and drop the duplicated products
        category_features = split['train'][category_features_columns].drop_duplicates(subset=["BEX_PRODUCT_NO"], keep="last", ignore_index=True)


        product_trends_inference_features_masked = (
            product_trends_inference_features_masked.merge(
                category_features, on=["BEX_PRODUCT_NO"]
            )
        )

        # apply date transformation here before dropping the order_date
        product_trends_inference_features_masked = date_transform_df(product_trends_inference_features_masked)
        product_trends_inference_features_masked = normalize_transform_df(product_trends_inference_features_masked)
        product_trends_inference_features_masked = top_10_and_100_hit_14D_transform_validation_df(train_df=split['train'],
                                                                             validation_df=product_trends_inference_features_masked)

        validation_users = split["validation"]["CUSTOMER_NO"].unique().tolist()
        users_in_split = len(validation_users)

        with open(output_filename, "a") as f:
            f.write(f'data preprocessing for inference duration {time.time() - inference_and_evaluation_start_time}\n')
            f.close()
        # go over each user in the validation set and perform the predictions
        for user in validation_users:

            # will be needed for each user to store the saved values
            user_city = split["validation"].loc[
                split["validation"]["CUSTOMER_NO"] == user
                ]["CITY"].values.tolist()[0]

            if user == validation_users[0]:
                user_start_time = time.time()

            # values only need to be stored for the 3 cities since we have no features for individual users
            if f'{user_city}_ranking' not in split_inference_constants:

                validation_products = product_trends_inference_features_masked.copy()

                # set the city and customer_no columns to the values of the user
                validation_products["CITY"] = user_city
                validation_products["CUSTOMER_NO"] = user

                # merge with weather data in case of weather
                if args.weather:
                    validation_products = get_weather(city_encodings=split['encodings']['CITY'].tolist(), weather_files=weather_files,
                                          sales_df=validation_products)
                # needed to set the order of validation_products to the same order as X used for fitting the LGBMRanker
                X_train_column_order = X_train.columns.tolist()
                validation_products_bex_product_list = validation_products['BEX_PRODUCT_NO'].tolist()

                split_inference_constants[f'validation_products_bex_product_list'] = validation_products_bex_product_list

                validation_products = validation_products.drop(
                    columns=["ORDER_DATE", "CUSTOMER_NO", "BEX_PRODUCT_NO"]
                )

                validation_products = validation_products.reindex(
                    columns=X_train_column_order
                )


                ranker_predict_timer = time.time()
                user_ranking = ranker.predict(X=validation_products)

                # store the ranking per city
                split_inference_constants[f'{user_city}_ranking'] = user_ranking

                split_inference_constants[f'{user_city}_map_predicted'] = [product for _, product in
                                                                                 sorted(zip(split_inference_constants[
                                                                                                f'{user_city}_ranking'],
                                                                                            validation_products_bex_product_list),
                                                                                        key=lambda pair: pair[0],
                                                                                        reverse=True)]

                # here do the non user/city-specific stuff
                if user == validation_users[0]:
                    # get popular products only
                    popular_30_days_df = split['train'].copy()
                    validation_date = split['validation']['ORDER_DATE'].tolist()[0]
                    popular_30_days_df = popular_30_days_df[(popular_30_days_df['ORDER_DATE'] <= (validation_date - pd.Timedelta(f"1D"))) & (
                                popular_30_days_df['ORDER_DATE'] >= (validation_date - pd.Timedelta(f"{30 + 1}D")))]
                    bex_popular_probs_dict = (
                        popular_30_days_df['BEX_PRODUCT_NO'].value_counts(normalize=True).to_dict()
                    )

                    unique_bex_products_list_30D = popular_30_days_df['BEX_PRODUCT_NO'].unique().tolist()
                    popular_probabilities = [bex_popular_probs_dict[prod] for prod in unique_bex_products_list_30D]
                    split_inference_constants['unique_bex_products_list_30D'] = unique_bex_products_list_30D
                    split_inference_constants['popular_probabilities'] = popular_probabilities

                    # popular MAP
                    # get the popularity based ranking and also the corresponding product numbers
                    ranked_bex_popular_products = [product for _, product in
                                                   sorted(zip(popular_probabilities, unique_bex_products_list_30D),
                                                          key=lambda pair: pair[0], reverse=True)]
                    split_inference_constants['user_map_predicted_popular'] = ranked_bex_popular_products

                    # popular MRR also only need to be done once
                    argsorted_popular_probabilities = np.argsort(split_inference_constants['popular_probabilities'])[
                                                      ::-1].tolist()
                    split_inference_constants['argsorted_popular_probabilities'] = argsorted_popular_probabilities

            true_labels = split["validation"].loc[
                    split["validation"]["CUSTOMER_NO"] == user
                ]["BEX_PRODUCT_NO"].values.tolist()

            # map calculation
            K = 10

            user_map_actual = true_labels

            # perform evaluation on the individual products that the user has bought
            evaluate_timer = time.time()
            for label in true_labels:
                if label in split_inference_constants['unique_bex_products_list_30D']:
                    top_10_acc_popular = list(
                        evaluate(
                            prediction_probs=split_inference_constants['popular_probabilities'],
                            bex_products_list=split_inference_constants['unique_bex_products_list_30D'],
                            truth=label,
                            k=10,
                        ).values()
                    )[0]
                    top_100_acc_popular = list(
                        evaluate(
                            prediction_probs=split_inference_constants['popular_probabilities'],
                            bex_products_list=split_inference_constants['unique_bex_products_list_30D'],
                            truth=label,
                            k=100,
                        ).values()
                    )[0]
                else:
                    top_10_acc_popular, top_100_acc_popular = 0,0

                if label in validation_products_bex_product_list:
                    top_10_acc = list(
                        evaluate(
                            prediction_probs=user_ranking,
                            bex_products_list=validation_products_bex_product_list,
                            truth=label,
                            k=10,
                        ).values()
                    )[0]
                    top_100_acc = list(
                        evaluate(
                            prediction_probs=user_ranking,
                            bex_products_list=validation_products_bex_product_list,
                            truth=label,
                            k=100,
                        ).values()
                    )[0]
                    # get index from validation product list of label
                    label_idx = validation_products_bex_product_list.index(label)
                    argsorted_user_ranking = np.argsort(user_ranking)[::-1].tolist()
                    # this is the real mrr
                    mrr = 1 / (argsorted_user_ranking.index(label_idx) + 1)

                else:
                    top_10_acc, top_100_acc, mrr = 0, 0, 0

                if label in split_inference_constants['unique_bex_products_list_30D']:
                    mrr_popular = 1 / (split_inference_constants['user_map_predicted_popular'].index(label) + 1)
                    mrr_random_popular = 1 / (random.sample(split_inference_constants['user_map_predicted_popular'],len(split_inference_constants['user_map_predicted_popular'])).index(label) + 1)
                else:
                    mrr_popular, mrr_random_popular = 0,0

                if rows_in_split == 0:
                    split_results["top_10_accuracy"] = top_10_acc
                    split_results["top_100_accuracy"] = top_100_acc
                    split_results["top_10_accuracy_popular"] = top_10_acc_popular
                    split_results["top_100_accuracy_popular"] = top_100_acc_popular
                    split_results["MRR"] = mrr
                    split_results["MRR_popular"] = mrr_popular
                    split_results["MRR_random_popular"] = mrr_random_popular
                else:
                    split_results["top_10_accuracy"] += top_10_acc
                    split_results["top_100_accuracy"] += top_100_acc
                    split_results["top_10_accuracy_popular"] += top_10_acc_popular
                    split_results["top_100_accuracy_popular"] += top_100_acc_popular
                    split_results["MRR"] += mrr
                    split_results["MRR_popular"] += mrr_popular
                    split_results["MRR_random_popular"] += mrr_random_popular

                total_rows += 1
                rows_in_split += 1

            # here once per user
            split_total_evaluation_duration_counter += (time.time() - evaluate_timer)
            if 'MAP_actuals' in split_results:
                split_results['MAP_actuals'].append(user_map_actual)
                split_results['MAP_predicteds'].append(user_city)
                split_results['MAP_predicteds_popular'].append(None)

            else:
                split_results['MAP_actuals'] = [user_map_actual]
                split_results['MAP_predicteds'] = [user_city]
                split_results['MAP_predicteds_popular'] = [None]

        # here once per split

        split_results['users_count'] = users_in_split

        split_MAP_predicteds_list = [split_inference_constants[f'{user_city}_map_predicted'] for user_city in split_results["MAP_predicteds"]]
        split_MAP_predicteds_popular_list = [split_inference_constants['user_map_predicted_popular'] for _ in split_results["MAP_predicteds_popular"]]
        split_MAP_predicteds_random_list = [random.sample(split_inference_constants['user_map_predicted_popular'],len(split_inference_constants['user_map_predicted_popular'])) for _ in split_results["MAP_predicteds_popular"]]

        split_results['MAP'] = split_results['users_count'] * mapk(actual=split_results["MAP_actuals"], predicted=split_MAP_predicteds_list, k=K)
        split_results['MAP_popular'] = split_results['users_count'] * mapk(actual=split_results["MAP_actuals"], predicted=split_MAP_predicteds_popular_list, k=K)
        split_results['MAP_random'] = split_results['users_count'] * mapk(actual=split_results["MAP_actuals"],
                                                                           predicted=split_MAP_predicteds_random_list,
                                                                           k=K)

        split_keys_to_keep = ['top_10_accuracy', 'top_100_accuracy', 'top_10_accuracy_popular', 'top_100_accuracy_popular', 'MRR', 'MRR_popular', 'MRR_random_popular', 'users_count', 'MAP', 'MAP_popular','MAP_random']
        split_results_keep = {k: split_results[k] for k in split_keys_to_keep if k in split_results}
        split_results = split_results_keep
        if total_rows == rows_in_split:  # if first split
            total_results = split_results
        else:
            for key, _ in split_results.items():
                total_results[key] += split_results[key]

        with open(output_filename, "a") as f:
            f.write(f"split evaluation date: {split['validation']['ORDER_DATE'][0].strftime('%Y-%m-%d')}\n")
            f.write(
                f'top_10_accuracy for this split: {split_results["top_10_accuracy"] /  rows_in_split}\n'
            )
            f.write(
                f'top_100_accuracy for this split: {split_results["top_100_accuracy"] /  rows_in_split}\n'
            )
            f.write(f'MRR for this split: {split_results["MRR"] / rows_in_split}\n')
            f.write(f'MRR popular for this split: {split_results["MRR_popular"] / rows_in_split}\n')
            f.write(f'MRR random popular for this split: {split_results["MRR_random_popular"] / rows_in_split}\n')

            del split_MAP_predicteds_list
            del split_MAP_predicteds_popular_list
            del split_MAP_predicteds_random_list

        no_products_in_split = len(validation_products_bex_product_list)

        random_top_10_split = 10 / no_products_in_split
        random_top_100_split = 100 / no_products_in_split

        random_top_10_sum += random_top_10_split
        random_top_100_sum += random_top_100_split

        split_count += 1

        with open(output_filename, "a") as f:
            f.write(f'split count i: {split_count}\n')
            f.close()

        with open(output_filename, "a") as f:
            f.write(
                f'top_10_accuracy expected for this split: {random_top_10_split}, deviation: {abs((split_results["top_10_accuracy"] / rows_in_split) - random_top_10_split)}\n'
            )
            f.write(
                f'top_100_accuracy expected for this split: {random_top_100_split}, deviation: {abs((split_results["top_100_accuracy"] / rows_in_split) - random_top_100_split)}\n'
            )

            f.write(f'time to perform  ranker.predict: {split_total_ranker_predict_duration_counter}\n')
            f.write(f'time to perform  evaluation: {split_total_evaluation_duration_counter}\n')
            f.write(f"time to perform inference + evaluation (full step): {time.time() - inference_and_evaluation_start_time}\n")
            f.write(f"time for split: {time.time() - split_start_time}\n")
            f.close()

    with open(output_filename, "a") as f:
        f.write(
            f'top_10_accuracy for whole program: {total_results["top_10_accuracy"] /  total_rows}\n'
        )
        f.write(
            f'top_100_accuracy for whole program: {total_results["top_100_accuracy"] /  total_rows}\n'
        )
        f.write(
            f'popular products top_10_accuracy for whole program: {total_results["top_10_accuracy_popular"] / total_rows}\n'
        )
        f.write(
            f'popular products top_100_accuracy for whole program: {total_results["top_100_accuracy_popular"] / total_rows}\n'
        )
        # need to use split count in case any of the splits were non-valid / empty
        f.write(f"average random top_10 accuracy: {random_top_10_sum / split_count}\n")
        f.write(f"average random top_100 accuracy: {random_top_100_sum / split_count}\n")
        f.write(
            f'MRR for whole program: {total_results["MRR"] / total_rows}\n'
        )
        f.write(
            f'MRR popular for whole program: {total_results["MRR_popular"] / total_rows}\n'
        )
        f.write(
            f'MRR random popular for whole program: {total_results["MRR_random_popular"] / total_rows}\n'
        )

        f.write(
            f'MAP_{K} for whole program: {total_results["MAP"] / total_results["users_count"]}\n'
        )

        f.write(
            f'MAP_{K} popular for whole program: {total_results["MAP_popular"] / total_results["users_count"]}\n'
        )

        f.write(
            f'MAP_{K} random for whole program: {total_results["MAP_random"] / total_results["users_count"]}\n'
        )

        f.write(f"time to run whole program: {time.time() - start}\n")
        f.close()
