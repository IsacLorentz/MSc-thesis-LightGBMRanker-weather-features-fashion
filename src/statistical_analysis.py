import argparse
import os
import sys
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import scipy
import seaborn as sns
import statsmodels.tsa.stattools as smt

pio.renderers.default = "browser"

# input parsing
parser = argparse.ArgumentParser(description="Stockholm GBG comparison parameters")
parser.add_argument("c", type=str, help="the category")
parser.add_argument("w", type=str, help="the weather variable")

args = parser.parse_args()

category = args.c
weather_param = args.w

# check that valid category and weather params are entered

if not (
    Path(f"Data/Postort/stockholm_{category}_postort.csv").is_file()
    and Path(f"Data/Postort/gbg_{category}_postort.csv").is_file()
):
    print(f"incorrect clothing category: {category}")
    sys.exit()

if not weather_param in pd.read_csv("Data/Weather/stockholm_weather_merged.csv").columns:
    print(f"incorrect weather variable: {weather_param}")
    sys.exit()

################# filepaths #############################
title = f"Stockholm - Gothenburg"
title_ccf = f"Stockholm - Gothenburg"
filename = category + "_" + weather_param
directory = f"Outputs/StockholmGbgDelta/WithDuplicates/Postort/{filename}/base"
Path(f"{directory}").mkdir(parents=True, exist_ok=True)
filepath = f"{directory}/{filename}.txt"
imagepath = f"{directory}/{filename}.png"
imagepath_density = f"{directory}/{filename}_density.png"
lags_imagepath = f"{directory}/{filename}_lag.png"
ccf_imagepath = f"{directory}/{filename}_ccf.png"

# start printing the output to file
with open(filepath, "w") as f:
    f.write(f"title: {title}" + "\n")
    f.write(f"arguments: {args}" + "\n")

# get sales and weather data and process it

# sort by date
stockholm_sales = pd.read_csv(f"Data/Postort/stockholm_{category}_postort.csv")
stockholm_sales["date"] = stockholm_sales["ORDER_DATE_DIM_KEY"]
stockholm_sales = stockholm_sales.sort_values(by=["date"])

# sort by date
stockholm_total = pd.read_csv("Data/Postort/stockholm_total_sales_postort.csv")
stockholm_total = stockholm_total.sort_values(by=["ORDER_DATE_DIM_KEY"])

# get the total sales
stockholm_sales = stockholm_sales.merge(
    stockholm_total, how="inner", on="ORDER_DATE_DIM_KEY"
)
ph = 0
# get the share of sold items
stockholm_sales[f"{category}_part"] = (
    stockholm_sales["DAILY_QUANTITY"] / stockholm_sales["DAILY_TOTAL_QUANTITY"]
)

# sort sales by date
gbg_sales = pd.read_csv(f"Data/Postort/gbg_{category}_postort.csv")
gbg_sales["date"] = gbg_sales["ORDER_DATE_DIM_KEY"]
gbg_sales = gbg_sales.sort_values(by=["date"])

# sort by date
gbg_total = pd.read_csv("Data/Postort/gbg_total_sales_postort.csv")
gbg_total = gbg_total.sort_values(by=["ORDER_DATE_DIM_KEY"])

gbg_sales = gbg_sales.merge(gbg_total, how="inner", on="ORDER_DATE_DIM_KEY")

gbg_sales[f"{category}_part"] = (
    gbg_sales["DAILY_QUANTITY"] / gbg_sales["DAILY_TOTAL_QUANTITY"]
)

# sort by date
stockholm_weather = pd.read_csv("Data/Weather/stockholm_weather_merged.csv")
stockholm_weather = stockholm_weather.sort_values(by=["date"])

# sort by date
gbg_weather = pd.read_csv("Data/Weather/gbg_weather_merged.csv")
gbg_weather = gbg_weather.sort_values(by=["date"])

gbg_dates = gbg_sales[["ORDER_DATE_DIM_KEY"]].copy()
stockholm_dates = stockholm_sales[["ORDER_DATE_DIM_KEY"]].copy()

# make so that the sale and weather only considers the dates that have sales in both stockholm and gbg
stockholm_sales = stockholm_sales.merge(gbg_dates, how="inner", on="ORDER_DATE_DIM_KEY")
gbg_sales = gbg_sales.merge(stockholm_dates, how="inner", on="ORDER_DATE_DIM_KEY")

shared_dates = stockholm_sales[["ORDER_DATE_DIM_KEY"]].copy()
shared_dates.columns = ["datetime"]

stockholm_weather = stockholm_weather.merge(shared_dates, how="inner", on="datetime")
gbg_weather = gbg_weather.merge(shared_dates, how="inner", on="datetime")

# create the df that will take the differences in sale share and temperature
stockholm_sales_delta = stockholm_sales[["date"]].copy()
stockholm_sales_delta[f"{category}_share_difference_percent"] = (
    100
    * (stockholm_sales[f"{category}_part"] - gbg_sales[f"{category}_part"])
    / gbg_sales[f"{category}_part"]
)

stockholm_sales_delta[f"{weather_param}_difference"] = (
    stockholm_weather[weather_param] - gbg_weather[weather_param]
)

# will be used for duplicating the datapoints
stockholm_sales_delta["DAILY_QUANTITY"] = stockholm_sales["DAILY_QUANTITY"].astype(
    int
) + gbg_sales["DAILY_QUANTITY"].astype(int)

stockholm_sales_delta["DAILY_QUANTITY"] = stockholm_sales_delta[
    "DAILY_QUANTITY"
].astype(int)

# duplicate the datapoints
stockholm_sales_delta_dupl = stockholm_sales_delta.loc[
    stockholm_sales_delta.index.repeat(stockholm_sales_delta["DAILY_QUANTITY"])
]

stockholm_sales_delta_dupl = stockholm_sales_delta_dupl.drop(
    "DAILY_QUANTITY", axis=1
).reset_index(drop=True)

stockholm_sales_delta = stockholm_sales_delta_dupl.copy()

stockholm_sales_delta_w_outliers = stockholm_sales_delta.copy()

stockholm_sales_delta[f"{category}_share_difference_percent"] = stockholm_sales_delta[
    f"{category}_share_difference_percent"
].clip(lower=-100, upper=100)
# remove some 0-values
if weather_param in ["snow", "snowdepth"]:
    stockholm_sales_delta = stockholm_sales_delta[
        (stockholm_sales_delta[f"{weather_param}_difference"] != 0.0)
    ]

if weather_param == "precip":
    stockholm_sales_delta = stockholm_sales_delta[
        (stockholm_sales_delta[f"{weather_param}_difference"] != 0.000)
    ]

# Pearson correlation

pearson = scipy.stats.pearsonr(
    stockholm_sales_delta[f"{weather_param}_difference"],
    stockholm_sales_delta[f"{category}_share_difference_percent"],
)

relationship_exists_pearson = (
    "pearson relationship exists"
    if abs(pearson[0]) >= (2 / np.sqrt(stockholm_sales_delta.shape[0]))
    else "pearson relationship doesnt exists"
)
pearson_to_beat = 2 / np.sqrt(stockholm_sales_delta.shape[0])

with open(filepath, "a") as f:
    f.write("----------- Pearson correlation --------------------\n")
    # f.write(f'this is pearson 0::{abs(pearson[0])}\n')
    f.write(f"value to beat: {pearson_to_beat}\n")
    f.write(f"{relationship_exists_pearson}\n")
    f.write(str(pearson) + "\n")

# Spearman R

spearman = scipy.stats.spearmanr(
    a=stockholm_sales_delta[f"{weather_param}_difference"],
    b=stockholm_sales_delta[f"{category}_share_difference_percent"],
)

with open(filepath, "a") as f:
    f.write("----------- Spearman R correlation --------------------\n")
    f.write(f"{spearman}\n")

# linear regression

with open(filepath, "a") as f:
    f.write("--------------- Linear Regression ----------------\n")


fig = px.scatter(
    stockholm_sales_delta,
    x=f"{weather_param}_difference",
    y=f"{category}_share_difference_percent",
    title=title,
    trendline="ols",
)

fig.write_image(imagepath)

trendline = px.get_trendline_results(fig)
with open(filepath, "a") as f:
    f.write(str(trendline.iloc[0]["px_fit_results"].summary()) + "\n")
    f.write(
        "\n"
        f'slope: {trendline.iloc[0]["px_fit_results"].params[1]}, constant: {trendline.iloc[0]["px_fit_results"].params[0]}'
        + "\n"
    )


def kaggle_linear_assumptions():

    # The code in this function is adapted from a Kaggle notebook by Shruti Iyyer (Apache 2.0 licence)
    # https://www.kaggle.com/code/shrutimechlearn/step-by-step-assumptions-linear-regression

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    # 1st assumption: exists linear correlation - see scatter plot for this
    # 2nd assumption: X are independent of the residuals: there is no clear trend in the residuals as time goes on

    linreg_df = stockholm_sales_delta[
        [f"{weather_param}_difference", f"{category}_share_difference_percent"]
    ]

    x = linreg_df.drop([f"{category}_share_difference_percent"], axis=1)
    y = linreg_df[f"{category}_share_difference_percent"]

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X = sc.fit_transform(x)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.05
    )
    from sklearn import linear_model
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_train)

    residuals = y_train - y_pred
    mean_residuals = np.mean(residuals)

    sns.scatterplot(x=y_pred, y=residuals)
    plt.xlabel("y_pred/fitted value")
    plt.ylabel("Residuals")
    plt.axhline(0)
    plt.title("Residuals vs fitted values, daily values")
    plt.savefig(f"{directory}/{filename}_lintest2_1.png")
    plt.close()

    sns.scatterplot(x=range(len(y_pred)), y=residuals)
    plt.xlabel("X-index")
    plt.ylabel("Residual")
    plt.axhline(0)
    plt.title("2/4: Residuals vs X-indexes, daily values")
    plt.savefig(f"{directory}/{filename}_lintest2_2.png")
    plt.close()

    # 3rd assumption: error terms follow a mean 0 normal distribution, homoscedasticity
    with open(filepath, "a") as f:
        f.write(f"3rd assumption: (should be 0) Mean of Residuals: {mean_residuals}\n")

    sns.distplot(residuals, kde=True)
    plt.title("Normality of error terms/residuals, daily values")
    plt.savefig(f"{directory}/{filename}_lintest3_1.png")
    plt.close()

    # # QQ plot
    import statsmodels.api as sm

    residuals_qq = sm.qqplot(residuals, line="45")
    plt.savefig(f"{directory}/{filename}_lintest3_2.png")
    plt.close()

    # 4th assumption: no autocorrelation of residuals

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=y_pred, y=residuals, marker="o", color="blue")
    plt.xlabel("y_pred/predicted values")
    plt.ylabel("Residuals")
    plt.axhline(0)
    plt.title("(4th) Residuals vs fitted values plot for autocorrelation check")
    plt.savefig(f"{directory}/{filename}_lintest4_1.png")
    plt.close()

    import statsmodels.api as sm

    # autocorrelation
    sm.graphics.tsa.plot_acf(residuals, lags=40)
    plt.title("(4th) Autocorrelation of residuals")
    plt.savefig(f"{directory}/{filename}_lintest4_2.png")
    plt.close()

    # partial autocorrelation
    sm.graphics.tsa.plot_pacf(residuals, lags=40)
    plt.title("(4th) Partial autocorrelation of residuals")
    plt.savefig(f"{directory}/{filename}_lintest4_3.png")
    plt.close()


kaggle_linear_assumptions()

# lag vs pearson
lags = [i for i in range(-14, 15)]
correlations = []
p_values = []

for lag in lags:

    stockholm_sales_lagged = pd.read_csv(f"Data/Postort/stockholm_{category}_postort.csv")
    stockholm_sales_lagged["date"] = stockholm_sales_lagged["ORDER_DATE_DIM_KEY"]
    stockholm_sales_lagged = stockholm_sales_lagged.sort_values(by=["date"])

    stockholm_total_lagged = pd.read_csv("Data/Postort/stockholm_total_sales_postort.csv")
    stockholm_total_lagged = stockholm_total_lagged.sort_values(
        by=["ORDER_DATE_DIM_KEY"]
    )
    stockholm_sales_lagged = stockholm_sales_lagged.merge(
        stockholm_total_lagged, how="inner", on="ORDER_DATE_DIM_KEY"
    )

    stockholm_sales_lagged[f"{category}_part"] = (
        stockholm_sales_lagged["DAILY_QUANTITY"]
        / stockholm_sales_lagged["DAILY_TOTAL_QUANTITY"]
    )

    gbg_sales_lagged = pd.read_csv(f"Data/Postort/gbg_{category}_postort.csv")
    gbg_sales_lagged["date"] = gbg_sales_lagged["ORDER_DATE_DIM_KEY"]
    gbg_sales_lagged = gbg_sales_lagged.sort_values(by=["date"])

    gbg_total_lagged = pd.read_csv("Data/Postort/gbg_total_sales_postort.csv")
    gbg_total_lagged = gbg_total_lagged.sort_values(by=["ORDER_DATE_DIM_KEY"])

    gbg_sales_lagged = gbg_sales_lagged.merge(
        gbg_total_lagged, how="inner", on="ORDER_DATE_DIM_KEY"
    )
    gbg_sales_lagged[f"{category}_part"] = (
        gbg_sales_lagged["DAILY_QUANTITY"] / gbg_sales_lagged["DAILY_TOTAL_QUANTITY"]
    )

    gbg_dates_lagged = gbg_sales_lagged[["ORDER_DATE_DIM_KEY"]].copy()
    stockholm_dates_lagged = stockholm_sales_lagged[["ORDER_DATE_DIM_KEY"]].copy()

    # get overlapping dates only
    stockholm_sales_lagged = stockholm_sales_lagged.merge(
        gbg_dates_lagged, how="inner", on="ORDER_DATE_DIM_KEY"
    )
    gbg_sales_lagged = gbg_sales_lagged.merge(
        stockholm_dates_lagged, how="inner", on="ORDER_DATE_DIM_KEY"
    )

    shared_dates_lagged = stockholm_sales_lagged[["ORDER_DATE_DIM_KEY"]].copy()
    shared_dates_lagged.columns = ["datetime"]

    stockholm_weather_lagged = pd.read_csv("Data/Weather/stockholm_weather_merged.csv")
    stockholm_weather_lagged = stockholm_weather_lagged.sort_values(by=["date"])

    gbg_weather_lagged = pd.read_csv("Data/Weather/gbg_weather_merged.csv")
    gbg_weather_lagged = gbg_weather_lagged.sort_values(by=["date"])

    stockholm_weather_lagged = stockholm_weather_lagged.merge(
        shared_dates_lagged, how="inner", on="datetime"
    )
    gbg_weather_lagged = gbg_weather_lagged.merge(
        shared_dates_lagged, how="inner", on="datetime"
    )

    # perform the shifting / lag
    stockholm_weather_lagged[weather_param] = stockholm_weather_lagged[
        weather_param
    ].shift(lag)
    gbg_weather_lagged[weather_param] = gbg_weather_lagged[weather_param].shift(lag)

    # keep only rows that are not noll in the weather column (because we shifted)
    stockholm_weather_lagged = stockholm_weather_lagged[
        pd.notnull(stockholm_weather_lagged[weather_param])
    ]
    gbg_weather_lagged = gbg_weather_lagged[
        pd.notnull(gbg_weather_lagged[weather_param])
    ]
    stockholm_weather_lagged = stockholm_weather_lagged.reset_index(drop=True)
    gbg_weather_lagged = gbg_weather_lagged.reset_index(drop=True)

    print(f"shape after shift: {stockholm_weather_lagged.shape}")

    # get the dates for weather days that dont have nan in the weahter column
    non_nan_dates_stockholm = stockholm_weather_lagged[["datetime"]].copy()
    non_nan_dates_stockholm.columns = ["date"]
    non_nan_dates_gbg = gbg_weather_lagged[["datetime"]].copy()
    non_nan_dates_gbg.columns = ["date"]

    non_nan_dates_shared = non_nan_dates_stockholm.merge(
        non_nan_dates_gbg, how="inner", on="date"
    )

    print(
        "non Nan dates equal:"
        + str(
            np.array_equal(
                non_nan_dates_stockholm[["date"]], non_nan_dates_gbg[["date"]]
            )
        )
    )

    stockholm_sales_lagged = stockholm_sales_lagged.merge(
        non_nan_dates_shared, how="inner", on="date"
    )

    gbg_sales_lagged = gbg_sales_lagged.merge(
        non_nan_dates_shared, how="inner", on="date"
    )

    stockholm_sales_delta_lagged = stockholm_sales_lagged[["date"]].copy()
    stockholm_sales_delta_lagged[f"{category}_share_difference_percent"] = (
        100
        * (
            stockholm_sales_lagged[f"{category}_part"]
            - gbg_sales_lagged[f"{category}_part"]
        )
        / gbg_sales_lagged[f"{category}_part"]
    )

    stockholm_sales_delta_lagged[f"{weather_param}_difference"] = (
        stockholm_weather_lagged[weather_param] - gbg_weather_lagged[weather_param]
    )

    stockholm_sales_delta_lagged["DAILY_QUANTITY"] = stockholm_sales_lagged[
        "DAILY_QUANTITY"
    ].astype(int) + gbg_sales_lagged["DAILY_QUANTITY"].astype(int)

    stockholm_sales_delta_lagged["DAILY_QUANTITY"] = stockholm_sales_delta_lagged[
        "DAILY_QUANTITY"
    ].astype(int)

    stockholm_sales_delta_lagged_dupl = stockholm_sales_delta_lagged.loc[
        stockholm_sales_delta_lagged.index.repeat(
            stockholm_sales_delta_lagged["DAILY_QUANTITY"]
        )
    ]

    stockholm_sales_delta_lagged_dupl = stockholm_sales_delta_lagged_dupl.drop(
        "DAILY_QUANTITY", axis=1
    ).reset_index(drop=True)

    stockholm_sales_delta_lagged = stockholm_sales_delta_lagged_dupl.copy()

    stockholm_sales_delta_lagged[
        f"{category}_share_difference_percent"
    ] = stockholm_sales_delta_lagged[f"{category}_share_difference_percent"].clip(
        lower=-100, upper=100
    )

    if weather_param in ["snow", "snowdepth"]:
        stockholm_sales_delta_lagged = stockholm_sales_delta_lagged[
            (stockholm_sales_delta_lagged[f"{weather_param}_difference"] != 0.0)
        ]

    if weather_param in ["precip"]:
        stockholm_sales_delta_lagged = stockholm_sales_delta_lagged[
            (stockholm_sales_delta_lagged[f"{weather_param}_difference"] != 0.000)
        ]

    pearson = scipy.stats.pearsonr(
        stockholm_sales_delta_lagged[f"{weather_param}_difference"],
        stockholm_sales_delta_lagged[f"{category}_share_difference_percent"],
    )

    correlations.append(pearson[0])
    p_values.append(pearson[1])

pearson_df = pd.DataFrame(
    list(zip(lags, correlations, p_values)), columns=["lag", "pearson", "p_value"]
)

# Show the major grid and style it slightly.
plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
# Show the minor grid as well. Style it in very light gray as a thin,
# dotted line.
plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.6)
plt.minorticks_on()

sns.scatterplot(pearson_df, x="lag", y="pearson")
plt.axhline(y=(-1 * pearson_to_beat), color="r")
plt.axhline(y=pearson_to_beat, color="r")

plt.savefig(lags_imagepath)
plt.close()

lags_str_list = [
    f"lag: {lags[i]}, pearson: {correlations[i]}, p-value: {p_values[i]}\n"
    for i in range(len(lags))
]

with open(filepath, "a") as f:
    f.write(f"lagged pearson correlations and p values: \n")
    f.write("".join(lags_str_list))


# cross correlation ccf

import matplotlib.pylab as plt

backwards = smt.ccf(
    stockholm_sales_delta_w_outliers[f"{category}_share_difference_percent"],
    stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
    unbiased=False,
)[::-1]
forwards = smt.ccf(
    stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
    stockholm_sales_delta_w_outliers[f"{category}_share_difference_percent"],
    unbiased=False,
)
ccf_output = np.r_[backwards[:-1], forwards]

# change size so that it only contains middle 14 + 14 values
K = 28
middle_idx = len(ccf_output) // 2
ccf_output_sliced = [
    ccf_output[i] for i in range(middle_idx - K // 2, middle_idx + (K // 2) + 1)
]

plt.stem(
    range(-len(ccf_output_sliced) // 2, len(ccf_output_sliced) // 2),
    ccf_output_sliced,
)
plt.title(title_ccf)
plt.xlabel("Lag")
plt.ylabel("CCF")
# 95% UCL / LCL
plt.axhline(-1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")
plt.axhline(1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")

# Show the major grid and style it slightly.
plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
# Show the minor grid as well. Style it in very light gray as a thin,
# dotted line.
plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.6)
plt.minorticks_on()
# plt.grid(axis='x', color='0.95', which='both')

plt.savefig(ccf_imagepath)
plt.close()


# acf
from statsmodels.graphics.tsaplots import plot_acf

acf_path_own = f"{directory}/{filename}_acf_own.png"
# https://stackoverflow.com/questions/63491991/how-to-use-the-ccf-method-in-the-statsmodels-library
backwards = smt.acf(
    x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
)[::-1]
forwards = smt.acf(
    x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
)

acf_output = np.r_[backwards[:-1], forwards]

# change size so that it only contains middle 14 + 14 values
K = 28
middle_idx = len(acf_output) // 2
acf_output_sliced = [
    acf_output[i] for i in range(middle_idx - K // 2, middle_idx + (K // 2) + 1)
]

plt.stem(
    range(-len(acf_output_sliced) // 2, len(acf_output_sliced) // 2),
    acf_output_sliced,
)
plt.title(title_ccf)
plt.xlabel("Lag")
plt.ylabel("ACF")
# 95% UCL / LCL
plt.axhline(-1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")
plt.axhline(1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")

# Show the major grid and style it slightly.
plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
# Show the minor grid as well. Style it in very light gray as a thin,
# dotted line.
plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.6)
plt.minorticks_on()
# plt.grid(axis='x', color='0.95', which='both')

plt.savefig(acf_path_own)
plt.close()

# pacf
from statsmodels.graphics.tsaplots import plot_pacf

pacf_path = f"{directory}/{filename}_pacf.png"

pacf_path_own = f"{directory}/{filename}_pacf_own.png"
# https://stackoverflow.com/questions/63491991/how-to-use-the-ccf-method-in-the-statsmodels-library
backwards = smt.pacf(
    x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
)[::-1]
forwards = smt.pacf(
    x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
)
pacf_output = np.r_[backwards[:-1], forwards]

# change size so that it only contains middle 14 + 14 values
K = 28
middle_idx = len(pacf_output) // 2
pacf_output_sliced = [
    pacf_output[i] for i in range(middle_idx - K // 2, middle_idx + (K // 2) + 1)
]

plt.stem(
    range(-len(pacf_output_sliced) // 2, len(pacf_output_sliced) // 2),
    pacf_output_sliced,
)
plt.title(title_ccf)
plt.xlabel("Lag")
plt.ylabel("PACF")
# 95% UCL / LCL
plt.axhline(-1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")
plt.axhline(1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")

# Show the major grid and style it slightly.
plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
# Show the minor grid as well. Style it in very light gray as a thin,
# dotted line.
plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.6)
plt.minorticks_on()
# plt.grid(axis='x', color='0.95', which='both')

plt.savefig(pacf_path_own)
plt.close()


# repeat the program for simple moving averages
smas = [2, 3, 5, 7, 10, 14]
for sma in smas:

    ################# filepaths #############################
    title = f"Stockholm - Gothenburg SMA_{sma}"
    title_ccf = f"Stockholm - Gothenburg SMA_{sma}"
    filename = category + "_" + weather_param
    directory = f"output_directory/{filename}/SMA_{sma}"
    Path(f"{directory}").mkdir(parents=True, exist_ok=True)
    filepath = f"{directory}/{filename}.txt"
    imagepath = f"{directory}/{filename}.png"
    imagepath_density = f"{directory}/{filename}_density.png"
    lags_imagepath = f"{directory}/{filename}_lag.png"
    ccf_imagepath = f"{directory}/{filename}_ccf.png"

    # start printing the output to file
    with open(filepath, "w") as f:
        f.write(f"title: {title}" + "\n")
        f.write(f"arguments: {args}" + "\n")

    # get sales and weather data and process it

    # sort by date
    stockholm_sales = pd.read_csv(f"Data/Postort/stockholm_{category}_postort.csv")
    stockholm_sales["date"] = stockholm_sales["ORDER_DATE_DIM_KEY"]
    stockholm_sales = stockholm_sales.sort_values(by=["date"])

    # sort by date
    stockholm_total = pd.read_csv("Data/Postort/stockholm_total_sales_postort.csv")
    stockholm_total = stockholm_total.sort_values(by=["ORDER_DATE_DIM_KEY"])

    # get the total sales
    stockholm_sales = stockholm_sales.merge(
        stockholm_total, how="inner", on="ORDER_DATE_DIM_KEY"
    )

    # get the share of sold items
    stockholm_sales[f"{category}_part"] = (
        stockholm_sales["DAILY_QUANTITY"] / stockholm_sales["DAILY_TOTAL_QUANTITY"]
    )

    # sort sales by date
    gbg_sales = pd.read_csv(f"Data/Postort/gbg_{category}_postort.csv")
    gbg_sales["date"] = gbg_sales["ORDER_DATE_DIM_KEY"]
    gbg_sales = gbg_sales.sort_values(by=["date"])

    # sort by date
    gbg_total = pd.read_csv("Data/Postort/gbg_total_sales_postort.csv")
    gbg_total = gbg_total.sort_values(by=["ORDER_DATE_DIM_KEY"])

    gbg_sales = gbg_sales.merge(gbg_total, how="inner", on="ORDER_DATE_DIM_KEY")

    gbg_sales[f"{category}_part"] = (
        gbg_sales["DAILY_QUANTITY"] / gbg_sales["DAILY_TOTAL_QUANTITY"]
    )

    # sort by date
    stockholm_weather = pd.read_csv("Data/Weather/stockholm_weather_merged.csv")
    stockholm_weather = stockholm_weather.sort_values(by=["date"])

    # sort by date
    gbg_weather = pd.read_csv("Data/Weather/gbg_weather_merged.csv")
    gbg_weather = gbg_weather.sort_values(by=["date"])

    gbg_dates = gbg_sales[["ORDER_DATE_DIM_KEY"]].copy()
    stockholm_dates = stockholm_sales[["ORDER_DATE_DIM_KEY"]].copy()

    # make so that the sale and weather only considers the dates that have sales in both stockholm and gbg
    stockholm_sales = stockholm_sales.merge(
        gbg_dates, how="inner", on="ORDER_DATE_DIM_KEY"
    )
    gbg_sales = gbg_sales.merge(stockholm_dates, how="inner", on="ORDER_DATE_DIM_KEY")

    shared_dates = stockholm_sales[["ORDER_DATE_DIM_KEY"]].copy()
    shared_dates.columns = ["datetime"]

    stockholm_weather = stockholm_weather.merge(
        shared_dates, how="inner", on="datetime"
    )
    gbg_weather = gbg_weather.merge(shared_dates, how="inner", on="datetime")

    # create the df that will take the differences in sale share and temperature
    stockholm_sales_delta = stockholm_sales[["date"]].copy()
    stockholm_sales_delta[f"{category}_share_difference_percent"] = (
        100
        * (stockholm_sales[f"{category}_part"] - gbg_sales[f"{category}_part"])
        / gbg_sales[f"{category}_part"]
    )

    stockholm_sales_delta[f"{weather_param}_difference"] = (
        stockholm_weather[weather_param] - gbg_weather[weather_param]
    )

    # will be used for duplicating the datapoints
    stockholm_sales_delta["DAILY_QUANTITY"] = stockholm_sales["DAILY_QUANTITY"].astype(
        int
    ) + gbg_sales["DAILY_QUANTITY"].astype(int)

    stockholm_sales_delta["DAILY_QUANTITY"] = stockholm_sales_delta[
        "DAILY_QUANTITY"
    ].astype(int)

    # make the moving averages
    stockholm_sales_delta["date"] = pd.to_datetime(stockholm_sales_delta["date"])
    stockholm_sales_delta = stockholm_sales_delta.set_index("date")

    print(
        f"-----------this is stockholm sales delta befor MA for sma:{sma} -------------------"
    )
    print(stockholm_sales_delta.head(20))
    print(f"{sma}D\n")
    print(f'category\n')
    print(
        stockholm_sales_delta[f"{category}_share_difference_percent"]
        .rolling(window=f"{sma}D", min_periods=sma)
        .mean()
        .head(20)
    )
    print(f'weather:\n')
    print(stockholm_sales_delta[f"{weather_param}_difference"]
        .rolling(window=f"{sma}D", min_periods=sma)
        .mean().head(20))

    print("----------------------------------")

    stockholm_sales_delta[f"{category}_share_difference_percent_SMA_{sma}"] = (
        stockholm_sales_delta[f"{category}_share_difference_percent"]
        .rolling(window=f"{sma}D", min_periods=sma)
        .mean()
    )

    stockholm_sales_delta[f"{weather_param}_difference_SMA_{sma}"] = (
        stockholm_sales_delta[f"{weather_param}_difference"]
        .rolling(window=f"{sma}D", min_periods=sma)
        .mean()
    )

    stockholm_sales_delta = stockholm_sales_delta.dropna(
        subset=[
            f"{category}_share_difference_percent_SMA_{sma}",
            f"{weather_param}_difference_SMA_{sma}",
        ]
    )

    stockholm_sales_delta[
        f"{category}_share_difference_percent"
    ] = stockholm_sales_delta[f"{category}_share_difference_percent_SMA_{sma}"]
    stockholm_sales_delta[f"{weather_param}_difference"] = stockholm_sales_delta[
        f"{weather_param}_difference_SMA_{sma}"
    ]

    stockholm_sales_delta_dupl = stockholm_sales_delta.loc[
        stockholm_sales_delta.index.repeat(stockholm_sales_delta["DAILY_QUANTITY"])
    ]

    stockholm_sales_delta_dupl = stockholm_sales_delta_dupl.drop(
        "DAILY_QUANTITY", axis=1
    ).reset_index(drop=True)

    stockholm_sales_delta = stockholm_sales_delta_dupl.copy()

    stockholm_sales_delta_w_outliers = stockholm_sales_delta.copy()

    print(f"---------data points SMA_{sma}-----------------")
    print(stockholm_sales_delta.shape)

    stockholm_sales_delta[
        f"{category}_share_difference_percent"
    ] = stockholm_sales_delta[f"{category}_share_difference_percent"].clip(
        lower=-100, upper=100
    )
    # remove some 0-values
    if weather_param in ["snow", "snowdepth"]:
        stockholm_sales_delta = stockholm_sales_delta[
            (stockholm_sales_delta[f"{weather_param}_difference"] != 0.0)
        ]

    if weather_param == "precip":
        stockholm_sales_delta = stockholm_sales_delta[
            (stockholm_sales_delta[f"{weather_param}_difference"] != 0.000)
        ]

    # Pearson correlation

    pearson = scipy.stats.pearsonr(
        stockholm_sales_delta[f"{weather_param}_difference"],
        stockholm_sales_delta[f"{category}_share_difference_percent"],
    )

    relationship_exists_pearson = (
        "pearson relationship exists"
        if abs(pearson[0]) >= (2 / np.sqrt(stockholm_sales_delta.shape[0]))
        else "pearson relationship doesnt exists"
    )
    pearson_to_beat = 2 / np.sqrt(stockholm_sales_delta.shape[0])

    with open(filepath, "a") as f:
        f.write("----------- Pearson correlation --------------------\n")
        # f.write(f'this is pearson 0::{abs(pearson[0])}\n')
        f.write(f"value to beat: {pearson_to_beat}\n")
        f.write(f"{relationship_exists_pearson}\n")
        f.write(str(pearson) + "\n")

    # Spearman R

    spearman = scipy.stats.spearmanr(
        a=stockholm_sales_delta[f"{weather_param}_difference"],
        b=stockholm_sales_delta[f"{category}_share_difference_percent"],
    )

    with open(filepath, "a") as f:
        f.write("----------- Spearman R correlation --------------------\n")
        f.write(f"{spearman}\n")

    # linear regression

    with open(filepath, "a") as f:
        f.write("--------------- Linear Regression ----------------\n")

    fig = px.scatter(
        stockholm_sales_delta,
        x=f"{weather_param}_difference",
        y=f"{category}_share_difference_percent",
        title=title,
        trendline="ols",
    )
    # fig.show()
    fig.write_image(imagepath)

    trendline = px.get_trendline_results(fig)
    with open(filepath, "a") as f:
        f.write(str(trendline.iloc[0]["px_fit_results"].summary()) + "\n")
        f.write(
            "\n"
            f'slope: {trendline.iloc[0]["px_fit_results"].params[1]}, constant: {trendline.iloc[0]["px_fit_results"].params[0]}'
            + "\n"
        )

    def kaggle_linear_assumptions():

        # The code in this function is adapted from a Kaggle notebook by Shruti Iyyer (Apache 2.0 licence)
        # https://www.kaggle.com/code/shrutimechlearn/step-by-step-assumptions-linear-regression

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        # 1st assumption: exists linear correlation - see scatter plot for this
        # 2nd assumption: X are independent of the residuals: there is no clear trend in the residuals as time goes on

        linreg_df = stockholm_sales_delta[
            [f"{weather_param}_difference", f"{category}_share_difference_percent"]
        ]

        x = linreg_df.drop([f"{category}_share_difference_percent"], axis=1)
        y = linreg_df[f"{category}_share_difference_percent"]

        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        X = sc.fit_transform(x)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.05
        )
        print(f"-------- sklearn train test shape:")
        print(X_train.shape)
        print(y_train.shape)

        print(f"-------- own shape:")
        X_train, y_train = (
            stockholm_sales_delta[f"{weather_param}_difference"]
            .to_numpy()
            .reshape(-1, 1),
            stockholm_sales_delta[f"{category}_share_difference_percent"].to_numpy(),
        )
        print(X_train.shape)
        print(y_train.shape)

        from sklearn import linear_model
        from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                     r2_score)

        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)

        # print("R squared: {}".format(r2_score(y_true=y_train, y_pred=y_pred)))

        residuals = y_train - y_pred
        print(f"residuals shape: {residuals.shape}")
        print(f"len x: {len(range(len(y_pred)))}")
        mean_residuals = np.mean(residuals)
        # print("Mean of Residuals {}".format(mean_residuals))

        sns.scatterplot(x=y_pred, y=residuals)
        plt.xlabel("y_pred/fitted value")
        plt.ylabel("Residual")
        plt.axhline(0)
        # plt.title("2/4: Residuals vs fitted values")
        plt.title(f"Residuals vs fitted values, SMA_{sma}")
        plt.savefig(f"{directory}/{filename}_lintest2_1.png")
        plt.close()

        sns.scatterplot(x=range(len(y_pred)), y=residuals)
        plt.xlabel("X index")
        plt.ylabel("Residual")
        plt.axhline(0)
        # plt.title("2/4: Residuals vs X-indexes")
        plt.title(f"Residuals vs X-indexes, SMA_{sma}")
        plt.savefig(f"{directory}/{filename}_lintest2_2.png")
        plt.close()

        # 3rd assumption: error terms follow a mean 0 normal distribution, homoscedasticity
        with open(filepath, "a") as f:
            f.write(
                f"3rd assumption: (should be 0) Mean of Residuals: {mean_residuals}\n"
            )


        # normality of error terms/residuals

        sns.distplot(residuals, kde=True)
        # plt.title("(3rd) Normality of error terms/residuals")
        plt.title(f"Normality of error terms/residuals, SMA_{sma}")
        plt.savefig(f"{directory}/{filename}_lintest3_1.png")
        plt.close()

        # # QQ plot
        import statsmodels.api as sm

        residuals_qq = sm.qqplot(residuals, line="45")
        plt.savefig(f"{directory}/{filename}_lintest3_2.png")
        plt.close()

        # 4th assumption: no autocorrelation of residuals

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=y_pred, y=residuals, marker="o", color="blue")
        plt.xlabel("y_pred/fitted values")
        plt.ylabel("Residuals")
        plt.axhline(0)
        # plt.ylim(-10, 10)
        # plt.xlim(0, 26)
        # sns.lineplot(x=[0, 26], y=[0, 0], color="red")
        plt.title("(4th) Residuals vs fitted values plot for autocorrelation check")
        plt.savefig(f"{directory}/{filename}_lintest4_1.png")
        plt.close()

        import statsmodels.api as sm

        # autocorrelation
        sm.graphics.tsa.plot_acf(residuals, lags=40)
        plt.title("(4th) Autocorrelation of residuals")
        plt.savefig(f"{directory}/{filename}_lintest4_2.png")
        plt.close()

        # partial autocorrelation
        sm.graphics.tsa.plot_pacf(residuals, lags=40)
        plt.title("(4th) Partial autocorrelation of residuals")
        plt.savefig(f"{directory}/{filename}_lintest4_3.png")
        plt.close()

    kaggle_linear_assumptions()

    # cross correlation ccf

    import matplotlib.pylab as plt

    # https://stackoverflow.com/questions/63491991/how-to-use-the-ccf-method-in-the-statsmodels-library
    backwards = smt.ccf(
        stockholm_sales_delta_w_outliers[f"{category}_share_difference_percent"],
        stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
        unbiased=False,
    )[::-1]
    forwards = smt.ccf(
        stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
        stockholm_sales_delta_w_outliers[f"{category}_share_difference_percent"],
        unbiased=False,
    )
    ccf_output = np.r_[backwards[:-1], forwards]

    # change size so that it only contains middle 14 + 14 values
    K = 28
    middle_idx = len(ccf_output) // 2
    ccf_output_sliced = [
        ccf_output[i] for i in range(middle_idx - K // 2, middle_idx + (K // 2) + 1)
    ]

    plt.stem(
        range(-len(ccf_output_sliced) // 2, len(ccf_output_sliced) // 2),
        ccf_output_sliced,
    )
    plt.title(title_ccf)
    plt.xlabel("Lag")
    plt.ylabel("CCF")
    # 95% UCL / LCL
    plt.axhline(-1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")
    plt.axhline(1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")

    # Show the major grid and style it slightly.
    plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.6)
    plt.minorticks_on()
    # plt.grid(axis='x', color='0.95', which='both')

    plt.savefig(ccf_imagepath)
    plt.close()

    # acf
    from statsmodels.graphics.tsaplots import plot_acf

    acf_path_own = f"{directory}/{filename}_acf_own.png"
    # https://stackoverflow.com/questions/63491991/how-to-use-the-ccf-method-in-the-statsmodels-library
    backwards = smt.acf(
        x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
    )[::-1]
    forwards = smt.acf(
        x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
    )

    acf_output = np.r_[backwards[:-1], forwards]

    K = 28
    middle_idx = len(acf_output) // 2
    acf_output_sliced = [
        acf_output[i] for i in range(middle_idx - K // 2, middle_idx + (K // 2) + 1)
    ]

    plt.stem(
        range(-len(acf_output_sliced) // 2, len(acf_output_sliced) // 2),
        acf_output_sliced,
    )
    plt.title(title_ccf)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    # 95% UCL / LCL
    plt.axhline(-1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")
    plt.axhline(1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")

    # Show the major grid and style it slightly.
    plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.6)
    plt.minorticks_on()
    # plt.grid(axis='x', color='0.95', which='both')

    plt.savefig(acf_path_own)
    plt.close()

    # pacf
    from statsmodels.graphics.tsaplots import plot_pacf

    pacf_path = f"{directory}/{filename}_pacf.png"

    pacf_path_own = f"{directory}/{filename}_pacf_own.png"
    # https://stackoverflow.com/questions/63491991/how-to-use-the-ccf-method-in-the-statsmodels-library
    backwards = smt.pacf(
        x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
    )[::-1]
    forwards = smt.pacf(
        x=stockholm_sales_delta_w_outliers[f"{weather_param}_difference"],
    )
    pacf_output = np.r_[backwards[:-1], forwards]

    # change size so that it only contains middle 14 + 14 values
    K = 28
    middle_idx = len(pacf_output) // 2
    pacf_output_sliced = [
        pacf_output[i] for i in range(middle_idx - K // 2, middle_idx + (K // 2) + 1)
    ]

    plt.stem(
        range(-len(pacf_output_sliced) // 2, len(pacf_output_sliced) // 2),
        pacf_output_sliced,
    )
    plt.title(title_ccf)
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    # 95% UCL / LCL
    plt.axhline(-1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")
    plt.axhline(1.96 / np.sqrt(len(stockholm_sales_delta)), color="k", ls="--")

    # Show the major grid and style it slightly.
    plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    plt.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.6)
    plt.minorticks_on()
    # plt.grid(axis='x', color='0.95', which='both')

    plt.savefig(pacf_path_own)
    plt.close()
