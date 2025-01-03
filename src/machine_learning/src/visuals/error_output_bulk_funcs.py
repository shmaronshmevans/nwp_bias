import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statistics as st
from datetime import datetime
from scipy.stats import gaussian_kde
import matplotlib.dates as mdates
import calendar
from sklearn.metrics import r2_score


def myround(x, base):
    return base * round(x / base)


# function to get unique values
def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def round_small(full_df, met_col, rounded_base):
    """
    Groups errors into temperature buckets by rounding the met_col values based on conditions and calculates the mean
    absolute error (MAE) across all 'diff' columns for each bucket.

    Parameters:
        full_df (pd.DataFrame): DataFrame containing the data.
        met_col (str): The column used for temperature rounding (e.g., temperature column).
        rounded_base (int or float): The base to round small values (0 <= value < 1.0).

    Returns:
        temp_df (pd.Series): DataFrame containing the mean absolute error for each temperature bucket.
        instances (pd.Series): Series containing the counts of instances in each temperature bucket.
    """
    # Step 1: Round `met_col` values based on conditions and create a list of rounded values
    temps = []
    for i, _ in enumerate(full_df[met_col]):
        value = full_df[met_col].iloc[i]
        if 0 <= value < 1.0:
            rounded = round(value, rounded_base)
        else:
            rounded = myround(value, 1)
        temps.append(rounded)

    # Step 2: Identify unique temperature buckets
    unique_temps = unique(temps)

    # Step 3: Create a zero-filled DataFrame for temperature buckets
    zeros = np.zeros(len(unique_temps))
    rs = np.resize(zeros, (len(unique_temps), len(unique_temps)))
    temp_df = pd.DataFrame(
        data=rs, index=[np.arange(len(unique_temps))], columns=sorted(unique_temps)
    )

    # Step 4: Identify all 'diff' columns in the DataFrame
    diff_columns = [col for col in full_df.columns if "diff" in col]

    # Step 5: Accumulate absolute errors and instance counts into the temperature buckets
    for i, _ in enumerate(full_df[met_col]):
        value = full_df[met_col].iloc[i]
        if 0 <= value < 1.0:
            rounded = round(value, rounded_base)
        else:
            rounded = float(myround(value, base=1))

        abs_err_sum = 0
        valid_count = 0

        # Iterate through all 'diff' columns to calculate absolute errors
        for col in diff_columns:
            err = full_df[col].iloc[i]
            if err > -999:  # Exclude invalid entries
                abs_err_sum += abs(err)
                valid_count += 1

        # Only update if at least one valid error exists
        if valid_count > 0:
            temp_df[rounded].iloc[0] += abs_err_sum / valid_count  # Add MAE
            temp_df[rounded].iloc[
                -1
            ] += valid_count  # Increment by the number of valid instances

    # Step 6: Extract instance counts and mean absolute errors
    instances = temp_df.iloc[-1]
    temp_df = temp_df.iloc[0]

    # Step 7: Remove zero entries from both temp_df and instances
    temp_df = temp_df.loc[~(temp_df == 0)]
    instances = instances.loc[~(instances == 0)]

    return temp_df, instances


def err_bucket(full_df, met_col, rounded_base):
    """
    Groups errors into temperature buckets by rounding the met_col values and calculating the mean absolute error (MAE)
    across all 'diff' columns.

    Parameters:
        full_df (pd.DataFrame): DataFrame containing the data.
        met_col (str): The column used for temperature rounding (e.g., temperature column).
        rounded_base (int or float): The base to round temperatures to for bucketing.

    Returns:
        temp_df (pd.Series): DataFrame containing the mean absolute error for each temperature bucket.
        instances (pd.Series): Series containing the counts of instances in each temperature bucket.
    """
    # Step 1: Round the `met_col` values to the nearest `rounded_base`
    temps = []
    for i, _ in enumerate(full_df[met_col]):
        rounded = myround(full_df[met_col].iloc[i], rounded_base)
        temps.append(rounded)

    # Step 2: Identify unique temperature buckets
    unique_temps = unique(temps)

    # Step 3: Create a zero-filled DataFrame for temperature buckets
    zeros = np.zeros(len(unique_temps))
    rs = np.resize(zeros, (len(unique_temps), len(unique_temps)))
    temp_df = pd.DataFrame(
        data=rs, index=[np.arange(len(unique_temps))], columns=sorted(unique_temps)
    )

    # Step 4: Find all 'diff' columns in the DataFrame
    diff_columns = [col for col in full_df.columns if "diff" in col]

    # Step 5: Accumulate absolute errors and instance counts into the temperature buckets
    for i, _ in enumerate(full_df[met_col]):
        rounded = myround(full_df[met_col].iloc[i], rounded_base)
        abs_err_sum = 0
        valid_count = 0

        # Calculate absolute errors across all 'diff' columns
        for col in diff_columns:
            err = full_df[col].iloc[i]
            if err > -100:
                abs_err_sum += abs(err)
                valid_count += 1

        # Only update if at least one valid error exists
        if valid_count > 0:
            temp_df[rounded].iloc[0] += abs_err_sum / valid_count  # Add MAE
            temp_df[rounded].iloc[-1] += valid_count  # Increment the instance count

    instances = temp_df.iloc[-1]
    temp_df = temp_df.iloc[0]

    return temp_df, instances


def plot_buckets(temp_df, instances, var_name, cmap, width, title):
    my_cmap = plt.get_cmap(cmap)
    averages = temp_df / instances
    averages = averages.dropna()
    y = averages
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    the_list = averages.tolist()
    fig, ax = plt.subplots(
        figsize=(30, 10), facecolor="slategrey", constrained_layout=True
    )
    bars = plt.bar(temp_df.keys(), the_list, color=my_cmap(rescale(y)), width=width)
    ax.set_title("Absolute Error of LSTM", fontsize=28, c="white")
    ax.set_xlabel(var_name, fontsize=18, c="white")
    ax.set_ylabel("Mean Absolute Error", fontsize=18, c="white")
    # Iterating over the bars one-by-one
    # Annotate each bar with its value
    # Annotate each bar with the number of instances
    for bar, value, instance_count in zip(bars, the_list, instances):
        yval = value + 0.01  # Adjust the vertical position of the label
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"n={instance_count}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=12,
            rotation=90,
        )
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/met_error_{title}.png"
    )


def groupby_month_total(df):
    # Filter columns that contain 'diff' in their name
    diff_columns = [col for col in df.columns if "diff" in col]

    # Create an empty list to store the aggregated results
    aggregated_results = []

    # Group by month for each diff column and aggregate (e.g., mean or sum)
    for month in range(1, 13):  # 1 to 12 for each month
        monthly_data = []

        # Collect data for all 'diff' columns for this month
        for col in diff_columns:
            df_filtered = df[
                df[col] > -999
            ]  # Filter rows where column values are greater than -999
            # Group by month and get the mean for this month
            monthly_mean = df_filtered[df_filtered.valid_time.dt.month == month][
                col
            ].mean()
            if not np.isnan(monthly_mean):
                monthly_data.append(monthly_mean)

        # If there is any data for the month, aggregate it (e.g., take the mean of the values for that month)
        if monthly_data:
            aggregated_results.append(
                np.mean(monthly_data)
            )  # You can also sum or apply other aggregations here

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))
    x = np.arange(1, len(aggregated_results) + 1)

    # Define a colormap
    cmap = plt.get_cmap("Blues")

    # Normalize data to apply the colormap
    norm = plt.Normalize(min(aggregated_results) - 0.5, max(aggregated_results))
    colors = cmap(norm(aggregated_results))

    # Create a bar chart with color mapping
    ax.bar(x, aggregated_results, color=colors)

    # Set x-ticks and labels for months
    ax.set_xticks(x)
    month_labels = [calendar.month_name[month] for month in range(1, 13)]
    ax.set_xticklabels(month_labels)
    ax.set_xticklabels(month_labels, rotation=45)

    # Add labels and title
    ax.set_xlabel("Month", fontsize=20)
    ax.set_ylabel("Mean LSTM Error", fontsize=20)
    ax.set_title("Monthly Mean Error for LSTM Predictions", fontsize=32)
    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/month_error.png"
    )


def groupby_month_std(df):
    # Filter columns that contain 'diff' in their name
    diff_columns = [col for col in df.columns if "target" in col]

    # Create an empty list to store the aggregated results
    aggregated_results = []

    # Group by month for each diff column and aggregate (e.g., standard deviation)
    for month in range(1, 13):  # 1 to 12 for each month
        monthly_data = []

        # Collect data for all 'diff' columns for this month
        for col in diff_columns:
            df_filtered = df[
                df[col] > -999
            ]  # Filter rows where column values are greater than -999
            # Group by month and get the standard deviation for this month
            monthly_std = df_filtered[df_filtered.valid_time.dt.month == month][
                col
            ].std()
            if not np.isnan(monthly_std):
                monthly_data.append(monthly_std)

        # If there is any data for the month, aggregate it (e.g., take the mean of the standard deviations for that month)
        if monthly_data:
            aggregated_results.append(
                np.mean(monthly_data)
            )  # You can also sum or apply other aggregations here

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))
    x = np.arange(1, len(aggregated_results) + 1)

    # Define a colormap
    cmap = plt.get_cmap("Reds")

    # Normalize data to apply the colormap
    norm = plt.Normalize(min(aggregated_results) - 0.5, max(aggregated_results))
    colors = cmap(norm(aggregated_results))

    # Create a bar chart with color mapping
    ax.bar(x, aggregated_results, color=colors)

    # Set x-ticks and labels for months
    ax.set_xticks(x)
    month_labels = [calendar.month_name[month] for month in range(1, 13)]
    ax.set_xticklabels(month_labels, rotation=45)

    # Add labels and title
    ax.set_xlabel("Month", fontsize=20)
    ax.set_ylabel("Standard Deviation of NWP Error", fontsize=20)
    ax.set_title("Monthly Standard Deviation of NWP Error", fontsize=32)
    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/month_std_error.png"
    )


def groupby_abs_month_total(df):
    # Filter columns that contain 'diff' in their name
    diff_columns = [col for col in df.columns if "diff" in col]

    # Create an empty list to store the aggregated results
    aggregated_results = []

    # Group by month for each diff column and aggregate (e.g., mean of absolute values)
    for month in range(1, 13):  # 1 to 12 for each month
        monthly_data = []

        # Collect data for all 'diff' columns for this month
        for col in diff_columns:
            df_filtered = df[
                df[col] > -999
            ]  # Filter rows where column values are greater than -999
            # Group by month and calculate the mean of absolute values for this month
            monthly_abs_mean = (
                df_filtered[df_filtered.valid_time.dt.month == month][col]
                .abs()  # Take the absolute value
                .mean()  # Calculate the mean of absolute values
            )
            if not np.isnan(monthly_abs_mean):
                monthly_data.append(monthly_abs_mean)

        # If there is any data for the month, aggregate it (e.g., take the mean of the absolute values for that month)
        if monthly_data:
            aggregated_results.append(
                np.mean(monthly_data)
            )  # You can also sum or apply other aggregations here

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))
    x = np.arange(1, len(aggregated_results) + 1)

    # Define a colormap
    cmap = plt.get_cmap("Blues")

    # Normalize data to apply the colormap
    norm = plt.Normalize(min(aggregated_results) - 0.5, max(aggregated_results))
    colors = cmap(norm(aggregated_results))

    # Create a bar chart with color mapping
    ax.bar(x, aggregated_results, color=colors)

    # Set x-ticks and labels for months
    ax.set_xticks(x)
    month_labels = [calendar.month_name[month] for month in range(1, 13)]
    ax.set_xticklabels(month_labels, rotation=45)

    # Add labels and title
    ax.set_xlabel("Month", fontsize=20)
    ax.set_ylabel("Mean Absolute LSTM Error", fontsize=20)
    ax.set_title("Monthly Mean Absolute Error for LSTM Predictions", fontsize=32)
    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/month_abs_error.png"
    )

    return aggregated_results


def boxplot_monthly_error(df):
    # Filter columns that contain 'diff' in their name
    diff_columns = [col for col in df.columns if "target" in col]

    # Create a dictionary to store monthly data
    monthly_data = {month: [] for month in range(1, 13)}

    # Group by month for each diff column and collect all error values
    for col in diff_columns:
        df_filtered = df[
            df[col] > -999
        ]  # Filter rows where column values are greater than -999
        for month in range(1, 13):
            # Collect error values for the current month
            month_values = df_filtered[df_filtered.valid_time.dt.month == month][
                col
            ].values
            monthly_data[month].extend(month_values)

    # Convert monthly data to a list for plotting
    data_for_plot = [monthly_data[month] for month in range(1, 13)]

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))

    # Create a boxplot
    ax.boxplot(data_for_plot, patch_artist=True, showfliers=True)

    # Set x-ticks and labels for months
    month_labels = [calendar.month_name[month] for month in range(1, 13)]
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)

    # Add labels and title
    ax.set_xlabel("Month", fontsize=20)
    ax.set_ylabel("NWP Error", fontsize=20)
    ax.set_title("Monthly Box-and-Whisker Plot of Errors for NWP Error", fontsize=32)
    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xticklabels(month_labels, rotation=45)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/month_error_boxplot.png"
    )


def groupby_time_abs(df):
    # Filter columns that contain 'diff' in their name
    diff_columns = [col for col in df.columns if "diff" in col]

    # Create an empty list to store the aggregated results
    aggregated_results = []
    valid_hours = []  # To keep track of hours that have data

    # Group by hour (time of day) for each diff column and calculate the mean of absolute values
    for hour in range(0, 24):  # 0 to 23 for each hour of the day
        hourly_data = []

        # Collect data for all 'diff' columns for this hour
        for col in diff_columns:
            df_filtered = df[
                df[col] > -999
            ]  # Filter rows where column values are greater than -999
            # Group by hour and calculate the mean of absolute values for this hour
            hourly_abs_mean = (
                df_filtered[df_filtered.valid_time.dt.hour == hour][col]
                .abs()  # Take the absolute value
                .mean()  # Calculate the mean of absolute values
            )
            if not np.isnan(hourly_abs_mean):
                hourly_data.append(hourly_abs_mean)

        # If there is any data for the hour, aggregate it (e.g., take the mean of the absolute values for that hour)
        if hourly_data:
            aggregated_results.append(
                np.mean(hourly_data)
            )  # Aggregate the data (mean of hourly values)
            valid_hours.append(hour)  # Track the hour that has valid data

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))
    x = np.arange(0, len(aggregated_results))

    # Define a colormap
    cmap = plt.get_cmap("Greens")

    # Normalize data to apply the colormap
    norm = plt.Normalize(min(aggregated_results) - 0.2, max(aggregated_results))
    colors = cmap(norm(aggregated_results))

    # Create a bar chart with color mapping
    ax.bar(x, aggregated_results, color=colors)

    # Set x-ticks and labels for valid hours
    ax.set_xticks(x)
    hour_labels = [f"{hour:02d}:00" for hour in valid_hours]
    ax.set_xticklabels(hour_labels, rotation=45)

    # Add labels and title
    ax.set_xlabel("Time of Day (Hour)", fontsize=20)
    ax.set_ylabel("Mean Absolute LSTM Error", fontsize=20)
    ax.set_title("Mean Absolute Error Grouped by Time of Day", fontsize=32)

    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/time_of_day_abs_error_colored.png"
    )

    return aggregated_results


def groupby_time(df):
    # Filter columns that contain 'diff' in their name
    diff_columns = [col for col in df.columns if "diff" in col]

    # Create an empty list to store the aggregated results
    aggregated_results = []
    valid_hours = []  # To keep track of hours that have data

    # Group by hour (time of day) for each diff column and calculate the mean
    for hour in range(0, 24):  # 0 to 23 for each hour of the day
        hourly_data = []

        # Collect data for all 'diff' columns for this hour
        for col in diff_columns:
            df_filtered = df[
                df[col] > -999
            ]  # Filter rows where column values are greater than -999
            # Group by hour and calculate the mean for this hour
            hourly_mean = df_filtered[df_filtered.valid_time.dt.hour == hour][
                col
            ].mean()
            if not np.isnan(hourly_mean):
                hourly_data.append(hourly_mean)

        # If there is any data for the hour, aggregate it (e.g., take the mean of the values for that hour)
        if hourly_data:
            aggregated_results.append(
                np.mean(hourly_data)
            )  # Aggregate the data (mean of hourly values)
            valid_hours.append(hour)  # Track the hour that has valid data

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))
    x = np.arange(0, len(aggregated_results))

    # Define a colormap
    cmap = plt.get_cmap("Greens")

    # Normalize data to apply the colormap
    norm = plt.Normalize(min(aggregated_results) - 0.2, max(aggregated_results))
    colors = cmap(norm(aggregated_results))

    # Create a bar chart with color mapping
    ax.bar(x, aggregated_results, color=colors)

    # Set x-ticks and labels for valid hours
    ax.set_xticks(x)
    hour_labels = [
        f"{hour:02d}:00" for hour in valid_hours
    ]  # Labels for valid hours only
    ax.set_xticklabels(hour_labels, rotation=45)

    # Add labels and title
    ax.set_xlabel("Time of Day (Hour)", fontsize=20)
    ax.set_ylabel("Mean LSTM Error", fontsize=20)
    ax.set_title("Mean Error Grouped by Time of Day", fontsize=32)

    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/time_of_day_mean_error_colored.png"
    )

    return aggregated_results


def boxplot_time_of_day_error(df):
    # Filter columns that contain 'diff' in their name
    diff_columns = [col for col in df.columns if "target" in col]

    # Create a dictionary to store hourly data
    hourly_data = {hour: [] for hour in range(0, 24)}

    # Group by hour (time of day) for each diff column and collect all error values
    for col in diff_columns:
        df_filtered = df[
            df[col] > -999
        ]  # Filter rows where column values are greater than -999
        for hour in range(0, 24):
            # Collect error values for the current hour
            hour_values = df_filtered[df_filtered.valid_time.dt.hour == hour][
                col
            ].values
            hourly_data[hour].extend(hour_values)

    # Convert hourly data to a list for plotting
    data_for_plot = [hourly_data[hour] for hour in range(0, 24)]

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))

    # Create a boxplot
    ax.boxplot(data_for_plot, patch_artist=True, showfliers=True)

    # Set x-ticks and labels for hours
    hour_labels = [f"{hour:02d}:00" for hour in range(0, 24)]
    ax.set_xticks(range(1, 25))  # Boxplot indices start at 1
    ax.set_xticklabels(hour_labels, rotation=45)

    # Add labels and title
    ax.set_xlabel("Time of Day (Hour)", fontsize=20)
    ax.set_ylabel("NWP Error", fontsize=20)
    ax.set_title(
        "Box-and-Whisker Plot of NWP Error Grouped by Time of Day", fontsize=32
    )
    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/time_of_day_error_boxplot.png"
    )


def groupby_time_std(df):
    # Filter columns that contain 'target' in their name
    diff_columns = [col for col in df.columns if "target" in col]

    # Create an empty list to store the aggregated results
    aggregated_results = []
    valid_hours = []  # To keep track of hours that have data

    # Group by hour (time of day) for each diff column and calculate the standard deviation
    for hour in range(0, 24):  # 0 to 23 for each hour of the day
        hourly_data = []

        # Collect data for all 'diff' columns for this hour
        for col in diff_columns:
            df_filtered = df[
                df[col] > -999
            ]  # Filter rows where column values are greater than -999
            # Group by hour and calculate the standard deviation for this hour
            hourly_std = df_filtered[df_filtered.valid_time.dt.hour == hour][col].std()
            if not np.isnan(hourly_std):
                hourly_data.append(hourly_std)

        # If there is any data for the hour, aggregate it (e.g., take the mean of the standard deviations for that hour)
        if hourly_data:
            aggregated_results.append(
                np.mean(hourly_data)
            )  # Aggregate the data (mean of hourly standard deviations)
            valid_hours.append(hour)  # Track the hour that has valid data

    # Plotting
    fig, ax = plt.subplots(figsize=(30, 17))
    x = np.arange(0, len(aggregated_results))

    # Define a colormap
    cmap = plt.get_cmap("Reds")

    # Normalize data to apply the colormap
    norm = plt.Normalize(min(aggregated_results) - 0.1, max(aggregated_results))
    colors = cmap(norm(aggregated_results))

    # Create a bar chart with color mapping
    ax.bar(x, aggregated_results, color=colors)

    # Set x-ticks and labels for valid hours
    ax.set_xticks(x)
    hour_labels = [
        f"{hour:02d}:00" for hour in valid_hours
    ]  # Labels for valid hours only
    ax.set_xticklabels(hour_labels, rotation=45)

    # Add labels and title
    ax.set_xlabel("Time of Day (Hour)", fontsize=20)
    ax.set_ylabel("Standard Deviation of NWP Error", fontsize=20)
    ax.set_title("Standard Deviation of Error Grouped by Time of Day", fontsize=32)

    # Customize tick mark font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/time_of_day_std_error_colored.png"
    )

    return aggregated_results


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]
    return ldf


def create_scatterplot(x_column, y_column, fh):
    # Calculate point density
    xy = np.vstack([x_column, y_column])
    z = gaussian_kde(xy)(xy)

    plt.figure(figsize=(16, 12))

    # Create the scatterplot
    scatter = plt.scatter(
        x_column,
        y_column,
        c=z,
        cmap="viridis",
        s=100,
        # edgecolor="black",
        alpha=0.5,
    )

    # Add color bar with label
    cbar = plt.colorbar(scatter)
    cbar.set_label("Point Density")

    # Set labels and title
    plt.xlabel("Target", fontsize=24)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.ylabel("LSTM", fontsize=24)
    plt.title("Target Temp Error v LSTM Predictions", fontsize=32)
    # Customize tick mark font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Show the plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/scatter_{fh}.png"
    )


def ml_output(df, full_df, fold, station, test_set_start, test_set_finish, fh):
    fig, ax = plt.subplots(figsize=(24, 6))
    x = df["valid_time"]

    # Convert datetime values to numerical values
    x_numeric = mdates.date2num(x)

    # Assuming your timestamps are in a datetime64 format
    day_mask = (x.dt.hour >= 6) & (
        x.dt.hour < 18
    )  # Adjust the hours based on your day/night definition

    plt.plot(
        np.array(x),
        np.array(df["target_error_lead_0"]),
        c="black",
        linewidth=1,
        label="Target",
    )

    plt.plot(
        np.array(x),
        np.array(df["Model forecast"]),
        c="red",
        linewidth=3,
        alpha=0.7,
        label="LSTM Output",
    )

    # plt.axvline(
    #     x=test_set_start,
    #     c="green",
    #     linestyle="--",
    #     linewidth=2.0,
    #     label="Test Set Start",
    # )
    # plt.axvline(
    #     x=test_set_finish,
    #     c="red",
    #     linestyle="--",
    #     linewidth=2.0,
    #     label="Test Set Finish",
    # )

    # Fill daytime hours with white color
    ax.fill_between(
        x_numeric, -4, 4.1, where=day_mask, color="white", alpha=0.5, label="Daytime"
    )

    # Fill nighttime hours with grey color
    ax.fill_between(
        x_numeric, -4, 4.1, where=~day_mask, color="grey", alpha=0.2, label="Nighttime"
    )

    ax.set_title(f"Temp Error LSTM Output v Target: {station}: FH{fh}", fontsize=28)
    # plt.ylim(-5, 5.)
    ax.legend()
    plt.show()


def plot_fh_drift(mae_ls, sq_ls, r2_ls, fh):
    """
    Plots three lists (mae_ls, sq_ls, and r2_ls) as a function of forecast hour (fh).
    Each series will have a scatter-line plot with unique markers and colors,
    and the points will be annotated.

    Parameters:
        mae_ls (list): Mean absolute error values.
        sq_ls (list): Squared error values.
        r2_ls (list): R² values.
        fh (list): Forecast hours.
    """
    plt.figure(figsize=(15, 9))

    # Plot mae_ls
    plt.plot(fh, mae_ls, label="MAE", marker="o", linestyle="-", color="blue")
    plt.scatter(fh, mae_ls, marker="o", color="blue")
    # Annotate mae_ls points
    for i, txt in enumerate(mae_ls):
        plt.annotate(
            f"{txt:.2f}",
            (fh[i], mae_ls[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="blue",
        )

    # Plot sq_ls
    plt.plot(fh, sq_ls, label="MSE", marker="x", linestyle="-", color="green")
    plt.scatter(fh, sq_ls, marker="x", color="green")
    # Annotate sq_ls points
    for i, txt in enumerate(sq_ls):
        plt.annotate(
            f"{txt:.2f}",
            (fh[i], sq_ls[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="green",
        )

    # Plot r2_ls
    plt.plot(fh, r2_ls, label="R²", marker="s", linestyle="-", color="red")
    plt.scatter(fh, r2_ls, marker="s", color="red")
    # Annotate r2_ls points
    for i, txt in enumerate(r2_ls):
        plt.annotate(
            f"{txt:.2f}",
            (fh[i], r2_ls[i]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=10,
            color="red",
        )

    # Add labels, legend, and title
    plt.xlabel("Forecast Hour (FH)", fontsize=20)
    plt.ylabel("Error and R² Values", fontsize=20)
    plt.title(
        "Error Metrics as a Function of Forecast Hour \n GFS, Temp-Error, Voorheesville",
        fontsize=24,
    )
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Ensure x-ticks are integers
    plt.xticks(ticks=range(int(min(fh)), int(max(fh)) + 1), fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    # Show plot
    plt.show()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/visuals/fh_drift.png"
    )


def calculate_r2(df):
    lstms = []
    targets = []
    r2_ls = []

    for d in df[f"Model forecast"].values:
        if abs(d) < 100:
            lstms.append(d)

    for x in df[f"target_error_lead_0"].values:
        if abs(x) < 100:
            targets.append(x)

    # calculate r2
    r2 = r2_score(targets, lstms)
    r2_ls.append(r2)

    for i in np.arange(6, 37, 3):
        lstms_ = []
        targets_ = []

        for d in df[f"Model forecast_{i}"].values:
            if abs(d) < 100:
                lstms_.append(d)

        for x in df[f"target_error_lead_0_{i}"].values:
            if abs(x) < 100:
                targets_.append(x)

        r2_ = r2_score(targets_, lstms_)
        r2_ls.append(r2_)

    return r2_ls
