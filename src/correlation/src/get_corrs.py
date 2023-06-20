# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats


def rename_columns(df):
    new_df = df.rename(
        columns={
            "1": "January",
            "2": "February",
            "3": "March",
            "4": "April",
            "5": "May",
            "6": "June",
            "7": "July",
            "8": "August",
            "9": "September",
            "10": "October",
            "11": "November",
            "12": "December",
        }
    )
    return new_df


def get_corrs(df: pd.DataFrame, lulc: pd.DataFrame, keys: list, error_var) -> tuple:
    """
    Calculates the Pearson, Spearman, and Kendall's Tau correlation coefficients
    and their corresponding p-values between a dataframe of temperature errors
    (df) and a grographic data dataframe (lulc) for each month of the year.

    Args:
        df (pandas.DataFrame): A pandas dataframe containing the temperature errors.
        lulc (pandas.DataFrame): A pandas dataframe containing the geographic data.
        keys (list): A list of the column names to be used in the output dataframes.

    Returns:
        A tuple containing the following dataframes:

        - df_pers: A pandas dataframe containing the Pearson correlation coefficients for each month.
        - df_rho: A pandas dataframe containing the Spearman correlation coefficients for each month.
        - df_tau: A pandas dataframe containing the Kendall's Tau correlation coefficients for each month.
        - df_p_score: A pandas dataframe containing the p-values corresponding to the Pearson correlation coefficients for each month.
        - df_p_score_rho: A pandas dataframe containing the p-values corresponding to the Spearman correlation coefficients for each month.
        - df_p_score_tau: A pandas dataframe containing the p-values corresponding to the Kendall's Tau correlation coefficients for each month.
    """
    # Initialize empty dataframes for output
    df_pers = pd.DataFrame()
    df_rho = pd.DataFrame()
    df_tau = pd.DataFrame()
    df_p_score = pd.DataFrame()
    df_p_score_rho = pd.DataFrame()
    df_p_score_tau = pd.DataFrame()

    # Iterate over each month
    for i in np.arange(1, 13):
        # Initialize empty lists for each correlation coefficient and p-value
        pers_ls = []
        rho_ls = []
        tau_ls = []
        p_score_ls = []
        p_score_rho_ls = []
        p_score_tau_ls = []

        # Subset the dataframe to the current month
        months_df = df[df["time"] == i]

        # Iterate over each column in the land use/land cover dataframe
        for col, val in lulc.iteritems():
            # Calculate Pearson correlation coefficient and p-value
            p_score_pers = scipy.stats.pearsonr(lulc[col], months_df[error_var])[1]
            if p_score_pers > 0.05:
                pers = -999.99
            else:
                pers = scipy.stats.pearsonr(lulc[col], months_df[error_var])[0]

            # Calculate Spearman correlation coefficient and p-value
            p_score_rho = scipy.stats.spearmanr(lulc[col], months_df[error_var])[1]
            if p_score_rho > 0.05:
                rho = -999.99
            else:
                rho = scipy.stats.spearmanr(lulc[col], months_df[error_var])[0]

            # Calculate Kendall's Tau correlation coefficient and p-value
            p_score_tau = scipy.stats.kendalltau(lulc[col], months_df[error_var])[1]
            if p_score_tau > 0.05:
                tau = -999.99
            else:
                tau = scipy.stats.kendalltau(lulc[col], months_df[error_var])[0]

            # Append correlation coefficients and p-values to their respective lists
            pers_ls.append(pers)
            rho_ls.append(rho)
            tau_ls.append(tau)
            p_score_ls.append(p_score_pers)
            p_score_rho_ls.append(p_score_rho)
            p_score_tau_ls.append(p_score_tau)

        # coefficients
        df_pers1 = pd.DataFrame(index=keys)
        df_pers1[f"{i}"] = pers_ls
        df_rho1 = pd.DataFrame(index=keys)
        df_rho1[f"{i}"] = rho_ls
        df_tau1 = pd.DataFrame(index=keys)
        df_tau1[f"{i}"] = tau_ls
        # p_scores
        df_p_score1 = pd.DataFrame(index=keys)
        df_p_score1[f"{i}"] = p_score_ls
        df_p_score_rho1 = pd.DataFrame(index=keys)
        df_p_score_rho1[f"{i}"] = p_score_rho_ls
        df_p_score_tau1 = pd.DataFrame(index=keys)
        df_p_score_tau1[f"{i}"] = p_score_tau_ls

        # concat dataframes
        df_pers = pd.concat([df_pers, df_pers1], axis=1)
        df_rho = pd.concat([df_rho, df_rho1], axis=1)
        df_tau = pd.concat([df_tau, df_tau1], axis=1)
        df_p_score = pd.concat([df_p_score, df_p_score1], axis=1)
        df_p_score_rho = pd.concat([df_p_score_rho, df_p_score_rho1], axis=1)
        df_p_score_tau = pd.concat([df_p_score_tau, df_p_score_tau1], axis=1)

    # rename columns
    df_pers = rename_columns(df_pers)
    df_rho = rename_columns(df_rho)
    df_tau = rename_columns(df_tau)
    df_p_score = rename_columns(df_p_score)
    df_p_score_rho = rename_columns(df_p_score_rho)
    df_p_score_tau = rename_columns(df_p_score_tau)

    # return dataframes
    return df_pers, df_rho, df_tau, df_p_score, df_p_score_rho, df_p_score_tau
