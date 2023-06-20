# -*- coding: utf-8 -*-
def print_stats(index_df):
    """
    Prints the average correlational coefficient and average p-score across models for which the absolute value of the pers coefficient is less than or equal to 1.

    Args:
    - index_df: a pandas dataframe containing the correlation index and p-score for each model

    Returns:
    - None
    """

    # Select only the rows with pers coeff values between -1 and 1 (inclusive)
    mean_df = index_df[abs(index_df["pers"]) <= 1]

    # Compute the average correlational coefficient and average p-score for the selected rows
    mean1 = abs(mean_df["pers"]).mean()
    mean2 = mean_df["p_score"].mean()

    # Format the string with the results
    fstring = f"The Average Correlational Coefficient Across Models is: {mean1}, and the average p-score across models is {mean2}"

    # Print the string
    print(fstring)
