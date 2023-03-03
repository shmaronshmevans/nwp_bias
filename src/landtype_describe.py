import pandas as pd 


def landtype_describe(df:pd.DataFrame):
    """
    This prints out descriptive characteristics of the NLCD data from the dataframe

    Args: 
    dataframe (pd.DataFrame) : NLCD lantype

    Returns: 
    Dataframe of landtypes 
    """

    df['Value'].plot.hist(bins=80,rwidth=0.9,
                        color='red')
    print("The mode value is:")
    print(df['Value'].mode())

    print("The value counts for the NLCD landtypes are:")
    print(df['Value'].value_counts())

    print("The staistical analysis of the NLCD landtypes are:")
    print(df[['Value']].describe())