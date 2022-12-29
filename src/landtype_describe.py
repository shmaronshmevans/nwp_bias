import pandas as pd 


def landtype_describe(df:pd.DataFrame):
    """
    This prints out descriptive characteristics of the NLCD data from the dataframe

    Args: 
    dataframe (pd.DataFrame) : NLCD lantype

    Returns: 
    Dataframe of landtypes 
    """

    df['cover_2019'].plot.hist(bins=80,rwidth=0.9,
                        color='red')
    print("The mode value is:")
    print(df['cover_2019'].mode())

    print("The value counts for the NLCD landtypes are:")
    print(df['cover_2019'].value_counts())

    print("The staistical analysis of the NLCD landtypes are:")
    print(df[['cover_2019']].describe())