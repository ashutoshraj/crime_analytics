from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def filter_data(df):
    df["Per_Capita_Income"] = df["Per_Capita_Income"].fillna(0.0).astype(float)
    df["State_Average_Income"] = df["State_Average_Income"].fillna(0.0).astype(str)
    df["State_Average_Income"] = df["State_Average_Income"].str.replace(',', '')
    df["State_Average_Income"] = df["State_Average_Income"].fillna(0.0).astype(float)
    df['State'] = df['State'].astype('category')
    df['Crime_Density_Per_1000'] = df['Total_Crime']/df['Population']*1000
    df['Per_Capita_GDP'] = df['GDP']/df['Population']
    df['Diff_Income_County_vs_State'] = df['Per_Capita_Income'] - df['State_Average_Income']
    df['Perc_Diff_Income_County_vs_State'] = df['Diff_Income_County_vs_State']/df['Per_Capita_Income']*100
    return df


def fetch_data():
    engine = create_engine('sqlite:///../data/Crime_With_Social_Indicators.db', echo=False)
    sql_connection = engine.connect()

    dataframe = pd.read_sql('SELECT * FROM Crime', engine)
    sql_connection.close()
    dataframe = filter_data(df=dataframe)
    # print(dataframe.columns)
    selected_cols = ['Crime_Density_Per_1000', 'Per_Capita_Income', 'Perc_Diff_Income_County_vs_State',
                     'Per_Capita_GDP', 'UNP_Rate', 'higher_degree', 'Population_Density',
                      'State', 'Year', 'Metro', 'County', 'GeoFIPS']
    df_selected = dataframe[selected_cols]
    df_selected = df_selected.dropna()
    df_selected = df_selected[~df_selected.isin([np.nan, np.inf, -np.inf]).any(1)]
    return df_selected