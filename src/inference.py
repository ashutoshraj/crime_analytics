import streamlit as st
import pickle
import pandas as pd


def generate_dummies(x, dummies, var):
        vector = []
        dums = dict(zip(dummies.columns, [0]*dummies.shape[1]))
        for k, v in dums.items():
            if k == x[var]:
                vector.append(1)
            else:
                vector.append(0)
        temp = pd.Series(dict(zip(dummies.columns, vector)))
        return temp


def app():
    st.markdown("## Crime Predictor Page!")
    st.write("\n")
    model, sc, metro_dummies, state_dummies = None, None, None, None

    with open('models/model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    with open('models/standardScaler.pkl', 'rb') as handle:
        sc = pickle.load(handle)
    
    State_dummies = pd.read_csv('models/State_dummies.csv')
    Metro_dummies = pd.read_csv('models/Metro_dummies.csv')
    
    num_cols = ['Per_Capita_Income', 'Perc_Diff_Income_County_vs_State', 'Per_Capita_GDP', 'UNP_Rate', 'higher_degree', 'Population_Density']

    col1, col2, col3, col4 = st.columns(4)

    per_capita_income = col1.number_input("Per_Capita_Income", value=33348.0)
    diff_income = col2.number_input("Perc_Diff_Income_County_vs_State", value=-27.713806)

    per_capita_gdp = col3.number_input("Per_Capita_GDP", value=22.315484)

    unp_rate = col4.number_input("UNP_Rate", value=8.8)

    higher_degree = col1.number_input("higher_degree", value=26.571573)

    population_density = col2.number_input("Population_Density", value=0.000036)

    metro_values = ['Metropolitan', 'Nonmetropolitan']
    metro = col3.selectbox("Metro", metro_values, index=metro_values.index("Nonmetropolitan"))

    state_values = ['ALABAMA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO',
       'DELAWARE', 'FLORIDA', 'GEORGIA', 'IDAHO', 'ILLINOIS', 'INDIANA',
       'IOWA', 'KANSAS', 'KENTUCKY', 'MAINE', 'MARYLAND', 'MICHIGAN',
       'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA',
       'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK',
       'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON',
       'PENNSYLVANIA', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE',
       'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON',
       'WEST VIRGINIA', 'WISCONSIN', 'WYOMING', 'MINNESOTA']

    state = col4.selectbox("State", state_values, index=state_values.index("ALABAMA"))

    test_dict = {'Per_Capita_Income': per_capita_income,
                 'Perc_Diff_Income_County_vs_State': diff_income,
                 'Per_Capita_GDP': per_capita_gdp, 'UNP_Rate': unp_rate,
                 'higher_degree': higher_degree, 'Population_Density': population_density,
                 'Metro': metro, "State": state}

    x_test = pd.Series(test_dict)

    predict = st.button("Predict")
    if predict:
        x_test_scaled = sc.transform(x_test[num_cols].values.reshape(1,-1))
        x_test.loc[num_cols] = x_test_scaled[0]

        metro_dum = generate_dummies(x_test, Metro_dummies, "Metro")
        state_dum = generate_dummies(x_test, State_dummies, "State")
        del x_test["State"]
        del x_test["Metro"]
        x_test = pd.concat([x_test, metro_dum, state_dum])

        pred = round(model.predict(x_test.values.reshape(1,-1))[0], 4)

        st.metric("Crime_Density_Per_1000", str(pred), str(12.3783))