from matplotlib import figure
import streamlit as st
from src.data import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from scipy import stats
import pickle
import matplotlib.pyplot as plt
from urllib.request import urlopen


def main():
    st.markdown("## Welcome to model creation page!")
    st.write("\n")
    data = fetch_data()
    st.write("Brief overview of data!")
    st.dataframe(data.describe())

    population_threshold = st.text_input("Select the population density threshold", value=1.734387e-05)
    population_threshold = float(population_threshold)

    num_estimators = st.text_input("Number of estimators", value=10)
    num_estimators = int(num_estimators)

    train_test_split_ratio = st.text_input("Enter train test split ratio", value=0.2)
    train_test_split_ratio = float(train_test_split_ratio)

    df_slice_dense = data[data["Population_Density"] >= population_threshold]
    df_slice_sparse = data[data["Population_Density"] < population_threshold]

    # Create Dummies for categorical variables
    X= data[['Per_Capita_Income', 'Perc_Diff_Income_County_vs_State', 'Per_Capita_GDP', 'UNP_Rate', 'higher_degree', 'Population_Density', 'State', 'Metro']]
    y = data['Crime_Density_Per_1000']

    X.to_csv("X.csv", index=False)
    y.to_csv("y.csv", index=False)
    # handle categorical variable
    State_dummies=pd.get_dummies(X['State'],drop_first=True)
    # state_dummies = OneHotEncoder()
    # state_dummies_arr = state_dummies.fit_transform(X['State']).toarray().reshape(1, -1)

    Metro_dummies=pd.get_dummies(X['Metro'],drop_first=True)
    # metro_dummies = OneHotEncoder()
    # metro_dummies_arr = metro_dummies.fit_transform(X['Metro']).toarray().reshape(1, -1)

    # dropping extra column
    X= X.drop('State',axis=1)
    X= X.drop('Metro',axis=1)

    # concatation of independent variables and new cateorical variable.
    X=pd.concat([X,State_dummies],axis=1)
    X=pd.concat([X,Metro_dummies],axis=1)

    # Split the dataset in train and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = train_test_split_ratio,
                                                         random_state = 42)

    num_cols = ['Per_Capita_Income', 'Perc_Diff_Income_County_vs_State', 'Per_Capita_GDP', 'UNP_Rate', 'higher_degree', 'Population_Density']
    cat_cols = ['State', 'Metro']

    # Standardize the data
    std = StandardScaler()
    std.fit(X_train[num_cols])
    X_train[num_cols] = std.transform(X_train[num_cols])
    X_test[num_cols] = std.transform(X_test[num_cols])

    start_training = st.button("Start Training")
    if start_training:
        # Use different algorithms to fit the model
        regressor = RandomForestRegressor(n_estimators = num_estimators, random_state = 0)

        # fitting the training data
        regressor.fit(X_train,y_train)

        # param_grid = {
        #         'n_estimators': [5, 10, 25, 50, 100]            
        #     }
            
        # # Instantiate the grid search model
        # gscv_rfc = GridSearchCV(estimator = regressor, param_grid = param_grid).fit(X_train, y_train)

        best_params = 10 #gscv_rfc.best_params_

        regressor_best_params = RandomForestRegressor(n_estimators = 10, random_state = 0)

        # fitting the training data
        regressor_best_params.fit(X_train,y_train)

        y_train_prediction =  regressor_best_params.predict(X_train)
        y_prediction =  regressor_best_params.predict(X_test)

        # Model Accuracy Metrics
        score=r2_score(y_test,y_prediction)
        st.write("R2 score:",score)
        st.write("Mean Squared Error:",mean_squared_error(y_test,y_prediction))
        st.write("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,y_prediction)))

        # Residual Analysis
        residuals = y_train - y_train_prediction
        
        fig, axes = plt.subplots(2, 2)
        fig.suptitle('Residual Analysis')
        fig.tight_layout(h_pad=5)

        sns.regplot(x=y_train_prediction, y=residuals,
                    ax=axes[0,0], scatter_kws={"color": "blue"},
                    line_kws={"color": "orange"}, marker='+')
        # axes[0,0].set_title ('Residual vs fitted plot')
        axes[0,0].set_xlabel("Fitted values")
        axes[0,0].set_ylabel("Residuals")
    
        axes[0,1].set_title('Histogram')
        sns.distplot(residuals,norm_hist=True,ax=axes[0,1])

        axes[1,0].set_title('Q-Q Plot')
        stats.probplot(residuals,plot=axes[1,0])
        st.balloons()
        st.pyplot(fig)

        with open('models/model.pkl', 'wb') as handle:
            pickle.dump(regressor, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('models/standardScaler.pkl', 'wb') as handle:
            pickle.dump(std, handle, protocol=pickle.HIGHEST_PROTOCOL)

        Metro_dummies.to_csv("models/Metro_dummies.csv", index=False)
        State_dummies.to_csv("models/State_dummies.csv", index=False)

def app():
    def is_authenticated(password):
        return password == "admin"


    def generate_login_block():
        block1 = st.empty()
        block2 = st.empty()

        return block1, block2


    def clean_blocks(blocks):
        for block in blocks:
            block.empty()


    def login(blocks):
        blocks[0].markdown("""
                <style>
                    input {
                        -webkit-text-security: disc;
                    }
                </style>
            """, unsafe_allow_html=True)

        return blocks[1].text_input('Password')


    login_blocks = generate_login_block()
    password = login(login_blocks)

    if is_authenticated(password):
        clean_blocks(login_blocks)
        main()
    elif password:
        st.info("Please enter a valid password")
