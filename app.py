import streamlit as st
import pickle
import pandas as pd
import xgboost

model = pickle.load(open('Insurance_model_xgb','rb'))


st.set_page_config(
    page_title="Medical cost prediction  App",
    layout="centered",
    initial_sidebar_state="expanded")

Sex_map = {"Male":1 , "Female":0}
Smoker = {"Yes": 1, "No":0}
Region = {"Northeast":0 , "Northwest":1, "Southeast":2, "Southwest":3}
Ethnicity = {"Asian":0, "Black":1, "Latino":2, "White":3}

def preprocess(age, sex, bmi, ethnicity, children, smoker, region):
    
    X_cat = pd.DataFrame({"children":children, "sex":sex,  "smoker":smoker, "region":region, "ethnicity":ethnicity}, index=[0])

    X_cat.sex = X_cat.sex.map(Sex_map)
    X_cat.smoker = X_cat.smoker.map(Smoker)
    X_cat.region = X_cat.region.map(Region)
    X_cat.ethnicity=X_cat.region.map(Ethnicity)

    X_num = pd.DataFrame({"age": age, "bmi":bmi}, index=[0])
    
    X = pd.concat([X_num, X_cat], axis=1)
    
    return X


def predict_age(data):
    prediction = model.predict(data)
    return int(prediction)


def main():

    #st.title("Medical Cost Estimation")
    
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Medical Cost Estimation </h2>
    </div>
    """

    st.subheader("Jose Villamor")

    st.markdown(html_temp, unsafe_allow_html = True)


    st.write('This app provides an estimation of total medical cost using a xgboost model. Select the corresponding variables and click predict to obtain a result. Be aware that this is a project for portofolio purposes so the outcome is not truly real just an approximation.')
    st.write("If you want to know more about the project or others that i have done visit my github account: https://github.com/Jose-Villamor?tab=repositories")

    age = st.number_input("Age (From 18 to 64)", value=18.0, step=1.0, min_value=18.0, max_value=64.0)
    sex = st.selectbox("Sex", ("Male",  "Female"))
    bmi = st.slider("BMI (From 15.9 to 53.1)", value=15.9, step=0.1, min_value=15.9, max_value=53.1)
    ethnicity = st.selectbox("Ethnicity",("Black","White", "Asian", "Latino"))
    children = st.number_input("Children (From 0 to 5)",value=0.0, step=1.0, min_value=0.0, max_value=5.0)
    smoker = st.selectbox("Smoker",("Yes","No"))
    region = st.selectbox("Region", ("Northeast", "Northwest", "Southeast", "Southwest"))


    if st.button("Predict cost"):
        
       output = predict_age(preprocess(age, sex, bmi, ethnicity, children, smoker, region))
        
       st.success(f'The cost are {output:,d} dollars')


if __name__=='__main__':
    main()
