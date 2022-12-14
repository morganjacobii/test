import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x==1,1,0)
    return x

ss = pd.DataFrame({
    'sm_li':clean_sm(s["web1h"]),
    'income':np.where(s["income"] > 9, np.nan, s["income"]),
    'education':np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    'parent':np.where(s["par"]==1,1,0),
    'married':np.where(s["marital"]==1,1,0),
    'female':np.where(s["gender"] >= 3, np.nan, np.where(s["gender"]==2,1,0)),
    'age':np.where(s["age"] > 98, np.nan, s["age"])}).dropna().sort_values(by=["income","education"], ascending = True)

Y = ss["sm_li"]
X = ss[["income", "education", "age", "female", "parent", "married"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    stratify=Y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

#Initialize algorithm
lr = LogisticRegression(class_weight = "balanced")

#Fit algorithm to training data
lr.fit(X_train, y_train)

# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)

#confusion matrix
confusion_matrix(y_test, y_pred)

#confusion_matrix(y_test, y_pred)
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

print(classification_report(y_test, y_pred))         

#### USER INPUT

st.title("Welcome!")
st.subheader("This application will predict whether you are a LinkedIn user based on different predictors. Enjoy!")
st.caption("Please answer the following questions.")

#income
income = st.selectbox("Select your yearly household income level:",
    options = ["Less than $10,000",
                "$10,000 to $20,000",
                "$20,000 to $30,000",
                "$30,000 to $40,000", 
                "$40,000 to $50,000",
                "$50,000 to $75,000",
                "$75,000 to $100,000",
                "$100,000 to $150,000",
                "$150,000 or more"])

if income == "Less than $10,000": income = 1
elif income == "$10,000 to $20,000": income = 2
elif income == "$20,000 to $30,000": income = 3
elif income == "$30,000 to $40,000": income = 4
elif income == "$40,000 to $50,000": income = 5
elif income == "$50,000 to $75,000": income = 6
elif income == "$75,000 to $100,000": income = 7
elif income == "$100,000 to $150,000": income = 8
else: income = 9

#education
education = st.selectbox("What is the highest level of school/degree you have completed?",
    options = ["Less than high school",
                "High school incomplete",
                "High school graduate",
                "Some college, no degree", 
                "Two-year associate degree from a college or university",
                "Four-year college or university degree/Bachelor's degree",
                "Some postgraduate or professional school, no postgraduate degree",
                "Postgraduate or professional degree, including master's, doctorate, medical, or law degree"])

if education == "Less than high school": education = 1
elif education == "High school incomplete": education = 2
elif education == "High school graduate": education = 3
elif education == "Some college, no degree": education = 4
elif education == "Two-year associate degree from a college or university": education = 5
elif education == "Four-year college or university degree/Bachelor's degree": education = 6
elif education == "Some postgraduate or professional school, no postgraduate degree": education = 7
else: education = 8

# age
age = st.slider('Enter age:')

#female
female = st.radio("Select gender:", ["Female", "Male"])
if female == "Female":
    female = 1
else:
    female = 0

#parent
parent = st.radio("Do you have children (under the age of 18)?", ["Yes", "No"])
if parent == "Yes":
    parent = 1
else:
    parent = 0

#married
married = st.radio("Are you married?", ["Yes", "No"])
if married == "Yes":
    married = 1
else:
    married = 0

person = [income, education, age, female, parent, married]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

#print probability that individual uses LinkedIn 
print_probs = round(probs[0][1]*100,2)
st.markdown(f"Based on your answers, there is a {print_probs}% probability that you are a LinkedIn user.")

#print whether the person would be classified as a LinkedIn user
if print_probs >= 50:
    st.markdown("Therefore, you are a LinkedIn user.")
else:
    st.markdown("Therefore, you are not a LinkedIn user.")

## Finishing up
st.subheader("Thank you for using my app!")
