import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression

st.title("Breakdown by Variable")

s = pd.DataFrame(pd.read_csv("social_media_usage.csv"))




s.describe()


# -------
# #### Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected
# --------



def clean_sm(i):
    return pd.DataFrame(np.where(i == 1,
                    1,
                    0))

# -------
# #### Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.
# -------



#Subset based on required features, using .loc for flexible DataFrame subsetting
ss = s.loc[:, ["web1h", "income", "educ2", "par", "marital", "sex", "age"]]
ss.loc[:, "sm_li"] = clean_sm(ss["web1h"])

#Rename Features
ss = ss.rename(columns={"educ2": "education",
                       "par": "parent",
                       "marital":"married",
                       "sex": "female"})

#Make Female Binary
ss["female"] = np.where(ss["female"] == 2,
                        1,
                        0)

#Make married Binary
ss["married"] = np.where(ss["married"] == 1,
                         1,
                         0)
#Make parent Binary
ss["parent"] = np.where(ss["parent"] == 1,
                        1,
                        0)

#Clean income, education, and age
ss["income"] = np.where(ss["income"] > 9,
                        np.nan,
                        ss["income"])

ss["education"] = np.where(ss["education"] > 8,
                           np.nan,
                           ss["education"])

ss["age"] = np.where(ss["age"] > 98,
                     np.nan,
                     ss["age"])

#Drop all na values
ss = ss.dropna()

#Make age label
ss["age_cat"] = np.where(
    ss["age"] >= 45,
    "Age >= 45",
    "Age < 45"
)

#All plots created based on ML visualization lessons
#Added Palette after creating the plots for design purposes

#User vs. Income
st.subheader("Income vs. User by Income Level")

#Income  mean
income_means = ss.groupby("income")["sm_li"].mean().reset_index()

fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(data = income_means, x="income", 
            y="sm_li",
            palette = ["#336699", "#6699CC"])
plt.xlabel("Income Level")
plt.ylabel("Linkedin Usage Rate by Level")
plt.title("Probability of Linkedin Use by Income Level")
plt.show()
st.pyplot(fig)

#User vs. Education Level
st.subheader("Education vs. User by Education Level")

#Education mean 
education_means = ss.groupby("education")["sm_li"].mean().reset_index()
fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(data = education_means, x="education",
             y="sm_li",
             palette = ["#336699", "#6699CC"])
plt.xlabel("Education Level")
plt.ylabel("Linkedin Usage Rate by Level")
plt.title("Probability of Linkedin Use by Education Level")
plt.show()
st.pyplot(fig)

st.subheader("Age vs. User Stratified by Age Group")
#Plot
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(data = ss,
               x="sm_li",
                 hue="age_cat",
                 palette = ["#336699", "#6699CC"])
plt.xlabel("User indicator")
plt.ylabel("Count")
plt.title("User vs. Non-User, Stratified by Age Group")
plt.legend(title="Age group")
plt.show()
st.pyplot(fig)

#User vs. Parental Status
st.subheader("Parental Status vs. User, Stratified by Parental Status")
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(data = ss,
               x="sm_li",
                 hue="parent",
                 palette = ["#336699", "#6699CC"])
plt.xlabel("User indicator")
plt.ylabel("Count")
plt.title("User vs. Non-User, Stratified by Parental Status")
plt.legend(title="Parental Status")
plt.show()
st.pyplot(fig)

#User vs. Marital Status
st.subheader("Marital Status vs. User, Stratified by Marital Status")
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(data = ss,
               x="sm_li",
                 hue="married",
                 palette = ["#336699", "#6699CC"])
plt.xlabel("User indicator")
plt.ylabel("Count")
plt.title("User vs. Non-User, Stratified by Marital Status")
plt.legend(title="Marital Status")
plt.show()
st.pyplot(fig)

#User vs. Gender
st.subheader("Gender vs. User, Stratified by Gender")
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(data = ss,
               x="sm_li",
                 hue="female",
                 palette = ["#336699", "#6699CC"])
plt.xlabel("User indicator")
plt.ylabel("Count")
plt.title("User vs. Non-User, Stratified by Gender")
plt.legend(title="Female")
plt.show()
st.pyplot(fig)    
