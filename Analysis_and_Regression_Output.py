# coding: utf-8

# -------
# # Paul Glenchur
# ## Final Project
# ### 12/9/2025
# -------

# ----
# #### Package Imports/Setup
# ------

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

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

st.title("""
         Linkedin User Prediction App
         """)

#Download Button! Seemed interesting in videos
with open("social_media_usage_README.txt", "rb") as file:
    st.download_button(
        label = "Variable Definitions",
        data = file,
        file_name = "social_media_usage_README.txt",
        mime = "text/txt"
    )

s = pd.DataFrame(pd.read_csv("social_media_usage.csv"))



#EDA
s.describe()

def clean_sm(i):
    return pd.DataFrame(np.where(i == 1,
                    1,
                    0))

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

ss.isna().sum()

#EDA
st.header("Exploratory Data Analysis")
raw_desc = ss.drop(columns=["web1h"]).describe()

st.subheader("Early Summary Statistics")
raw_desc = st.columns(1)[0].write(raw_desc)

st.subheader("Mean Comparison by LinkedIn User")
st.write(ss.groupby("sm_li")[["income", "education", "parent", "married", "female", "age"]].mean())



# Set up for Pairplot/Model Creation
features = ["income", "education", "parent",
            "married", "female", "age"]


eda_df = ss.copy()
eda_df["linked_in_label"] = np.where(ss["sm_li"] == 1, "Linkedin User", "Not a Linkedin User")


eda_plot = sns.pairplot( 
    data = eda_df,
    vars=features,
    hue="linked_in_label",
    diag_kind="kde",
    corner=True 
)
eda_plot._legend.set_title("Linkedin Label")
plt.show()

x = ss[["income", "education", "parent", "married", "female", "age"]]
y = ss["sm_li"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.20,
    stratify=y,
    random_state=55
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

st.subheader("Model Exploration")

linkedin_reg = LogisticRegression(
    class_weight="balanced",  
    max_iter=1000             
)
linkedin_reg.fit(x_train, y_train)

#Get Probabilities
y_prob = linkedin_reg.predict_proba(x_test)[:,1]

#Plot ROC Curve & Calculate Supporting Metrics
y_pred = linkedin_reg.predict(x_test) #Scikit-Learn threshold automatically set at 0.5
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc_log = roc_auc_score(y_test, y_prob)

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Model (AUC={roc_auc_log:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on Test Set")
plt.legend()
textstr = '\n'.join((
                      '0       1',
    f'  0  {cm[0,0]:3d}    {cm[0,1]:3d}',
    f'  1  {cm[1,0]:3d}    {cm[1,1]:3d}',
    '',
    f'Accuracy: {accuracy:.3f}',
    f'AUC:       {roc_auc_log:.3f}'
))

plt.text(
    0.98, 0.02, textstr,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8)
)
plt.show()

st.subheader("In-sample Confusion Matrix")
cm_df = (pd.DataFrame(
    cm,
    index=["Actual 0 (Negative)", "Actual 1 (Positive)"],
    columns=["Predicted 0 (Negative)", "Predicted 1 (Positive)"]
))
print("\nIn-sample confusion matrix (threshold = 0.5):")
cm_df


by_hand_precision = (49 / (49+52))
by_hand_recall = (49 / (49+35))
by_hand_f1_score =  (2 * by_hand_precision * by_hand_recall) / (by_hand_precision + by_hand_recall)
print(f' Precision Calculated Manually = {by_hand_precision}')
print(f'Recall Calculated Manually = {by_hand_recall}')
print(f' F1 Score Calculated Manually = {by_hand_f1_score}')
print(classification_report(y_test, y_pred))

st.subheader("Model Performance")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("ROC-AUC", f"{roc_auc_log:.3f}")
col3.metric("Precision", f"{by_hand_precision:.3f}")
col4.metric("Recall", f"{by_hand_recall:.3f}")
col5.metric("F1 Score", f"{by_hand_f1_score:.3f}")

st.subheader("ROC AUC Curve")
auc_plot = plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Model (AUC={roc_auc_log:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on Test Set")
plt.legend()
st.pyplot(auc_plot)

user1 = pd.DataFrame({
    "income": [8],
    "education": [7],
    "parent": [0],
    "married": [1],
    "female":[1],
    "age":[42]
})

user2 = pd.DataFrame({
    "income": [8],
    "education": [7],
    "parent": [0],
    "married": [1],
    "female":[1],
    "age":[82]
})

prob_user1 = linkedin_reg.predict_proba(user1)[:, 1]
prob_user2 = linkedin_reg.predict_proba(user2)[:, 1]

print(f' The probability of user1 being a Linkedin user is {prob_user1}')
print(f' The probability of user2 being a Linkedin user is {prob_user2}')

#Feature selection sidebar
st.sidebar.header("Feature Selection for Prediction")
def feature_selection():
    income = st.sidebar.slider("Income (1–9)", 1, 9, 8)
    education = st.sidebar.slider("Education Level (1–8)", 1, 8, 7)
    parent = st.sidebar.selectbox("Parent?", [0, 1])
    married = st.sidebar.selectbox("Married?", [0, 1])
    female = st.sidebar.selectbox("female?", [0, 1])
    age = st.sidebar.slider("Age", 18, 98, 42)
    features =  pd.DataFrame({
        "income": [income],
        "education": [education],
        "parent": [parent],
        "married": [married],
        "female": [female],
        "age": [age]
    })
    return features

user_features = feature_selection()

st.subheader("User Input Features for Prediction")
st.write("Select values from sidebar to fit desired features.")
st.write(user_features)

user_prob = linkedin_reg.predict_proba(user_features)[:, 1]
st.subheader("Prediction Result")
st.metric("Probability this person uses LinkedIn", f"{user_prob[0]:.3f}")
st.write("Model Predicts: LinkedinUser" if user_prob[0] >= 0.5 else "Model Predicts: Not a Linkedin User")

#Refit in stats models for better marginal effects access
#Balance class weights first
pos_idx = y_train.index[y_train == 1]
neg_idx = y_train.index[y_train == 0]

rng = np.random.default_rng(seed=55)#For reproducability purposes
keep_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
keep_idx = np.concatenate([pos_idx, keep_neg])

x_train_bal = x_train.loc[keep_idx].copy()
y_train_bal = y_train.loc[keep_idx].copy()

#Fit Mode
sm_log = sm.Logit(
    y_train_bal,
    sm.add_constant(x_train_bal)).fit()

#Marginal Effects
# Average marginal effect from logistic model
mfx_approx_sm_model = sm_log.get_margeff(at="overall", method="dydx")

# Exact Odds marginal effects
mfx_odds_sm_model = np.exp(sm_log.params) - 1

# Get the feature names used in the model, excluding the constant
sm_model_feature_names = ["income", "education", "parent", "married","female","age"]

# Look at the result in a dataframe

st.subheader("Marginal Effects from Statsmodels Logistic Regression")

mfx_df_sm_model = pd.DataFrame({
    "Percent Change in Odds Per 1 Unit Increase": mfx_odds_sm_model[1:] * 100, # drop the first because we don't care about the constant
    "Percent change in Probability Per 1 Unit Increase": mfx_approx_sm_model.margeff * 100
})

st.write(mfx_df_sm_model)

st.write("Results show that the feature with the greatest positive impact on the probability of being a Linkedin user is education. The feature with the greatest negative impact on the probability of being a Linkedin user is marriage status.")

