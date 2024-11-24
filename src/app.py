#*******************************************************************************************
# Project : Miniproject - Visualization using Plotly and deployment on Streamlit Cloud
# Task    : Model Deployment on Streamlit Cloud
#*******************************************************************************************
#===========================================================================================
# 1. Import the required libraries
#===========================================================================================
import streamlit as st
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
from copy import deepcopy
import json
from datetime import datetime, timedelta
import pytz  # For timezone handling
import time
import pytz  # For timezone handling
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
#==================================================================================================

#----------------------------------------------------------------------------------------
# 1. Take input data file from user
#----------------------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df_raw = load_data(path="./data/Mastodon_Hashtag_Data.csv")
df = deepcopy(df_raw)

#----------------------------------------------------------------------------------------
# 2. Titel & Header
#----------------------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.image("./data/logo.png", width=100)

st.title("Mastodon Analysis")
st.header("Hashtag Analysis")

if st.checkbox("Show Dataframe"):
    st.subheader("Dataset:")
    st.dataframe(data=df)

#----------------------------------------------------------------------------------------
# 3. Data Preprocessing 
#----------------------------------------------------------------------------------------
def extract_number(url):    # Extract number from URL
    return url.split('/')[-1]

df['URL'] = df['URL'].apply(extract_number)  # Extract post id
df.rename(columns={"URL": "Post_id"}, inplace=True)


df['Date'] = pd.to_datetime(df['Date'])
df['Account_date'] = pd.to_datetime(df['Account_date'])

# today date
today = pd.to_datetime(datetime.now(pytz.UTC))

# difference in days between 'Account_date' and today
df['AccountAge_days'] = (today - df['Account_date']).dt.days

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply text cleaning
df['Content'] = df['Content'].apply(clean_text)

fig_1 = px.histogram(
    data_frame=df, 
    x='AccountAge_days', 
    range_x=[0, 1000],  
    range_y=[0, 8000],  
    opacity=0.8,        
    height=500)
fig_1.update_layout(
    title={"text": "Distribution of Mastodon Account Ages for Selected Hashtags", "font": {"size": 24}},
    xaxis={"title": {"text": "Account Age (Days)", "font": {"size": 16}},
           "tickfont": {"size": 14}},            
    yaxis={"title": {"text": "Number of Users", "font": {"size": 16}}, "tickfont": {"size": 14}},
)

#---------------------------------------------------------------------------------------
#3.2. Comparison of Selected Popular Hashtags in the last 30 days 
#
# Extract the hashtags column and split multiple hashtags into separate rows
df['Hashtags'] = df['Hashtags'].str.split(', ')
df_exploded = df.explode('Hashtags')

# function to normalize hashtags
def normalize_hashtags(hashtags):
    if 'kamalaharris' in hashtags or ('kamala' in hashtags or 'harris' in hashtags):
        return 'harris'
    if 'trump' in hashtags or 'donaldtrump' in hashtags:
        return 'trump'
    if 'elonmusk' in hashtags or 'musk' in hashtags:
        return 'elonmusk'
    if 'biden' in hashtags or 'joebiden' in hashtags:
        return 'biden'
    if 'scholz' in hashtags or 'spd' in hashtags:
        return 'scholz'
    return hashtags

# apply the normalize_hashtags function to each group of hashtags within each post.
df_exploded['Hashtags'] = df_exploded.groupby(level=0)['Hashtags'].transform(normalize_hashtags)
#groupby(level=0): Groups the DataFrame by the original index level (e.g., each post), 
#so that the normalization function is applied to all hashtags in the same post.
    
# Specify the hashtags to filter
specified_hashtags = ['trump', 'biden', 'scholz', 'afd', 
                      'taylorswift', 'volkswagen', 'elonmusk', 
                      'backtoschool', 'paralympics', 'harris']

# Filter the DataFrame for the specified hashtags
df_filtered = df_exploded[df_exploded['Hashtags'].isin(specified_hashtags)]

# Group by hashtags and count occurrences
hashtag_counts = df_filtered['Hashtags'].value_counts().reset_index()
hashtag_counts.columns = ['Hashtag', 'Count']

fig_2 = px.bar(hashtag_counts,
             x='Hashtag',
             y='Count',
             title='Count of Selected Hashtags Over the Last 30 Days',
             labels={'Hashtag': 'Hashtag', 'Count': 'Number of Posts'},
             text='Count',height=600)  

fig_2.update_layout(
    title={"font": {"size": 24}},
    xaxis={
        "title": {"text": "Hashtag", "font": {"size": 16}},
        "tickfont": {"size": 14} 
    },
    yaxis={
        "title": {"text": "Number of Posts (log)", "font": {"size": 16}},
        "tickfont": {"size": 14}, "type": "log"
    },
    showlegend=False,  
    # Customize text labels
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

#----------------------------------------------------------------------------------------
df_trump = df_exploded[df_exploded['Hashtags'] == 'trump'].copy()  # select 'trump' hashtag

df_trump['Date'] = pd.to_datetime(df_trump['Date'], errors='coerce')  # Adjust 'Date' to your actual column name

# Drop rows with invalid dates
df_trump = df_trump.dropna(subset=['Date'])

# Group by date and count occurrences
daily_trump_trend = df_trump.groupby(df_trump['Date'].dt.date).size().reset_index(name='Count') 

# Create the line chart
fig_3 = px.line(daily_trump_trend,
              x='Date',
              y='Count',
              title='Daily Trend of Trump Hashtag Posts in the Last 30 Days',
              labels={'Date': 'Date', 'Count': 'Number of Posts'},
              markers=True, height=550)

fig_3.update_layout(
    title={"text": "Daily Posts Trend for #Trump in the Last 30 Days", "font": {"size": 24}},
    xaxis={
        "title": {"text": "Date", "font": {"size": 16}},
        "tickfont": {"size": 14}  
    },
    yaxis={
        "title": {"text": "Number of Posts", "font": {"size": 16}},
        "tickfont": {"size": 14}  
    }
)
fig_3.update_traces(line=dict(color='darkblue'))

# Specify the point to annotate
selected_date = '2024-08-13'  # specific date
selected_count = 1000  # specific count value

fig_3.add_annotation(
    x=selected_date,
    y=selected_count,
    text="Trump's interview with Elon Musk",
    showarrow=False,  
    xshift=50,
    yshift=10,     
    font=dict(color='darkblue',size=14)  
)

selected_date = '2024-08-23'  #specific date
selected_count = 800 #  specific count value

fig_3.add_annotation(
    x=selected_date,
    y=selected_count,
    text="Robert F. Kennedy Jr. supports Trump",
    showarrow=True,
    arrowhead=2,
    ax=100,
    ay=-40,
    font=dict(color='darkblue', size=14),
    arrowcolor='darkblue'
)

selected_date = '2024-09-11'  # specific date
selected_count = 2180  # specific count value

fig_3.add_annotation(
    x=selected_date,
    y=selected_count,
    text="Trump and Harris' first presidential debate",
    showarrow=False,  
    xshift=-170,  # Shift text 
    font=dict(color='darkblue',size=14)  
) 


# Update layout to change plot background and show x and y axis lines
fig_3.update_layout(
    plot_bgcolor='#f2f5f7',  # Set the plot background color
    xaxis=dict(
        showline=True,  # Show x-axis line
        linecolor='black',  # Set x-axis line color
    ),
    yaxis=dict(
        showline=True,  # Show y-axis line
        linecolor='black',  # Set y-axis line color
    )
)

#----------------------------------------------------------------------------------------
# 4. Visualization 
#----------------------------------------------------------------------------------------
st.subheader("1. User Growth Data")

st.plotly_chart(fig_1)

#----------------------------------------------------------------------------------------

st.subheader("2. Trending Hashtags")
st.plotly_chart(fig_2)

st.plotly_chart(fig_3)

#----------------------------------------------------------------------------------------

st.subheader("3. Language Distribution")

#----------------------------------------------------------------------------------------

st.subheader("4. Sentiment Analysis")







