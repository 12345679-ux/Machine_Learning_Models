import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Load your combined DataFrame
# Ensure you've saved combined_df to a CSV
combined_df = pd.read_csv('combined_df.csv')

st.title("Sentiment Analysis Dashboard")

# 2. Show metrics
st.header("Model vs VADER Agreement")
bert_3_counts = combined_df['bert_3'].value_counts().sort_index()
vader_3_counts = combined_df['vader_3'].value_counts().sort_index()
col1, col2 = st.columns(2)
col1.bar_chart(bert_3_counts)
col2.bar_chart(vader_3_counts)

# 3. Scatter Plot
st.header("VADER Compound vs BERT Stars")
fig, ax = plt.subplots()
ax.scatter(combined_df['compound'], combined_df['bert_star'], alpha=0.5)
ax.set_xlabel('VADER Compound')
ax.set_ylabel('BERT Stars')
st.pyplot(fig)

# 4. Boxplot
st.header("VADER Distribution by BERT Stars")
fig2, ax2 = plt.subplots()
data = [combined_df[combined_df['bert_star']==s]['compound'] for s in sorted(combined_df['bert_star'].unique())]
labels = [f"{s}★" for s in sorted(combined_df['bert_star'].unique())]
ax2.boxplot(data, labels=labels)
st.pyplot(fig2)

# 5. Word Clouds
st.header("Word Clouds")
pos_text = " ".join(combined_df[combined_df['bert_star']>=4]['content_x'])
neg_text = " ".join(combined_df[combined_df['bert_star']<=2]['content_x'])
col3, col4 = st.columns(2)
with col3:
    st.subheader("Positive")
    wc1 = WordCloud(width=300, height=150).generate(pos_text)
    st.image(wc1.to_array())
with col4:
    st.subheader("Negative")
    wc2 = WordCloud(width=300, height=150).generate(neg_text)
    st.image(wc2.to_array())