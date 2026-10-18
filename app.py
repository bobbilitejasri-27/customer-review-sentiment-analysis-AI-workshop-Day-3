import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.title("💬 Customer Review Sentiment Analyzer")
st.write("Classifies reviews as positive, neutral or negative.")

# Sample Data - 3 classes
data = {
    'review': [
        'I loved every moment', 'Great product', 'Excellent quality',
        'Worst product', 'Bad quality', 'Terrible',
        'It is okay', 'Average product', 'Works fine'
    ],
    'sentiment': ['positive','positive','positive','negative','negative','negative','neutral','neutral','neutral']
}
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
model = LogisticRegression(max_iter=200)
model.fit(X, df['sentiment'])

tab1, tab2 = st.tabs(["Single review", "Batch (CSV)"])
with tab1:
    text = st.text_area("Enter a customer review", "I loved every moment")
    if st.button("Analyze"):
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        st.success(f"{pred.upper()} - Confidence: {max(prob)*100:.1f}%")
        fig, ax = plt.subplots()
        ax.bar(model.classes_, prob)
        st.pyplot(fig)

with tab2:
    file = st.file_uploader("Upload CSV", type=['csv'])
    if file:
        df2 = pd.read_csv(file)
        pred2 = model.predict(vectorizer.transform(df2['review']))
        df2['sentiment'] = pred2
        st.dataframe(df2)
        st.bar_chart(df2['sentiment'].value_counts())