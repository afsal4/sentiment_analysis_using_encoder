import streamlit as st
import matplotlib.pyplot as plt 
import torch 
from encoder_pipeline import get_sentiment 

PATH='sentiment_lstm_cpu.pt'

def show_piechart(values):
    labels = ['Negative', 'Positive']
    explode = (0, 0.1) 

    # Adjust figure size and font size
    fig1, ax1 = plt.subplots(figsize=(4, 4))  # Decreased figure size
    ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'color': 'white', 'fontsize': 7})  # Decreased font size
    
    ax1.axis('equal') 
    fig1.patch.set_alpha(0)  
    fig1.patch.set_facecolor('#333333') 
    st.pyplot(fig1, use_container_width=True)
    




def load_model(path):
    model = torch.load(path)
    model.eval()
    return model


def main():
    st.title('Sentiment Analysis using Encoder')
    text = st.text_input('Enter the sentence to find the sentiment: ')

    if text:
            pred, percent = get_sentiment(text)
            color = 'yellow' if pred == 'Positive' else 'blue'
            st.markdown(f'### <span style="color: {color};">{pred}</span>', unsafe_allow_html=True)
            show_piechart(percent)

if __name__ == '__main__':
    main()