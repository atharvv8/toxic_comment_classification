import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

df=pd.read_csv('Data.csv')
X = df['comment_text']
vectorizer = TextVectorization(max_tokens=200000,output_sequence_length=1800,output_mode='int')
model=tf.keras.models.load_model('toxicity')
model.summary()
vectorizer.adapt(X.values)

def score_comment(text):
    predictions = model.predict(np.array([text])) > 0.5
    prediction_text = ""
    for i, col in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
        prediction_text += f"{col:<{20}}: {predictions[0][i]:<{5}}\n"
    print(prediction_text)

print("Enter your comment")
str = input()
print(str)
score_comment(str)
