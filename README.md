# imdb-sentiment
IMDB - Sentiment Analysis. Deep learning model to learn sentiment analysis from movie reviews
This project uses an Long short-term memory (LSTM) network on the IMDB dataset to predict sentiment classification

## Architecture

Embedding layer - encodes the text indices as a dense vector for analysis

LSTM 128 - <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">Long Short Term Memory layer</a> with 128 units. and a dropout value of 0.2, recurrent dropout of 0.2. 

Dense classification layer - 1 unit (maps to positive or negative) uses sigmoid activation function

## Results
```
Train on 25000 samples, validate on 25000 samples

Model accuracy: 0.82048

text: This movie was an amazing experience. Loved it!
sent score: 0.9999765157699585
classification: positive

text: Best movie I've seen in a while.
sent score: 0.9993841648101807
classification: positive

text: I fell asleep.
sent score: 0.1708667129278183
classification: negative
```

## Next steps
Analyze and use other RNN architectures to compare accuracy of models in NLP classification tasks. 
