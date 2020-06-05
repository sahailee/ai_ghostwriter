# ai_ghostwriter
Generate lyrics with a model trained on a famous artist lyrics

This is the code of the webapp, url here.

In order to make this I used Flask to run the backend and generate text from the machine learning models.

One of the models used in this project, I trained from scratch, and the code to that is available in the training_simple_model.py file.
I experimented with many different model variations and different ways of tokenizing the corpus. In the end, I ended up using a model with
an embbedding layer and an LSTM layer with 400 hidden units. This model contains newlines at the end of some words. I tried having new
line characters as separate words themselves, but this caused the model to guess a new line character after every word it generates
because of the high probality of a new line character. I wanted the model to know when to end a line in the lyrics, so I chose to still
have a new line character but not as its own word. This does limit what the model can generate, but it was the best I could do given
the size of my corpus, the nature of hip-hop lyrics, and the amount of time I was willing to allow the models to train for.

The GPT-2 model is able to do all the things I dreamed and goals I had in mind for my model. Its able to mark choruses and verses, and
knows when it wants to end a line of the lyrics. This model is amazing and I don't think I could make a model that would come close to
how well the gpt-2 model performs by myself.  I will still continue to experiment with my simple model and try to improve it into a more
complex model.
