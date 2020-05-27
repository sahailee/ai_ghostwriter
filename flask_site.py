from flask import Flask, render_template, request
from lyric_generation import generate_stochastic_sampling

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        seed_text = request.form['seed']
        next_words = int(request.form['count'])
        temperature = int(request.form['temp']) / 50.0
        print(temperature)
        model_name = 'model_f.h5'
        tokenizer_name = tokenizer_verse_1 = 'tokenizer_verse_newlines_1.pickle'
        input_seq = 'input_sequence_verses_5.pickle'
        output = generate_stochastic_sampling(seed_text, next_words, temperature, model_name, tokenizer_name, input_seq)
        print(output)
        print('\n' in output)
        return "<h1>" + output + "<h1>"

    return render_template('index.html')


if __name__ == "__main__":
    app.run()
