from flask import Flask, render_template, request, jsonify, make_response
from lyric_generation import generate_stochastic_sampling
from lyric_gpt_generation import generate_lyrics_with_gpt
import random

app = Flask(__name__)

artist_map = {
    0: 'Random',
    1: 'Drake',
    2: 'Frank Ocean',
    3: 'Future',
    4: 'JAY Z',
    5: 'Kanye West',
    6: 'Kendrick Lamar',
    7: 'Playboi Carti',
    8: 'Rihanna',
    9: 'The Weeknd',
    10: 'Travis Scott'
}

title_top = 'gpt2/all_title_top/checkpoint'
title_bot = 'gpt2/all_title_bottom/checkpoint'


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')


@app.route("/background", methods=["POST"])
def generate_lyrics():
    req = request.get_json()
    is_gpt = req['gpt']
    is_gpt = int(is_gpt)
    seed_text = req['seed']
    next_words = req['count']
    temperature = req['temp']
    temperature = float(temperature)
    if is_gpt is 1:
        title = req['title']
        artist = int(req['artist'])
        topk = int(req['topk'])
        if artist is 0:
            artist = random.randint(1, 11)
        if title is '' and seed_text is '':
            model_name = title_top
            seed_text = 'Song: Artist: ' + artist_map[artist] + ' Title: '
        elif title is '':
            model_name = title_bot
            seed_text = 'Song: Artist: ' + artist_map[artist] + '\n' + seed_text
        else:
            model_name = title_top
            seed_text = 'Song: Artist: ' + artist_map[artist] + 'Title: ' + title + '\n'
        if next_words is '':
            next_words = 0
        else:
            next_words = int(next_words)
        output = generate_lyrics_with_gpt(seed_text=seed_text, next_words=next_words, temperature=temperature,
                                          model_name=model_name, top_k=topk)
        end_of_first = output.find('\n')
        if model_name is title_top:
            title = output[output.find('Title: ') + 7:end_of_first]
            output = output[end_of_first + 1:]
        else:
            end_of_lyric = output.find('Title: ')
            if end_of_lyric is -1:
                title = 'Generated Lyrics'
            else:
                title = output[end_of_lyric + 7:]
            output = output[end_of_first+1:end_of_lyric]
        artist = 'By AI GhostWriter in the style of ' + artist_map[artist]
    else:
        title = 'Generated Lyrics'
        artist = 'By AI GhostWriter in the style of Drake'
        if next_words is '' or next_words is '0':
            output = ''
            return make_response(jsonify({"lyrics": output, "title": title, "artist": artist}), 200)
        next_words = int(next_words)
        output = generate_stochastic_sampling(seed_text, next_words, temperature)
    res = make_response(jsonify({"lyrics": output, "title": title, "artist": artist}), 200)
    return res


if __name__ == "__main__":
    app.run()
