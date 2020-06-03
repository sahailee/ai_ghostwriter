import gpt_2_simple as gpt2


def generate_lyrics_with_gpt(seed_text="Song:", next_words=0, temperature=1, model_name="", top_k=0):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, checkpoint_dir=model_name)
    if next_words is 0:
        return gpt2.generate(sess, prefix=seed_text, truncate='\nSong:',
                             checkpoint_dir=model_name, temperature=temperature, return_as_list=True, top_k=top_k)[0]
    else:
        return gpt2.generate(sess, prefix=seed_text, truncate='\nSong:',
                             checkpoint_dir=model_name, length=next_words, temperature=temperature,
                             return_as_list=True, top_k=top_k)[0]
