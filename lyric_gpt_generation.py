import gpt_2_simple as gpt2
import tensorflow as tf


def generate_lyrics_with_gpt(seed_text="Song:", next_words=0, temperature=1, model_name="", top_k=0, truncate='\nSong',
                             include_prefix=False):
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, checkpoint_dir=model_name)
    if next_words == 0:
        out = gpt2.generate(sess, prefix=seed_text, truncate=truncate, include_prefix=include_prefix,
                            checkpoint_dir=model_name, temperature=temperature, return_as_list=True, top_k=top_k)[0]

    else:
        out = gpt2.generate(sess, prefix=seed_text, truncate=truncate, include_prefix=include_prefix,
                            checkpoint_dir=model_name, length=next_words, temperature=temperature,
                            return_as_list=True, top_k=top_k)[0]
    sess.close()
    return out
