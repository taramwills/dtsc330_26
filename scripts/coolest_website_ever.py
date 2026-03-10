import flask
import fasttext
import numpy as np

website = flask.Flask('API')
# ft_model = fasttext.load_model('data/cc.en.50.bin')

gen_model = gensim.models.fasttext.load_facebook_vectors('.')

# Hierarchical navigable small world graphs (HNSW)

 
@website.route('/')
def heartbeat():
    return flask.jsonify({'alive': True})

@website.route('/vec')
def vec():
    input_words = flask.request.args.get('words')
    if input_words and len(input_words) > 0:
        words = input_words.split('_')
        vecs = [ft_model.get_sentence_vector(word) for word in words]
        avg_vec = np.mean(vecs, axis = 0)
        return flask.jsonify({'vector': avg_vec})

website.run(port = 8000)