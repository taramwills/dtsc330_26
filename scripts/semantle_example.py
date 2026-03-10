import flask
import gensim
import numpy as np

app = flask.Flask("API")
fasttext_model_path = "data/cc.en.50.bin"
# ft_model = fasttext.load_model(fasttext_model_path)
gen_model = gensim.models.fasttext.load_facebook_vectors(fasttext_model_path)

# Hierarchical navigable small-world graphs (HNSW)


@app.route("/")
def heartbeat():
    return flask.jsonify({"alive": True})


@app.route("/sentence", methods = ["GET"])
def sentence():
    input_sentence = flask.request.args.get("sentence")
    if input_sentence and len(input_sentence) > 0:
        input_sentence = [v.lower() for v in input_sentence.split("_")]
        model_vectors = [gen_model.get_vector(v) for v in input_sentence]
        av_vector = np.mean(model_vectors, axis = 0)
        similar = gen_model.similar_by_vector(av_vector)
        return flask.jsonify({"success": True, "similar": similar})


if __name__ == "__main__":
    app.run(port = 8000)