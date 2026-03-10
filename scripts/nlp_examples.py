import fasttext
import nltk
import spacy

# NLTK is a set of rules that are useful for parsing language.
# Think a list of every type of punctuation as an example

# Spacy is built on neural networks and is used for things like
# parsing out noun phrases


# Break apart text into sentences
def break_sentences(txt: str) -> list[str]:
    """Break text into a list of sentences"""
    return nltk.tokenize.sent_tokenize(txt)


print(break_sentences("This is a test. Are you excited yet?"))


# Find noun phrases
class NounParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def phrases(self, txt: str) -> list[str]:
        """Convert text into noun phrases"""
        doc = self.nlp(txt.lower())
        words = []
        for phrase in doc.noun_chunks:
            word = phrase.text
            word = word.strip()
            if not (word == "" or word.isspace()):
                words.append(word.replace("\n", ""))
        return words


np = NounParser()
print(np.get_phrases("This is the biggest, fuzziest narwhal I have ever seen in my life!"))


ft_model = fasttext.load_model("data/cc.en.50.bin")
print(ft_model.get_sentence_vector("This is the fuzziest narwhal I've seen in my life!"))