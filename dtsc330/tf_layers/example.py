from dtsc330.tf_layers import seq2seq_transformer
import random


def delete_random_letter(word):
    if len(word) <= 2:
        return word

    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i + 1:]


def swap_random_letters(word):
    if len(word) <= 2:
        return word

    i = random.randint(0, len(word) - 2)
    letters = list(word)
    letters[i], letters[i + 1] = letters[i + 1], letters[i]
    return "".join(letters)


def duplicate_random_letter(word):
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i] + word[i:]


def replace_random_letter(word):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    i = random.randint(0, len(word) - 1)
    new_letter = random.choice(alphabet)
    return word[:i] + new_letter + word[i + 1:]


def make_training_pairs(correct_words, examples_per_word=10):
    pairs = []

    for word in correct_words:
        pairs.append((word, word))

        for _ in range(examples_per_word):
            error_function = random.choice([
                delete_random_letter,
                swap_random_letters,
                duplicate_random_letter,
                replace_random_letter
            ])

            misspelled = error_function(word)

            if misspelled != word:
                pairs.append((misspelled, word))

    return pairs


if __name__ == "__main__":
    correct_words = [
        "receive",
        "definitely",
        "weird",
        "address",
        "accommodate",
        "separate",
        "until",
        "government",
        "because",
        "friend",
        "beautiful",
        "necessary",
        "different",
        "tomorrow",
        "wrong",
        "school",
        "language",
        "computer",
        "science",
        "python"
    ]

    manual_pairs = [
        ("recieve", "receive"),
        ("definately", "definitely"),
        ("wierd", "weird"),
        ("adres", "address"),
        ("acommodate", "accommodate"),
        ("seperate", "separate"),
        ("untill", "until"),
        ("goverment", "government"),
        ("becuase", "because"),
        ("freind", "friend"),
        ("beautifull", "beautiful"),
        ("neccessary", "necessary"),
        ("diffrent", "different"),
        ("tommorow", "tomorrow"),
        ("worng", "wrong")
    ]

    training_pairs = make_training_pairs(correct_words)
    training_pairs.extend(manual_pairs)

    model = seq2seq_transformer.Seq2SeqTransformer(max_len = 32)

    model.fit(training_pairs, training_epochs = 300)

    test_words = [
        "recieve",
        "definately",
        "wierd",
        "adres",
        "seperate",
        "goverment",
        "worng",
        "tommorow"
    ]

    for word in test_words:
        corrected = model.correct(word)
        print(word, "->", corrected)