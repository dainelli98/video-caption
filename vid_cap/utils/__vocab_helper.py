import pandas as pd
import numpy as np
import collections
import csv

def get_sorted_words_by_frequency(captions_path, words_amount=10000):
    df = pd.read_csv(captions_path)

    token_dictionary = np.concatenate(df.iloc[:, 0].str.split().values)

    unique_token_dictionary = set(token_dictionary)

    words_frequencies = collections.Counter(unique_token_dictionary)
    words_frequencies = sorted(words_frequencies, key=lambda x: words_frequencies[x], reverse=True)

    words_frequencies.insert(0, "<eos>")
    words_frequencies.insert(0, "<sos>")
    words_frequencies.insert(0, "<unk>")
    words_frequencies.insert(0, "<pad>")

    truncated_words_frequencies = words_frequencies[:words_amount]

    return truncated_words_frequencies

def get_sorted_sentences_by_length(captions_path):
    sentences_dict = {}

    with open(captions_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present
        for row in reader:
            sentence = row[0]  # Sentence is in the first column
            video_id = row[1]  # Video ID is in the second column
            sentences_dict[video_id] = sentence

    sorted_sentences_dict = {k: v for k, v in sorted(sentences_dict.items(), key=lambda x: len(x[1]), reverse=True)}

    return sorted_sentences_dict
