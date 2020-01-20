import numpy as np
import argparse
import json
import re
import os
import csv
import statistics


FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
FUTURE_TENSE = {
    'will', 'gonna', 'shall'}

# Parts of Speech
PAST_TENSE = {
    'VBD', 'VBN'}
PUNCTUATION = {
    '.', ':', '\'', ',', 'NFP', 'HYPH', "''", "``"}
NOUNS = {
    'NNS', 'NN'}
PROPER_NOUNS = {
    'NNP', 'NNPS'}
ADVERBS = {
    'RB', 'RBR', 'RBS'}
WHWORDS = {
    'WDT', 'WP', 'WP$', 'WRB'}

COMMENT_CLASS = {'Male': 0, 'Female': 1}

# CSV dictionaries for optimized searching
bristol_map = {}
warringer_map = {}


def extract(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # List for features data
    feats = [0.0 for _ in range(173)]

    # Extract the first sixteen features
    non_punctuation_tokens = get_word_features(feats, comment)

    # Extract Bristol and Warringer features
    bristol_norms_features(feats, non_punctuation_tokens)
    warringer_features(feats, non_punctuation_tokens)

    return np.array(feats)

def get_word_features(feats, comment):
    ''' Extracts the first sixteen features. '''

    # Extract features that rely on capitalization.
    tokens = comment.split(" ")
    for token in tokens:
        word = token[0:token.rfind("/")]
        if len(word) > 2 and word.isupper():
            feats[0] += 1

    # Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    for i in range(len(tokens)):
        new_word = re.sub(r"(.*/)", lambda x: x.group(0).lower(), tokens[i])
        tokens[i] = new_word

    # Extract features that do not rely on capitalization.
    for i in range(len(tokens)):
        # Token declarations
        token = tokens[i]
        word = token[0:token.rfind("/")]
        pos = token[token.rfind("/") + 1:]
        next_token = tokens[i + 1] if len(tokens) > i + 1 else None

        # Feature counting for tokens
        if word in FIRST_PERSON_PRONOUNS:
            feats[1] += 1
        if word in SECOND_PERSON_PRONOUNS:
            feats[2] += 1
        if word in THIRD_PERSON_PRONOUNS:
            feats[3] += 1
        if pos == "CC":
            feats[4] += 1
        if pos in PAST_TENSE:
            feats[5] += 1
        if word in FUTURE_TENSE:
            feats[6] += 1
        # 'Going to VB'
        if word == "go" and pos == "VBG" and next_token:
            next_word = next_token[0:next_token.rfind("/")]
            if next_word == "to":
                feats[6] += 1
        if word == ",":
            feats[7] += 1
        if pos in PUNCTUATION and len(word) > 1:
            feats[8] += 1
        if pos in NOUNS:  # 5
            feats[9] += 1
        if pos in PROPER_NOUNS:  # 2
            feats[10] += 1
        if pos in ADVERBS:  # 2
            feats[11] += 1
        if pos in WHWORDS:  # 1
            feats[12] += 1
        if word in SLANG:  # 1
            feats[13] += 1

    # Average length of sentences
    num_sentences = comment.count("\n")
    num_sentences = num_sentences if num_sentences > 0 else 1
    num_tokens = len(tokens) - num_sentences
    average_length = num_tokens / num_sentences
    feats[14] = average_length

    # Average length of non punctuation tokens
    non_punctuation_tokens = []
    for token in tokens:
        word = token[0:token.rfind("/")]
        pos = token[token.rfind("/") + 1:]
        if pos not in PUNCTUATION and token != "\n":
            non_punctuation_tokens.append(word)

    average_length = 0
    if len(non_punctuation_tokens) > 0:
        average_length = sum(len(token) for token in non_punctuation_tokens) / len(non_punctuation_tokens)
    feats[15] = average_length
    feats[16] = num_sentences

    return non_punctuation_tokens

def bristol_norms_features(feats, non_punctuation_tokens):
    ''' Extracts all features related to the Bristol Norms CSV. '''

    # Collect data points from the Bristol csv
    aoa = []
    img = []
    fam = []
    for word in non_punctuation_tokens:
        first_char = word[0]
        try:
            a_v, i_v, f_v = bristol_map[first_char][word]
            aoa.append(a_v)
            img.append(i_v)
            fam.append(f_v)
        except KeyError:
            pass

    # Averages
    average_aoa = sum(aoa) / len(aoa) if len(aoa) > 0 else 0.0
    average_img = sum(img) / len(img) if len(img) > 0 else 0.0
    average_fam = sum(fam) / len(fam) if len(fam) > 0 else 0.0
    feats[17] = average_aoa
    feats[18] = average_img
    feats[19] = average_fam

    # Standard deviations
    stdev_aoa = statistics.stdev(aoa) if len(aoa) > 1 else 0.0
    stdev_img = statistics.stdev(img) if len(img) > 1 else 0.0
    stdev_fam = statistics.stdev(fam) if len(fam) > 1 else 0.0
    feats[20] = stdev_aoa
    feats[21] = stdev_img
    feats[22] = stdev_fam


def warringer_features(feats, non_punctuation_tokens):
    ''' Extracts all features related to the Warringer CSV. '''

    # Collect data points from the warringer csv
    v_mean = []
    a_mean = []
    d_mean = []

    for word in non_punctuation_tokens:
        first_char = word[0]
        try:
            v_m, a_m, d_m = warringer_map[first_char][word]
            v_mean.append(v_m)
            a_mean.append(a_m)
            d_mean.append(d_m)
        except KeyError:
            pass

    # Averages
    average_v_mean = sum(v_mean) / len(v_mean) if len(v_mean) > 0 else 0.0
    average_a_mean = sum(a_mean) / len(a_mean) if len(a_mean) > 0 else 0.0
    average_d_mean = sum(d_mean) / len(d_mean) if len(d_mean) > 0 else 0.0
    feats[23] = average_v_mean
    feats[24] = average_a_mean
    feats[25] = average_d_mean

    # Standard deviations
    stdev_v_mean = statistics.stdev(v_mean) if len(v_mean) > 1 else 0.0
    stdev_a_mean = statistics.stdev(a_mean) if len(a_mean) > 1 else 0.0
    stdev_d_mean = statistics.stdev(d_mean) if len(d_mean) > 1 else 0.0
    feats[26] = stdev_v_mean
    feats[27] = stdev_a_mean
    feats[28] = stdev_d_mean

def load_norms():
    ''' Loads CSV files and saves the data to dictionaries for fast searching. '''

    # Load lexical norm CSVs
    path_to_norms = args.a1_dir[0:args.a1_dir.rfind("/", 0, len(args.a1_dir) - 1)] + "/Wordlists/"
    bristol = "BristolNorms+GilhoolyLogie.csv"
    warringer = "Ratings_Warriner_et_al.csv"
    warringer = os.path.join(path_to_norms, warringer)

    bristol = os.path.join(path_to_norms, bristol)
    bristol_data = open(bristol, 'r')
    bristol_reader = csv.reader(bristol_data, delimiter=",")
    warringer_data = open(warringer, 'r')
    warringer_reader = csv.reader(warringer_data, delimiter=",")

    # Bristol dict
    for row in bristol_reader:
        if row[1] != "WORD" and row[1] != "":
            first_char = row[1][0]
            if first_char not in bristol_map:
                bristol_map[first_char] = {}
            bristol_map[first_char][row[1]] = (float(row[3]), float(row[4]), float(row[5]))

    # Warringer dict
    for row in warringer_reader:
        if row[1] != "Word" and row[1] != "":
            first_char = row[1][0]
            if first_char not in warringer_map:
                warringer_map[first_char] = {}
            warringer_map[first_char][row[1]] = (float(row[2]), float(row[5]), float(row[8]))


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # Initialize the Bristol and Warringer dictionaries
    load_norms()

    # For each comment, extract all features and record in numpy array feats
    for j in range(len(data)):
        comment = data[j]
        body = comment["body"]
        comment_class = comment["cat"]
        comment_id = comment["id"]

        # Use extract to find the first 29 features for each data point
        feats[j] = np.append(extract(body), COMMENT_CLASS[comment_class])

        #  Use extract2 to copy LIWC features (features 30-173)
        feats[j] = extract2(feats[j], comment_class, comment_id)

    # Save feature array to file
    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in PreprocessData", required=True)
    args = parser.parse_args()

    main(args)