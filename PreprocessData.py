import os
import argparse
import json
import re
import html
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

GENDERS = {"Male": 0, "Female": 0}

def preprocess_text(comment):
    ''' Process comment text and apply nlp tagging and lemma '''
    processed_comment = clean_text(comment)

    # If the comment was not deleted, keep it in the data set
    if comment != "[deleted]":
        # Get segmented sentences
        sentences = nlp(processed_comment)

        # Replace the token with its lemma, unless it is a dash lemma
        processed_comment = ""
        for sentence in sentences.sents:
            for token in sentence:
                token_l = token.text if token.lemma_[0] == "-" else token.lemma_
                processed_comment += token_l + "/" + token.tag_ + " "
            processed_comment += "\n "

        # Remove trailing whitespace
        processed_comment = processed_comment.rstrip(" ")

    return processed_comment


def clean_text(comment):
    ''' Removes newlines, html, urls, and duplicate spaces.'''

    # Replace newlines with spaces
    processed_comment = comment
    processed_comment = re.sub(r"\n{1,}", " ", processed_comment)

    # Unescape html
    processed_comment = html.unescape(processed_comment)
    processed_comment = processed_comment.replace(u"\xa0", u" ")

    # Remove URLs
    processed_comment = re.sub(r"(http|www)\S+", "", processed_comment)

    # Remove duplicate spaces
    processed_comment = re.sub(r"\s\s+", " ", processed_comment)
    processed_comment = processed_comment.strip()

    return processed_comment

def get_gender(title):
    gender = "Unknown"
    title = clean_text(title)
    title = re.sub(r"[\(\[\]\)/]", "", title).lower()
    re_male = re.compile(r"(i|me|my)\s?[0-9][0-9]?m")
    re_female = re.compile(r"(i|me|my)\s?[0-9][0-9]?f")
    if bool(re_male.search(title)):
        gender = "Male"
    if bool(re_female.search(title)):
        gender =  "Female"

    return gender

def main(args):
    dataOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            # Load JSON file
            file_name = os.path.join(subdir, file)
            data = json.load(open(file_name))["data"]

            # Process comments and posts
            for i in range(1, len(data)):
                item = data[i]
                item = clean_json(item)

                # Process the body of the comment or post
                gender = "Unknown"
                new_body = ""

                # Known gender subreddits
                if "Men" in file_name or "Women" in file_name:
                    new_body = preprocess_text(item["body"])
                    gender = "Male" if "Men" in file_name else "Female"
                elif "selftext" in item:
                    new_body = preprocess_text(item["selftext"])
                    title = item["title"]
                    gender = get_gender(title)

                # Add category field for gender
                if gender == "Male":
                    item["gender"] = "Male"
                elif gender == "Female":
                    item["gender"] = "Female"

                # If there is still text in the body, save to output
                if new_body != "":
                    item["body"] = new_body
                    dataOutput.append(item)

        # Write output to a file
        fout = open(args.output, 'w')
        fout.write(json.dumps(dataOutput))
        fout.close()

def clean_json(item):
    ''' Only keep relevent fields from the comment json'''
    id = item.pop("id", None)
    body = item.pop("body", None)
    title = item.pop("title", None)
    selftext = item.pop("selftext", None)

    item = {}
    if id:
        item["id"] = id
    if body:
        item["body"] = body
    if title:
        item["title"] = title
    if selftext:
        item["selftext"] = selftext

    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data from Reddit')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    args = parser.parse_args()
    indir = os.path.join(".", "Data")
    main(args)