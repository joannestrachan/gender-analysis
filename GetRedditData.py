import requests
import json
import argparse
import sys

def getPushshiftData(max):
    '''

    :param max: the max number of comments or posts to extract
    :return:
    '''
    # Postshift URLS
    url_askmen = "https://api.pushshift.io/reddit/search/comment/?size=" + str(max) + "&subreddit=askmen&score=>100"
    url_askwomen = "https://api.pushshift.io/reddit/search/comment/?size=" + str(max) + "&subreddit=askwomen&score=>100"
    url_relationship_advice = "https://api.pushshift.io/reddit/search/submission/?size=" + str(max * 2) +\
                              "&subreddit=relationship_advice&score=>100"
    url_men = "https://api.pushshift.io/reddit/search/comment/?size=" + str(max) + "&subreddit=men&score=>100"
    url_women = "https://api.pushshift.io/reddit/search/comment/?size=" + str(max) + "&subreddit=women&score=>100"


    # Make requests
    r = requests.get(url_askmen)
    askmen_data = json.loads(r.content)
    r = requests.get(url_askwomen)
    askwomen_data = json.loads(r.content)
    r = requests.get(url_relationship_advice)
    relationship_data = json.loads(r.content)
    r = requests.get(url_men)
    men_data = json.loads(r.content)
    r = requests.get(url_women)
    women_data = json.loads(r.content)

    return askmen_data, askwomen_data, relationship_data, men_data, women_data


def main(args):
    askmen_data, askwomen_data, relationship_data, men_data, women_data = getPushshiftData(args.max)

    # Write output to file
    with open("Data/AskMen", "w") as output:
        json.dump(askmen_data, output)

    with open("Data/AskWomen", "w") as output:
        json.dump(askwomen_data, output)

    with open("Data/RelationshipAdvice", "w") as output:
        json.dump(relationship_data, output)

    with open("Data/Men", "w") as output:
        json.dump(men_data, output)

    with open("Data/Women", "w") as output:
        json.dump(women_data, output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data from pushshift')
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)