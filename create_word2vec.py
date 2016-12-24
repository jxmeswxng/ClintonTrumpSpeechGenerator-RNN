#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
from gensim.models import word2vec
import numpy as np


class Word2Vec(object):

    def __init__(self):
        self.url_debates = "http://www.presidency.ucsb.edu/debates.php"
        self.url_clinton = "http://www.presidency.ucsb.edu/2016_election_speeches.php?candidate=70&campaign=2016CLINTON&doctype=5000"
        self.url_trump = "http://www.presidency.ucsb.edu/2016_election_speeches.php?candidate=45&campaign=2016TRUMP&doctype=5000"

    def __call__(self):

        # load lists of sentences
        lists_sentences_clinton, lists_sentences_trump = self.scraper()

        # create vector representation of words with word2vec
        model_clinton = word2vec.Word2Vec(lists_sentences_clinton,
                                          size=300,
                                          min_count=1,
                                          sample=1e-2,
                                          workers=4)
        model_trump = word2vec.Word2Vec(lists_sentences_trump,
                                        size=300,
                                        min_count=1,
                                        sample=1e-7,
                                        workers=4)

        # build word to vector representation dictionaries
        words_clinton = sum(lists_sentences_clinton, [])
        words_trump = sum(lists_sentences_trump, [])
        w2v_dict_clinton = {i: model_clinton[i] for i in words_clinton}
        w2v_dict_trump = {i: model_trump[i] for i in words_trump}

        # ignore sentences with less than or equal to three words
        s_lists_sentences_clinton = [x for x in lists_sentences_clinton if len(x) > 3]
        s_lists_sentences_trump = [x for x in lists_sentences_trump if len(x) > 3]

        return s_lists_sentences_clinton, s_lists_sentences_trump, w2v_dict_clinton, w2v_dict_trump, model_clinton, model_trump

    def scraper(self):

        # get debate links
        r = requests.get(self.url_debates)
        s = BeautifulSoup(r.text, "html5lib")
        tags = s.find_all("a")
        urls_debates = []
        for i in tags:
            try:
                condition = i.text.split()
            except:
                pass
            if "Debate" in condition:
                urls_debates.append(i["href"])

        # get clinton campaign speeches links
        site = "http://www.presidency.ucsb.edu"
        r_clinton = requests.get(self.url_clinton)
        s_clinton = BeautifulSoup(r_clinton.text, "html5lib")
        tags_clinton = [(i.find_all("td", {"class": "listdate"}), i.find_all("a"))
                        for i in s_clinton.find_all("tr")]
        campaign_clinton_urls = []
        for i in tags_clinton:
            date = False
            try:
                date = i[0][1]
            except:
                pass
            if date is not False:
                year = int(date.text[-4:])
                if year >= 2015:
                    campaign_clinton_urls.append(site + i[1][0]["href"][2:])

        # get trump campaign speeches links
        r_trump = requests.get(self.url_trump)
        s_trump = BeautifulSoup(r_trump.text, "html5lib")
        tags_trump = [(i.find_all("td", {"class": "listdate"}), i.find_all("a"))
                      for i in s_trump.find_all("tr")]
        campaign_trump_urls = []
        for i in tags_trump:
            date = False
            try:
                date = i[0][1]
            except:
                pass
            if date is not False:
                year = int(date.text[-4:])
                if year >= 2015:
                    campaign_trump_urls.append(site + i[1][0]["href"][2:])
        campaign_trump_urls = campaign_trump_urls[2:]

        # pull debate text
        count = 1
        n = len(urls_debates)
        lists_sentences_clinton_debate = []
        lists_sentences_trump_debate = []
        for i in urls_debates:
            # print("{} out of {}".format(count, n))
            count += 1
            r = requests.get(i)
            s = BeautifulSoup(r.text, "html5lib")
            tag = s.find_all("span", {"class": "displaytext"})

            # words
            words = str(tag[0]).lower().split("<b>")
            clinton = [i[11:] for i in words if i[:8] == "clinton:"]
            trump = [i[11:] for i in words if i[:6] == "trump:"]
            tag_chars = ["p", "b", "i"]
            strings_sentences_clinton = [k.strip() for j in [re.findall(r"[a-zA-Z ]+", i) for i in clinton]
                                         for k in j if k not in tag_chars]
            strings_sentences_trump = [k.strip() for j in [re.findall(r"[a-zA-Z ]+", i) for i in trump]
                                       for k in j if k not in tag_chars]
            lists_sentences_clinton_debate += [i.split() for i in strings_sentences_clinton if len(i.split()) > 0]
            lists_sentences_trump_debate += [i.split() for i in strings_sentences_trump if len(i.split()) > 0]

        # pull clinton campaign speech text
        count = 1
        n = len(campaign_clinton_urls)
        lists_sentences_clinton_campaign = []
        for i in campaign_clinton_urls:
            # print("{} out of {}".format(count, n))
            count += 1
            r = requests.get(i)
            s = BeautifulSoup(r.text, "html5lib")
            tag = s.find_all("span", {"class": "displaytext"})

            # words
            all_sentences_b = str(tag[0]).lower().split("<b>")
            all_sentences_i = str(tag[0]).lower().split("<i>")
            if len(all_sentences_b) > len(all_sentences_i):
                all_sentences = all_sentences_b
            else:
                all_sentences = all_sentences_i
            clinton_sentences = [i[7:] for i in all_sentences if i[:7] == "clinton"]
            tag_chars = ["p", "b", "i"]
            strings_sentences_clinton_campaign = [k.strip() for j in [re.findall(r"[a-zA-Z ]+", i) for i in clinton_sentences]
                                                  for k in j if k not in tag_chars]
            lists_sentences_clinton_campaign += [i.split() for i in strings_sentences_clinton_campaign if len(i.split()) > 0][:-40]

        # pull trump campaign speech text
        count = 1
        n = len(campaign_trump_urls)
        lists_sentences_trump_campaign = []
        for i in campaign_trump_urls:
            # print("{} out of {}".format(count, n))
            count += 1
            r = requests.get(i)
            s = BeautifulSoup(r.text, "html5lib")
            tag = s.find_all("span", {"class": "displaytext"})

            # words
            all_sentences_b = str(tag[0]).lower().split("<b>")
            all_sentences_i = str(tag[0]).lower().split("<i>")
            if len(all_sentences_b) > len(all_sentences_i):
                all_sentences = all_sentences_b
            else:
                all_sentences = all_sentences_i
            tag_chars = ["p", "b", "i"]
            strings_sentences_trump_campaign = [k.strip() for j in [re.findall(r"[a-zA-Z ]+", i) for i in all_sentences]
                                                for k in j if k not in tag_chars]
            lists_sentences_trump_campaign += [i.split() for i in strings_sentences_trump_campaign if len(i.split()) > 0][:-40]

        # compile lists of sentences
        lists_sentences_clinton = lists_sentences_clinton_debate + lists_sentences_clinton_campaign
        lists_sentences_trump = lists_sentences_trump_debate + lists_sentences_trump_campaign

        return lists_sentences_clinton, lists_sentences_trump

    def generateInputData(self, lists_sentences, model):

        # create 3-grams from sentences
        trainData = []
        trainLabels = []
        SLIDER = 3
        for sentence in lists_sentences:
            for i in range(0, len(sentence) - SLIDER):
                if i < (len(sentence) - SLIDER):
                    trainData.append(np.array([model[x] for x in sentence[i: i + SLIDER]]))
                    trainLabels.append(np.array(model[sentence[i + SLIDER]]))

        return trainData, trainLabels


def main():
    pass

if __name__ == "__main__":
    main()
