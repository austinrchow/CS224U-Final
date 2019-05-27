import os
import collections
import re

def train_unigrams():
    counts = collections.Counter()
    total_counts = 0

    for root, dirs, files in os.walk('imsdb'):
        for file in files:
            file = root + "/" + file
            print(file)
    #
    #         with open(file, 'r') as f:
    #             for line in f:
    #                 arr = line.split()
    #                 arr.append("</s>")
    #                 for word in arr:
    #                     word = re.sub('\W+','', word)
    #                     counts[word] = 1
    #                     total_counts += 1
    #
    # #
    # #f = open("unigrams.txt", "w")
    # print(total_counts)
    # for word, count in counts.items():
    #     print(str(counts[word]) + "/" + str(total_counts))
    #     prob = counts[word] / total_counts
    #     print(word + " " + str(prob))
    #     #f.write(word + " " + str(prob))
    # #f.close()

train_unigrams()
