import os

def train_unigrams():
    for root, dirs, files in os.walk('imsdb'):
        count = 0;
        for file in files:
            count += 1;
            print(file)
            print(count)

train_unigrams()
