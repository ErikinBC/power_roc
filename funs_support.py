import os

def makeifnot(path):
    if not os.path.exists(path):
        os.makedirs(path)







