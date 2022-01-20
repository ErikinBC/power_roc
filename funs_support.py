import os

def makeifnot(path):
    if not os.path.exists(path):
        os.makedirs(path)

def round_up(num, factor):
    w, r = divmod(num, factor)
    return factor*(w + int(r>0))





