import os
import os.path
import math

import pickle

def ensureDir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def writeToFile(filename, content):
    ensureDir(filename)
    file = open(filename, 'w')
    file.truncate()
    file.write(content)
    file.close()


def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result


def serialize(filename, obj):
    ensureDir(filename)
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()


def deserialize(filename):
    file = open(filename, 'rb')
    result = pickle.load(file)
    file.close()
    return result
