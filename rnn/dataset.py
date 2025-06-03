import glob
import os
import unicodedata
import string
import random
import torch

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def readLines(filename):
    try:
        content = open(filename, encoding='utf-8').read()
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        return []
    lines = content.strip().split('\n')
    ascii_lines = [unicodeToAscii(line) for line in lines]
    return ascii_lines


def loadData(path):
    category_lines = {}
    all_categories = []
    files = findFiles(path)
    print(f"Found files: {files}")
    for filename in files:
        category = os.path.splitext(os.path.basename(filename))[0]
        print(f"Loading category: {category}")
        lines = readLines(filename)
        print(f"  Loaded {len(lines)} names")
        all_categories.append(category)
        category_lines[category] = lines
    return category_lines, all_categories

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(category_lines, all_categories):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
