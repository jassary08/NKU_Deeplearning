import time
import math
import random
import os
import glob
import unicodedata
import string

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import rnn  # 假设这里是你自己写的 RNN 模型类

# ---------------------------
# 参数配置
# ---------------------------
learning_rate = 0.005
n_iters = 100000
print_every = 5000
plot_every = 1000

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# ---------------------------
# 数据加载与预处理
# ---------------------------
def findFiles(path):
    return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# 读取所有类别（语言）和名字
category_lines = {}
all_categories = []

for filename in findFiles('../data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    category_lines[category] = readLines(filename)

n_categories = len(all_categories)

# ---------------------------
# 工具函数
# ---------------------------
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

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# ---------------------------
# 模型与训练
# ---------------------------
criterion = nn.NLLLoss()
current_loss = 0
all_losses = []

def train(category_tensor, line_tensor):
    h0, c0 = rnn.initHidden()
    rnn.zero_grad()

    output, h, c = rnn(line_tensor, h0, c0)
    loss = criterion(output[-1], category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

# ---------------------------
# 主训练循环
# ---------------------------
start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output[-1])
        correct = '✓' if guess == category else f'✗ ({category})'
        print(f'{iter} {iter / n_iters * 100:.2f}% ({timeSince(start)}) {loss:.4f} {line} / {guess} {correct}')

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# ---------------------------
# 损失曲线可视化
# ---------------------------
plt.figure()
plt.plot(all_losses)
plt.title('Training Loss')
plt.xlabel('Iterations (per 1000)')
plt.ylabel('Loss')
plt.show()

# ---------------------------
# 评估与混淆矩阵
# ---------------------------
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

def evaluate(line_tensor):
    h0, c0 = rnn.initHidden()
    output = rnn(line_tensor, h0, c0)
    return output[-1]

for _ in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 混淆矩阵可视化
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.title('Confusion Matrix')
plt.show()
