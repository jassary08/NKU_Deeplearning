import time
import math
import torch

from model import rnn
from dataset import loadData, randomTrainingExample, lineToTensor, all_letters, n_letters
from runner import train_step_rnn, evaluate_rnn, categoryFromOutput, plot_loss, plot_confusion,evaluate_accuracy_rnn,plot_accuracy

# 参数
learning_rate = 0.005
n_iters = 100000
print_every = 5000
plot_every = 1000

# 数据
category_lines, all_categories = loadData('./data/names/*.txt')
n_categories = len(all_categories)

# 模型
rnn = rnn(n_letters, 128, n_categories)
criterion = torch.nn.NLLLoss()
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'

# 训练循环
start = time.time()
accuracy_every = 1000
accuracies = []

# 修改训练循环
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
    output, loss = train_step_rnn(category_tensor, line_tensor, rnn, criterion, learning_rate)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output, all_categories)
        correct = '✓' if guess == category else f'✗ ({category})'
        print(f'{iter} {iter / n_iters * 100:.2f}% ({timeSince(start)}) {loss:.4f} {line} / {guess} {correct}')

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    if iter % accuracy_every == 0:
        acc = evaluate_accuracy_rnn(rnn, category_lines, all_categories)
        accuracies.append(acc)

# 绘制损失曲线
plot_loss(all_losses)

plot_accuracy(accuracies, accuracy_every)

# 混淆矩阵评估
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

for _ in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
    output = evaluate_rnn(line_tensor, rnn)
    guess, guess_i = categoryFromOutput(output, all_categories)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

plot_confusion(confusion, all_categories)
