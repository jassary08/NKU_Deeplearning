import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataset import randomTrainingExample
def train_step_rnn(category_tensor, line_tensor, rnn, criterion, learning_rate):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def train_step_lstm(category_tensor, line_tensor, lstm, criterion, learning_rate):
    hidden = lstm.initHidden()
    cell = lstm.initCell()
    lstm.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden, cell = lstm(line_tensor[i], hidden, cell)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 手动参数更新
    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def evaluate_rnn(line_tensor, rnn):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def evaluate_lstm(line_tensor, lstm):
    hidden = lstm.initHidden()
    cell = lstm.initCell()

    with torch.no_grad():  # 不需要梯度
        for i in range(line_tensor.size(0)):
            output, hidden, cell = lstm(line_tensor[i], hidden, cell)

    return output


def evaluate_accuracy_rnn(rnn_model, category_lines, all_categories, n_samples=1000):
    correct = 0
    with torch.no_grad():
        for _ in range(n_samples):
            category, line, _, line_tensor = randomTrainingExample(category_lines, all_categories)
            output = evaluate_rnn(line_tensor, rnn_model)
            guess, _ = categoryFromOutput(output, all_categories)
            if guess == category:
                correct += 1
    return correct / n_samples

def evaluate_accuracy_lstm(model, category_lines, all_categories, n_samples=1000):
    correct = 0
    with torch.no_grad():
        for _ in range(n_samples):
            category, line, _, line_tensor = randomTrainingExample(category_lines, all_categories)
            output = evaluate_lstm(line_tensor, model)
            guess, _ = categoryFromOutput(output, all_categories)
            if guess == category:
                correct += 1
    return correct / n_samples


def plot_accuracy(accuracies, step_interval=1000):
    plt.figure()
    steps = [i * step_interval for i in range(1, len(accuracies)+1)]
    plt.plot(steps, accuracies)
    plt.title("Validation Accuracy over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()


def plot_loss(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations (per 1000)')
    plt.ylabel('Loss')
    plt.show()


def plot_confusion(confusion, all_categories):
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
