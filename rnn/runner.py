import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def train_step(category_tensor, line_tensor, rnn, criterion, learning_rate):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def evaluate(line_tensor, rnn):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


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
