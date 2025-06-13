import time
import math
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import EncoderRNN, AttnDecoderRNN
from dataset import get_dataloader, prepareData

EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 时间工具函数
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# 损失绘图函数
def showPlot(points, title="Training Loss", save_path="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")

# 单个 epoch 的训练逻辑
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, epoch=None, total_epochs=None):
    encoder.train()
    decoder.train()

    total_loss = 0
    progress = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100,
                    desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for step, (input_tensor, target_tensor) in progress:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)),
                         target_tensor.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"\u2713 Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss

# 完整训练过程
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_every=1, plot_every=1, save_every=1000, model_save_path="models"):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    best_loss = float('inf')

    os.makedirs(model_save_path, exist_ok=True)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=5)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=5)

    print(f"Training for {n_epochs} epochs | LR={learning_rate} | Device={device}")

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, criterion,
                           epoch=epoch, total_epochs=n_epochs)

        print_loss_total += loss
        plot_loss_total += loss

        encoder_scheduler.step(loss)
        decoder_scheduler.step(loss)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(model_save_path, 'best_model.pth'))

        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(model_save_path, f'checkpoint_epoch_{epoch}.pth'))

    print(f"Training finished | Best loss: {best_loss:.4f}")
    showPlot(plot_losses)
    return plot_losses

# 模型加载
def load_model(encoder, decoder, model_path):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Model loaded from {model_path} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.4f}")
        return checkpoint['epoch'], checkpoint['loss']
    else:
        print(f"No model found at {model_path}")
        return 0, float('inf')

# 评估相关
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attentions = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, attentions

def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=5):
    encoder.eval()
    decoder.eval()

    print("\nEvaluation Samples:")
    for i in range(n):
        pair = random.choice(pairs)
        print(f"> {pair[0]}\n= {pair[1]}")
        output_words, attn = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        print(pair[0])
        print(output_words)
        showAttention(pair[0], output_words, attn[0:len(output_words), :], idx=i)
        print(f"< {' '.join(output_words)}")
        print("-" * 30)

# 修改后的 Attention 可视化函数
def showAttention(input_sentence, output_words, attentions, idx=0):
    print(attentions.shape)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    plt.tight_layout()
    filename = f'attention_plot_{idx}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved attention plot to {filename}")

# 主函数
def main():
    hidden_size = 64
    batch_size = 32
    n_epochs = 100
    learning_rate = 0.001

    print("Loading data...")
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
    _, _, pairs = prepareData('eng', 'fra', True)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    print(f"Input lang: {input_lang.name} | Output lang: {output_lang.name}")
    print(f"Train batches: {len(train_dataloader)}")

    load_model(encoder, decoder, "models/best_model.pth")

    # print("Starting training...")
    # train(train_dataloader, encoder, decoder, n_epochs, learning_rate)

    print("Evaluating...")
    evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=7)

if __name__ == "__main__":
    import random
    main()
