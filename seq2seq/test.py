import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="E:/Downloads/tokenizer.model")
with open("vocab.txt", "w", encoding="utf-8") as out:
    for i in range(sp.get_piece_size()):
        out.write(f"{sp.id_to_piece(i)}\n")