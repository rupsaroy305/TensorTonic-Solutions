def wordpiece_tokenize(text: str, vocab: dict,
                       unk_token: str = "[UNK]", max_word_length: int = 100) -> list:
    tokens= []
    for word in text.lower().split():
        if len(word)> max_word_length:
            tokens.append(unk_token)
            continue
        pieces= []
        start= 0
        while start<len(word):
            end= len(word)
            piece= None
            while end>start:
                subword= word[start:end]
                if start>0:
                    subword="##"+subword
                if subword in vocab:
                    piece= subword
                    break
                end-= 1
            if piece is None:
                pieces= [unk_token]
                break
            pieces.append(piece)
            start= end
        tokens.extend(pieces)
    return tokens