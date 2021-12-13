import MeCab
import os


def extract_keyphrase_candidates(text, tokenizer):
    tagger = MeCab.Tagger()
    tagger.parse("")

    t = [to.split('\t') for to in tagger.parse(text).split('\n') if to]
    t = [(to[0], to[1].split(',')[0]) for to in t if len(to) > 1]

    keyphrase_candidates = []
    phrase = []

    tokens = []
    idx = len(t) - 1
    start_pos = -1
    end_pos = -1
    cnt = 0
    phrase_set = set()

    while idx >= 0:
        while idx >= 0 and t[idx][1] != '名詞':
            tokens.append(t[idx][0])
            idx -= 1
        
        if idx >= 0 and t[idx][1] == '名詞':
            tokens.append(t[idx][0])
            end_pos = len(tokens)
            phrase.append(t[idx][0])
            idx -= 1

        while idx >= 0 and t[idx][1] == '名詞':
            tokens.append(t[idx][0])
            phrase.append(t[idx][0])
            idx -= 1

        while idx >= 0 and t[idx][1] == '形容詞':
            tokens.append(t[idx][0])
            phrase.append(t[idx][0])
            idx -= 1

        if len(phrase) > 1:
            start_pos = len(tokens)
            keyphrase_candidates.append(('_'.join(phrase[::-1]), (len(t) - start_pos, len(t) - end_pos)))

        phrase = []
        start_pos = -1
        end_pos = -1

    while idx >= 0:
        tokens.extend(tokenizer.tokenize(t[idx][0])[::-1])
        idx -= 1


    outputs = []
    for keyphrase in keyphrase_candidates[::-1]:
        if keyphrase[0] not in phrase_set:
            outputs.append(keyphrase)
            phrase_set.add(keyphrase[0])
    return tokens[::-1], outputs
