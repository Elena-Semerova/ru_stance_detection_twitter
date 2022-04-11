import pymorphy2

def make_keywords(topic_dict):
    morph = pymorphy2.MorphAnalyzer()
    
    keywords_final = []
    for_processing = []

    for key, value in topic_dict.items():
        if value == 0:
            keywords_final.append(key)
        elif value == 1:
            word_parse = morph.parse(key)[0]

            if word_parse.tag.POS == 'INFN' or word_parse.tag.POS == 'VERB':
                new_words = list(set([lexem.inflect({'VERB'}).word for lexem in word_parse.lexeme]))
                keywords_final.extend(new_words)
            else:
                new_words = list(set([lexem.word for lexem in word_parse.lexeme]))
                keywords_final.extend(new_words)
        elif value == 2 or value == 3:
            for_processing.append(key)

    return keywords_final, for_processing
