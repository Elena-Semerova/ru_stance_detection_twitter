from typing import Dict, List, Tuple

import pymorphy2

LIST_TAGS_POS = ["INFN", "VERB"]


def make_keywords(
    topic_dict: Dict[str, int], list_pos_tags: List[str] = LIST_TAGS_POS
) -> Tuple[List[str], List[str]]:
    """
    Making list with final keywords and list with keywords that should be processing

    Params:
    -------
        topic_dict (Dict[str, int]): dictionary with some keywords by which new extended lists will be built
        list_pos_tags (List[str]): list of pos tags

    Returns:
    --------
        (Tuple[List[str], List[str]]): list with final keywords and list with keywords for processing
    """
    morph = pymorphy2.MorphAnalyzer()

    keywords_final = []
    for_processing = []

    for key, value in topic_dict.items():
        if value == 0:
            keywords_final.append(key)
        elif value == 1:
            word_parse = morph.parse(key)[0]

            if word_parse.tag.POS in list_pos_tags:
                new_words = list(
                    set([lexem.inflect({"VERB"}).word for lexem in word_parse.lexeme])
                )
                keywords_final.extend(new_words)
            else:
                new_words = list(set([lexem.word for lexem in word_parse.lexeme]))
                keywords_final.extend(new_words)
        elif value == 2 or value == 3:
            for_processing.append(key)

    return keywords_final, for_processing
