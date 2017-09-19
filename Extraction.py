from rake_nltk import Rake
import nltk as nlp


def removePunctuation(document):
    puncs = ", . / # [ ] { } ( ) - _ + = & * $ £ # ' : ; ! ` ~ > <"
    for i in puncs.split():
        document=document.replace(i, "")
    return document.strip()

def pre_process(document):
    document = removePunctuation(document).lower().strip()
    return document


def extractSentences(document):
    sentences = document.split(".")
    sentences = [i.strip() for i in sentences]
    return sentences

def extractWords(document):
    cleaned = removePunctuation(document)
    words = cleaned.split()
    words = [i.strip() for i in words]
    return words


def ie_preprocess(document):
    sentences = extractSentences(document)
    sentences = [extractWords(sentence) for sentence in sentences]
    sentences = [nlp.pos_tag(sentence) for sentence in sentences]
    return sentences

def extractKeywords(document):
    r = Rake()
    keys = r.extract_keywords_from_sentences(document)
    return keys

def extractEntities(document):
    entities=[]
    sentences = ie_preprocess(str(document))
    tagged = [nlp.ne_chunk(sentence) for sentence in sentences]
    ttg=list(tagged)

    for i in ttg:
        subs = []
        listed=list(i)
        for k in i:
            #if k.label() == 'ORGANIZATION' or k.label() == 'PERSON' or k.label() == 'GPE':
            if len(k) == 1:
                subs.append(k)
        if len(subs) > 1:
            entities.append(subs)
    return entities