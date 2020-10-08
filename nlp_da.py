import spacy
import nltk
from nltk.stem.porter import *
from spacy.lang.en import English

# NLP PIPELINE - STEP WISE - SANJIT C K S - 18BCE0715
# Load the large English NLP model

nlp = spacy.load('en_core_web_lg')

# The text we want to examine
text = u"""Erode is the seventh largest urban agglomeration in Tamil Nadu. It is also the administrative headquarters of Erode district. Erode has a hilly terrain with undulating and semi-arid climate. River Kaveri flows through the city in and an abundance if limestone is found in its beds. It is located centrally in the south Indian peninsula. It is located at 80km from Coimbatore and 50km form Tiruppur. Being extemely popular for the textile industry, a lot of the cotton spinning, weaving and knitting industries can be found in the region. Historically, it was part of the Kongu Nadu region in the Sangam age and was ruled by the Cheras before being ousted by the Pandyas in 590 CE. It was later a prominent British trading point till independence was gained in 1947."""

# Parse the text with spaCy. This runs the entire pipeline.
doc = nlp(text)

# Segment Sentenses
print("1. Sentence Segmentation:\n")
for sent in doc.sents:
    print(sent)

# Word Tokenization
print("\n\n")
print("2. Word Tokenization:\n")
word_tokens = []
for word in doc:
    word_tokens.append(word.text)
    print(word.text,end=",")

# POS Tagging
print("\n\n")
print("3. POS Tagging:\n")
for word in doc:
    print(word.text,  word.pos_, end=", ")

# Lemmatisation
print("\n\n")
print("4. Lemmatisation:\n")
lemmatised = []
for word in doc:
    lemmatised.append(word.lemma_)
    print(word.text + '  ===>', word.lemma_)

# Remove Stop Words
print("\n\n")
print("5. Remove Stop Words:\n")
from spacy.lang.en.stop_words import STOP_WORDS
filtered_sentence =[] 
for word in word_tokens:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        filtered_sentence.append(word) 
print("Before Removal: ",word_tokens)
print("After Removal: ",filtered_sentence)

# Dependency Parser
print("\n\n")
print("6. Dependency Parser:\n")
from spacy.pipeline import DependencyParser
from spacy import displacy
displacy.serve(doc, style='dep')
# parser = DependencyParser(nlp.vocab)
# processed = parser(doc)
# print(processed)

# NER
print("\n\n")
print("7. Name Entity Recognition:\n")
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")
"""
# Co-reference Resolution
import neuralcoref
print("\n\n")
print("8. Co-reference resolution:\n")
neuralcoref.add_to_pipe(nlp)
doc2 = nlp('Angela lives in Boston. She is quite happy in that city.')
for ent in doc2.ents:
    print(ent._.coref_cluster)
def printMentions(doc):
    print ('\nAll the "mentions" in the given text:')
    for ent in doc.ents:
        print(ent._.coref_cluster)
    for cluster in doc._.coref_clusters:
        print (cluster.mentions)

def printPronounReferences(doc):
    print ('\nPronouns and their references:')
    for token in doc:
        if (token.pos_ == 'PRON' and token._.in_coref):
            for cluster in token._.coref_clusters:
                print (token.text + " => " + cluster.main.text)
if (doc):
        print ("Given text: " + text)
        printMentions(doc)
        printPronounReferences(doc)
"""