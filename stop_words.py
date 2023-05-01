import nltk
import spacy
from nltk.corpus import stopwords

#nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
stopwords_spacy = nlp.Defaults.stop_words

stop_words_nltk = stopwords.words('english')

print(len(stop_words_nltk))
print(len(stopwords_spacy))
new_set = set()

with open('stop_words.txt', 'r', encoding='utf-8') as file_in:
    for line in file_in:
        line = line.strip()
    line = line.split(',')
    line = set(line)

for word in stop_words_nltk:
    if word in stopwords_spacy and word not in line:
        new_set.add(word)

with open('stop_words_list.txt', 'w', encoding='utf-8') as file_out:
    for word in new_set:
        file_out.write(f'{word.strip()};{word.capitalize().strip()};')

'''
print(new_set)
print(len(new_set))
with open('stop_words.txt', 'w', encoding='utf-8') as file_out:
    file_out.write(f'SpaCy stop words: (len: {len(stopwords_spacy)}) \n')
    file_out.write(str(stopwords_spacy))
    file_out.write(f'\nnltk stop words: (len: {len(stop_words_nltk)}) \n')
    file_out.write(str(stop_words_nltk))
    file_out.write(f'\nstop words in common: (len: {len(new_set)}) \n')
    file_out.write(str(new_set))
'''

