import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc1 = nlp("a recent study that used a national sample found that more than half of the parents of school-aged children oppose school personnel carrying firearms")
doc2 = nlp("Firearm violence remains a significant problem in the US (with 2787 adolescents killed in 2015). However, the research on school firearm violence prevention practices and policies is scant. Parents are major stakeholders in relation to firearm violence by youths and school safety in general. The purpose of this study was to examine what parents thought schools should be doing to reduce the risk of firearm violence in schools. A valid and reliable questionnaire was mailed to a national random sample of 600 parents who had at least one child enrolled in a public secondary school (response rate=47%). Parents perceived inadequate parental monitoring/rearing practices (73%), peer harassment and/or bullying (58%), inadequate mental health care services for youth (54%), and easy access to guns (51%) as major causes of firearm violence in schools. The school policies perceived to be most effective in reducing firearm violence were installing an alert system in schools (70%), working with law enforcement to design an emergency response plan (70%), creating a comprehensive security plan (68%), requiring criminal background checks for all school personnel prior to hiring (67%), and implementing an anonymous system for students to report peer concerns regarding potential violence (67%). Parents seem to have a limited grasp of potentially effective interventions to reduce firearm violence. ")
doc3 = nlp("Autonomous cars shift insurance liability toward manufacturers. Autonomous cars shift insurance liability toward manufacturers.")

"""new_phrase = ''
for token in doc3:
    if (token.is_stop):
        continue
    new_phrase += token.text + ' '
print(new_phrase)"""

stopwords = nlp.Defaults.stop_words
nlp.Defaults.stop_words -= {"not", "never"}
print(stopwords)
print(len(stopwords))