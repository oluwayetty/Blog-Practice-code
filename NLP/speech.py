#!/usr/bin/env python3
# Install spacy and speech_recognition
import speech_recognition as sr
import spacy
from spacy.symbols import NOUN
import pyttsx3
from pathlib import Path

#Initiliaze the pyttsx3 text to speech library and spacy library
#Declaration of bag of words
engine = pyttsx3.init()
nlp = spacy.load('en_core_web_lg')
avail_coffee = ['Espresso','cappuccino','Vienna']
words = ['type','kind','have','what']
no_avail = ['nothing']
doc = []
nouns = set()
s = ''
#Welcome message
hi_speech = 'Hello! Welcome to our coffee shop. What kind of coffee do you want?'
engine.say(hi_speech)
engine.runAndWait()

#Take order from customer
def takeOrder():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)
        print("You said " + r.recognize_google(audio))
        example_text = r.recognize_google(audio)
        doc = nlp(example_text)
        s = nlp(example_text)
        # svg = spacy.displacy.render(doc,style='dep')
        # output_path = Path('sentence.svg')
        # output_path.open('w',encoding='utf-8').write(svg)
        # options = {'compact': True, 'bg': '#09a3d5','color': 'white', 'font': 'Trebuchet MS'}
        # spacy.displacy.serve(doc,style='dep', options=options)

    for possible_subject in doc:
        if (possible_subject.pos == NOUN and possible_subject.text.title() in avail_coffee):
            nouns.add(possible_subject.text.title())
            
    print('nouns are:',nouns)
    print('Length nouns',len(nouns))

    if(len(nouns) == 0):
        engine.say('I am sorry,I don\'t understand,Please come again.')
        engine.runAndWait()
        takeOrder()
        confirmOrder()

#Confirm customers order and deliver order
def confirmOrder():
    for drink in nouns:
        #If the coffee the customer ordered is available
        if drink in avail_coffee:
            print('You want ', drink)
            engine.say('Here is your '+ drink)
            engine.say('Enjoy!')
            engine.runAndWait()
            break;
        #If customer doesn't know what type of coffee that is available
        elif drink in words:
            engine.say('We have ')
            for a in avail_coffee:
                engine.say(a)
                engine.runAndWait()
            engine.say('What would you like?')
            engine.runAndWait()
            nouns.pop()
            takeOrder()
        #If customer doesn't want the available coffee
        elif drink in no_avail:
            engine.say('Okay. Thank you for stopping by!')
            engine.runAndWait()
            break
        #If customer request for something that is not available in the coffee shop
        elif drink not in avail_coffee and words and no_avail:
            print('Sorry, we don\'t have '+drink)
            engine.say('Sorry We don\'t have '+drink)
            engine.say('Here is what we have: ')
            for a in avail_coffee:
                engine.say(a)
                engine.runAndWait()
            engine.say('What would you like?')
            engine.runAndWait()
            nouns.pop()
            takeOrder()

#Program Execution
try:
    takeOrder()
    confirmOrder()
    svg = spacy.displacy.render(s,style='dep')
    output_path = Path('sentence.svg')
    output_path.open('w',encoding='utf-8').write(svg)
except sr.UnknownValueError:
    engine.say('I am sorry,I don\'t understand,Please come again.')
    engine.runAndWait()
    takeOrder()
    confirmOrder()
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))
    engine.say('I am sorry,I don\'t understand,Please come again.')
    engine.runAndWait()
    takeOrder()
    confirmOrder()
