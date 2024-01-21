from gtts import gTTS
from playsound import playsound

def sound(txt):
    language = 'en'
    myobj = gTTS(text=txt, lang=language, slow=False)
    myobj.save("welcome.mp3")
    playsound('welcome.mp3')