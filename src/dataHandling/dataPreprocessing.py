import re

def preprocess(speech):
    #Clean the speech field by removing html tags, new line and carriage characters and trailing whitespace
    speech = re.sub(r"<.*?>"," ", speech)
    speech = re.sub("\n", " ", speech)
    speech = re.sub("\r", " ", speech)
    speech = speech.strip()
    return speech