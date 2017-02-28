 # -*- coding: utf8 -*-
import requests
import json
import PyICU

def translate2thai(txt):
    english2thai = PyICU.Transliterator.createInstance('English-Thai')
    return english2thai.transliterate(txt)


def isThai(chr):
    cVal = ord(chr)
    if(cVal >= 3584 and cVal <= 3711):
        return True
    return False



def warp(txt):
    #print(txt)
    bd = PyICU.BreakIterator.createWordInstance(PyICU.Locale("th"))
    # bd = PyICU.BreakIterator.createWordInstance(PyICU.Locale("en"))
    bd.setText(txt)
    lastPos = bd.first()
    retTxt = ""
    try:
        while(1):
            currentPos = next(bd)
            # print("currentPos: ",currentPos)
            retTxt += txt[lastPos:currentPos]
            #เฉพาะภาษาไทยเท่านั้น
            if(isThai(txt[currentPos-1])):
                if(currentPos < len(txt)):
                    if(isThai(txt[currentPos])):
                        #คั่นคำที่แบ่ง
                        retTxt += "|"
            lastPos = currentPos
    except StopIteration:
        pass
        #retTxt = retTxt[:-1]
    return retTxt




def fullwrap(txt):
    txt_list = txt.split(' ')
    new_list = []
    for i in txt_list:
        #new_list.extend(wrap(i).split('|||'))
        new_list.extend(wrap(i))
        
    return new_list





r = requests.get("http://localhost:3000/get_data")
r = r.json()
text = ''
print(text + r[0]['message'])
for i in range(len(r)):
	tmp = ''
	if 'message' in r[i]:
		tmp= str(r[i]['message'])
		text += tmp

text = text.replace("\n", " ")
text = translate2thai(text)
text = warp(text)




print(text)