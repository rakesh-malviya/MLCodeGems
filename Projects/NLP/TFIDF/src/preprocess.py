def onlyascii(char,isSpace='True'):
  if ord(char) in [10,11,12]: 
    return char
  elif ord(char) < 32 or ord(char) > 127:             
    if isSpace: 
      return ' '
    else:
      return ''
  else: 
    return char


def only127(myStr):
  return "".join(map(onlyascii,myStr))


def mapAlnum(char):
  if char.isalnum():
    return char
  else:
    return ' '
  
def onlyAlnum(myStr):
  return "".join(map(mapAlnum,myStr))

 

def getWordsFromFile(filePath):
  with open(filePath) as myFile:
    data = myFile.read()
    data = onlyAlnum(data)
    return data.split()
    
       
