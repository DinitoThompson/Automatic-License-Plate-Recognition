import os

class Blacklist:
    def addToBlacklist(plate):

        blacklistFile = "TextFiles/Blacklist.txt"
        plateFound = False
        with open(blacklistFile, 'r') as f:
            # read all content of a file
            content = f.read()
            # check if string present in a file
            if plate in content:
                print('plate already found cannot be added to blacklist')
                plateFound = True
            else:
                print('plate can be added to blacklist')

        if plateFound == False :
            with open(blacklistFile, 'a') as f:
                f.write(plate)
                f.write("\n")
                print('plate added')








    