import os


class Blacklist:
    def addToBlacklist(plate):
        blacklistFile = "Data\Blacklist.txt"
        plateFound = False
        # Checks if plate is in black list
        with open(blacklistFile, 'r') as f:
            # read all content of a file
            content = f.read()
            # check if string present in a file
            if plate in content:
                print('plate already found cannot be added to blacklist')
                plateFound = True
                return False
            else:
                print('plate can be added to blacklist')

        # Adds to black list file if not found
        if plateFound == False:
            with open(blacklistFile, 'a') as f:
                f.write(plate)
                f.write("\n")
                print('plate added')
                return True

    def autoCheckBlacklist(plate):
        blacklistFile = "Data\Blacklist.txt"

        with open(blacklistFile, 'r') as f:
            # read all content of a file
            content = f.read()
            # check if string present in a file
            if plate in content:
                print('plate found')
                return True
                # gui stuff
            else:
                print('plate not found')
                return False
                # gui stuff

    def userCheckBlacklist(plate):
        blacklistFile = "../Data/Blacklist.txt"

        with open(blacklistFile, 'r') as f:
            # read all content of a file
            content = f.read()
            # check if string present in a file
            if plate in content:
                print('plate found')
                return True
                # gui stuff
            else:
                print('plate not found')
                return False
                # gui stuff
