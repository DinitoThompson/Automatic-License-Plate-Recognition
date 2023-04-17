import os
path = "" #insert path to saved folder here

class vehicleSearch:
    def licenseSearch(license): #accepts a plate number
        list = []
        for file in os.listdir(path): #iterate through all files in the specified path
            if (license in file): #if the plate is found in the file name it is added to a list
                list.append(file.replace(".png", ""))

        if not (list):
            print("License plate:", license, "not found")
        else:
            for file in list:
                print(file)


    def dateSearch(startDate, endDate):  
        list = []
        flag = False
        for file in os.listdir(path):
            if (startDate in file): #sets flag to true if start date is found
                flag = True
        
            if(endDate in file): #sets flag back to false if end date is found and add files at end date to list
                flag = False 
                list.append(file.replace(".png", ""))

            if (flag == True):
                list.append(file.replace(".png", ""))

        if not (list):
            print("License plates not found between:", startDate, "and", endDate)
        else:
            for file in list:
                print(file)


#vehicleSearch.licenseSearch("PGJA34")
#vehicleSearch.licenseSearch("111111")
vehicleSearch.dateSearch("4-16-2023","4-17-2023")
#vehicleSearch.dateSearch("1/4/2023")
