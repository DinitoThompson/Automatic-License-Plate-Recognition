import os
from datetimerange import DateTimeRange #pip install DateTimeRange
path = "Data\Saved_Plates"  # insert path to saved folder here


class vehicleSearch:
    def licenseSearch(license):  # accepts a plate number
        list = []
        # iterate through all files in the specified path
        for file in os.listdir(path):
            if (license in file):  # if the plate is found in the file name it is added to a list
                list.append(file)

        if not (list):
            print("License plate:", license, "not found")
        else:
            for file in list:
                print(file)

        return list

    def dateSearch(startDate, endDate):
        list = []

        for file in os.listdir(path):
            range = DateTimeRange(startDate, endDate)#Creates the date range
            range.start_time_format = "%m-%d-%Y"#Sets the date format to mm-dd-yyyy
            range.end_time_format = "%m-%d-%Y"
            if(file[0:10] in range):#if the date section of the file name is in range
                list.append(file)#add to list

        if not (list):
            print("License plates not found between:",
                  startDate, "and", endDate)
        else:
            for file in list:
                print(file)

        return list


# vehicleSearch.licenseSearch("PGJA34")
# vehicleSearch.licenseSearch("111111")
#vehicleSearch.dateSearch("4-16-2023", "4-17-2023")
# vehicleSearch.dateSearch("1/4/2023")
