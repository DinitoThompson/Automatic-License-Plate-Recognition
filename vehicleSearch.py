path = "test.txt" #text file path
def licenseSearch(license):
    file = open(path,"r")
    list = []
    for x in file:
        if(license in x):
            list.append(x.replace("\n",""))
    file.close
   
    if not(list):
        print("License plate:",license,"not found")
    else:
        for x in list:
            print(x)

def dateSearch(date): #not sure if I should include time with the date
    file = open(path,"r")
    list = []
    for x in file:
        if(date in x):
            list.append(x.replace("\n",""))
    file.close
    
    if not(list):
        print("License plates not found on:",date)
    else:
        for x in list:
            print(x)

    
#licenseSearch("PGJA34")
#licenseSearch("111111")
#dateSearch("1/2/2023")
#dateSearch("1/4/2023")
