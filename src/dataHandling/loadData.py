import os
import json
import pandas

def loadData():
    path = "/home/jacob/Documents/Uni/Scc453/Project/NLPClassificationProject/data/raw/speeches"

    #dictionary of all presidents and party affiliation for mapping
    presidentMapping = {
        "Donald Trump" : "Republican",#
        "Joe Biden" : "Democrat",#
        "Barack Obama" : "Democrat",#
        "George W. Bush" : "Republican",#
        "Bill Clinton" : "Democrat",#
        "George H. W. Bush" : "Republican",#
        "Ronald Reagan" : "Republican",#
        "Jimmy Carter" : "Democrat",#
        "Gerald Ford" : "Republican",#
        "Richard M. Nixon" : "Republican",#
        "Lyndon B. Johnson" : "Democrat",#
        "John F. Kennedy" : "Democrat",#
        "Dwight D. Eisenhower" : "Republican",#
        "Harry S. Truman" : "Democrat",#
        "Franklin D. Roosevelt" : "Democrat",#
        "Herbert Hoover" : "Republican",#
        "Calvin Coolidge" : "Republican",#
        "Warren G. Harding" : "Republican",#
        "Woodrow Wilson" : "Democrat",#
        "William Taft" : "Republican",#
        "Theodore Roosevelt" : "Republican",#
        "William McKinley" : "Republican",#
        "Grover Cleveland" : "Democrat",#
        "Benjamin Harrison" : "Republican",#
        "Chester A. Arthur" : "Republican",#
        "James A. Garfield" : "Republican",#
        "Rutherford B. Hayes" : "Republican",#
        "Ulysses S. Grant" : "Republican",#
        "Andrew Johnson" : "Democrat",#
        "Abraham Lincoln" : "Republican",#
        "James Buchanan" : "Democrat",#
        "Franklin Pierce": "Democrat",#
        "Millard Fillmore": "Whig",#
        "Zachary Taylor": "Whig",#
        "James K. Polk": "Democrat",#
        "John Tyler": "Whig",#
        "William Harrison": "Whig",#
        "Martin Van Buren": "Democrat",#
        "Andrew Jackson": "Democrat",#
        "John Quincy Adams": "Democratic-Republican",#
        "James Monroe": "Democratic-Republican",#
        "James Madison": "Democratic-Republican",#
        "Thomas Jefferson": "Democratic-Republican",#
        "John Adams": "Federalist",#
        "George Washington": "No Party"#
    }

    #loop through all files in corpus folder and add speech to speeches list
    speeches = []
    for e in os.scandir(path):
        if e.is_file():
            with open(e.path, "r") as f:
                data = json.load(f)
                speeches.append({
                    "president": data["president"],
                    "transcript": data["transcript"],
                    "year": int((data["date"])[:4])
                })

    #turn speeches list into a pandas dataframe, generating party column from mapping
    #get rid of rows where party isnt democrat or republican, and only keep speeches from after 1945
    speechesDf = pandas.DataFrame(speeches)
    speechesDf["party"] = speechesDf["president"].map(presidentMapping)
    speechesDf = speechesDf[speechesDf["party"].isin(["Democrat","Republican"])]
    speechesDf = speechesDf[speechesDf["year"]>1945]

    return speechesDf