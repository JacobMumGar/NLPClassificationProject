import os
import json
import pandas

path = "/home/jacob/Documents/Uni/Scc453/Project/NLPClassificationProject/data/raw/speeches"
presidentMapping = {
    "Donald Trump" : "Republican",
    "Joseph Biden" : "Democrat",
    "Barack Obama" : "Democrat",
    "George W. Bush" : "Republican",
    "Bill Clinton" : "Democrat",
    "George H. W. Bush" : "Republican",
    "Ronald Reagan" : "Republican",
    "Jimmy Carter" : "Democrat",
    "Gerald Ford" : "Republican",
    "Richard Nixon" : "Republican",
    "Lyndon Johnson" : "Democrat",
    "John F Kennedy" : "Democrat",
    "Dwight Eisenhower" : "Republican",
    "Harry Truman" : "Democrat",
    "Franklin Roosevelt" : "Democrat",
    "Herbert Hoover" : "Republican",
    "Calvin Coolidge" : "Republican",
    "Warren Harding" : "Republican",
    "Woodrow Wilson" : "Democrat",
    "William Taft" : "Republican",
    "Theodore Roosevelt" : "Republican",
    "William McKinley Jr." : "Republican",
    "Grover Cleveland" : "Democrat",
    "Benjamin Harrison" : "Republican",
    "Chester Arthur" : "Republican",
    "James Garfield" : "Republican",
    "Rutherford Hayes" : "Republican",
    "Ulysses Grant" : "Republican",
    "Andrew Johnson" : "Democrat",
    "Abraham Lincoln" : "Republican",
    "James Buchanan" : "Democrat",
    "Franklin Pierce": "Democrat",
    "Millard Fillmore": "Whig",
    "Zachary Taylor": "Whig",
    "James Knox Polk": "Democrat",
    "John Tyler": "Whig",
    "William Harrison": "Whig",
    "Martin Van Buren": "Democrat",
    "Andrew Jackson": "Democrat",
    "John Quincy Adams": "Democratic-Republican",
    "James Monroe": "Democratic-Republican",
    "James Madison": "Democratic-Republican",
    "Thomas Jefferson": "Democratic-Republican",
    "John Adams": "Federalist",
    "George Washington": "No Party"
}

speeches = []

for e in os.scandir(path):
    if e.is_file():
        with open(e.path, "r") as f:
            data = json.load(f)
            speeches.append({
                "president": data["president"],
                "transcript": data["transcript"],
                "date": data["date"]
            })

speechesDf = pandas.DataFrame(speeches)