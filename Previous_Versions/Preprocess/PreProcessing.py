from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import datetime
from random import shuffle
from operator import sub
import time

# PreProcessing module for Survival Neural Networks
# Purpose of this section is to create
# gapID is amount of hours that user not working
# Action is users playlist in each section
# HourID is the time and day that user comes back
# Assume each month, 30 days
# Assume each year, 365 days
# Distinct artID = 107296


# Function for creating dataframe
# def initialization(addr):


print('start')


# Start preProcessing

def preProcessing():
    inputFile = pd.read_csv(
        '/home/shayan/Desktop/javad/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv',
        sep='\t',
        error_bad_lines=False,
        names=["userid", "timestamp", "artid", "artname", "traid", "traname"])

    # Initializing variables
    iteration = 1
    gapThreshold = 50
    beginSessionIndex = 19098861  # len(inputFile.index) = 19098862
    endSessionIndex = 19098861
    holeSize = 0
    gapSize = 0
    week = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    output = defaultdict(list)
    sessionTracks = []

    # Fetch all action dictionary
    allActionList = list(set(inputFile.loc[:, "artname"]))
    print(len(allActionList))

    allActionDict = {}
    # for item in allActionList:
    #         if isinstance(item, str):
    #             allActionDict.update({item:0})

    # Check progress
    speedCount = 0

    # Iterating through dataframe
    for idx in reversed(inputFile.index):
        speedCount += 1
        sessionTracks = []
        # Show progress
        print(idx)

        # Calculate userID
        userId = inputFile.loc[idx, "userid"]
        newUserNumber = int(userId.split('_')[1])

        # Calculate current Timestamp
        newTimestamp = inputFile.loc[idx, "timestamp"].split("T")
        yyyymmdd = list(map(int, newTimestamp[0].split("-")))
        hhmmss = list(map(int, newTimestamp[1].split(":")[0:2]))
        newTimestamp = yyyymmdd + hhmmss

        # Check if it's first time to loop or not
        if iteration != 1:

            # Check userID change or not
            if (newUserNumber - oldUserNumber) == 0:
                # +1 is for reading dataset from bottom to top 
                endSessionIndex = idx + 1

                # Calculate gaps to find desirable gap
                diff = list(map(sub, newTimestamp, oldTimestamp))
                holeSize = diff[0] * 365 * 24 * 60 + diff[1] * 30 * 24 * 60 + diff[2] * 24 * 60 + diff[3] * 60 + diff[4]
                if (holeSize <= (180 * 24)) and (holeSize >= gapThreshold):
                    gapSize = holeSize

                    # Calculate session actions
                    sessionActionsList = list(inputFile.loc[endSessionIndex:beginSessionIndex, "artname"])
                    print(beginSessionIndex, endSessionIndex, len(sessionActionsList))
                    # sessionActionsDict = allActionDict
                    for item in sessionActionsList:
                        sessionTracks.append(item)

                    # Calculate day from return date of user -- Monday is 0 ... Sunday is 6
                    returnDayNumber = datetime.datetime(yyyymmdd[0], yyyymmdd[1], yyyymmdd[2]).weekday()
                    returnDayName = week[returnDayNumber]
                    returnDayHour = hhmmss[0]

                    # Calculate begin & end session timestamp for future usage
                    beginSessionTimestamp = inputFile.loc[beginSessionIndex, "timestamp"].split("T")
                    beginSession_yyyymmdd = list(map(int, beginSessionTimestamp[0].split("-")))
                    beginSession_hhmmss = list(map(int, beginSessionTimestamp[1].split(":")[0:2]))
                    beginSessionTimestamp = beginSession_yyyymmdd + beginSession_hhmmss

                    endSessionTimestamp = inputFile.loc[endSessionIndex, "timestamp"].split("T")
                    endSession_yyyymmdd = list(map(int, endSessionTimestamp[0].split("-")))
                    endSession_hhmmss = list(map(int, endSessionTimestamp[1].split(":")[0:2]))
                    endSessionTimestamp = endSession_yyyymmdd + endSession_hhmmss

                    beginSessionIndex = endSessionIndex

                    # create output from variables
                    output[newUserNumber].append(
                        [gapSize, returnDayName, returnDayHour, sessionTracks, beginSessionTimestamp,
                         endSessionTimestamp])
                    print(sessionTracks)
                    # time.sleep(3)
            # if user change, update session indexes        
            if (newUserNumber - oldUserNumber) == 1:
                beginSessionIndex = idx
                endSessionIndex = idx

        oldTimestamp = newTimestamp
        oldUserNumber = newUserNumber
        iteration += 1
    return output


a = preProcessing()
print("END")
with open('PreProcessingOutput.pk', 'wb') as fi:
    # dump your data into the file
    pickle.dump(a, fi)
