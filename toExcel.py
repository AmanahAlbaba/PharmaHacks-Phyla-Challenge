import pandas as pd

# the purpose of this file is to convert data to percents of total in sample
# date modified: 03/27/2022

data = pd.read_csv("challenge_1_gut_microbiome_data.csv")
# 2 is positive, 1 is negative, 0 is not related

#sets the max range to the number of bacteria samples + 1 in order to read all types of bacteria when summing
maxRange = 1095
#max samples takes the number of microorganism samples and ensures each one is cycled through
maxSamples = 7482
sums = []

#A for loop runs through each row in order to find the sum of all bacteria in each sample
for i, row in data.iterrows():
    #ensures the code stops at the last row by breaking out of the loop
    if (i == maxSamples):
        break
    #sets sum equal to 0 for the new sample
    sum = 0
    #runs through all of the bacteria values in a sample and adds them together inside of a sum variable
    for j in range(1,maxRange):
        sum += (row['Bacteria-'+str(j)])
    #places that sum variable value once complete into an array
    sums.append(sum)

#A for loop runs through each bacteria in a sample changing it from a number to a percent of the total bacteria in the sample 
for i, row in data.iterrows():
    if (i == maxSamples):
        break
    for j in range(1,maxRange):
        #Each bacteria becomes the original bacteria value divided by the sum of bacteria in that sample giving a decimal equivalent to the percentage
        row['Bacteria-'+str(j)] = (row['Bacteria-'+str(j)])/sums[i]
        #the percentage is then placed in the data array in place of the original number
        data.iat[i,j] = row['Bacteria-' + str(j)]

#All of the new percentage values are added to a CSV file 
data.to_csv('withPercents', index = False)
