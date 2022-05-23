import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import os

if not os.path.exists("graphs"):
    os.makedirs("graphs")

#reading the dataset which include normal and abnormal information
trainingDataset = pd.read_csv('TestingResults.txt', names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, "n/a"])

#reading the dataset for user energy information from csv file
userDataset = pd.read_csv('COMP3217CW2Inputcsv1.csv')

#Dividing the dataset into users
user1 = userDataset.iloc[0:10]
user2 = userDataset.iloc[10:20]
user3 = userDataset.iloc[20:30]
user4 = userDataset.iloc[30:40]
user5 = userDataset.iloc[40:50]
users = [user1, user2, user3, user4, user5]
userVariables = []

#filtering only abnormal data from the dataset
abnormalCurves = trainingDataset.loc[trainingDataset["n/a"] == 1].iloc[:, 0:24]
firstRow = abnormalCurves.iloc[0]

# calculating how many variables are needed
for user in users:
    totalVariables = 0
    for index, task in user.iterrows():
        for x in range(task["Ready Time"], task["Deadline"] + 1):
            totalVariables += 1
    userVariables.append(totalVariables)

# These will remain the same for every calculation
allA = []  # Contains all users 'a' equality constraint matrices.
allB = []  # Contains all users 'b' equality constraint vectors.
allBound = []  # Contains all users bounds

#creating user data from the parameters a(inequality constraints matrix), b(inequality constraints vector), c (coefficients of the linear objective function vector) that will be required for linear programming
for user in range(0, len(users)):
    a = []  #for setting the equality constraints matrix
    b = []  #for setting the equality constraints vector
    a_index = 0
    bounds = []  #to define min max values
    for index, task in users[user].iterrows():  # Loop for appending over each users task
        a_row = []
        for i in range(0, a_index):  # Every variable that has already been defined
            a_row.append(0)
        for x in range(task["Ready Time"], task["Deadline"] + 1):
            a_index += 1
            a_row.append(1)
            bounds.append((0, 1))
        for i in range(a_index, userVariables[user]):  # Variables that still need to be defined
            a_row.append(0)
        a.append(a_row)
        b.append(task["Energy Demand"])

    allA.append(a)
    allB.append(b)
    allBound.append(bounds)

for rowPos in range(0, len(abnormalCurves)):
    row = abnormalCurves.iloc[rowPos]
    total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}

    #running the linear programming function (linprog) for each user
    for user in range(0, len(users)):
        c = []  #for setting the coefficients of the linear objective function vector
        hours = []  # Maps to result array
        for index, task in users[user].iterrows():
            for x in range(task["Ready Time"], task["Deadline"] + 1):
                c.append(row[x])  # Set price
                hours.append(x)

        # Work out minimisation for user
        res = linprog(c, A_eq=allA[user], b_eq=allB[user], bounds=allBound[user])
        resX = res.x
        # Add to total
        for x in range(0, len(hours)):
            total[hours[x]] += resX[x]

    # Graph
    plt.plot(total.keys(), total.values())
    plt.title("Scheduling Results for Abnormal Curve: " + str(rowPos + 1))
    plt.xlabel("Hours")
    plt.ylabel("Energy Usage")
    plt.grid(True)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    plt.savefig("graphs/" + str(rowPos + 1) + ".png")
    plt.clf()
