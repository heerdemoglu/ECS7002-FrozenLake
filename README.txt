ECS7002P - FrozenLake
ECS7002P - Artificial Intelligence in Games Assignment 2
Due: December 11, 2020

### This is the text form of README.md. ###

Team Members:
Luke Abela (200588919), Hasan Emre Erdemoglu(200377106), Suraj Gehlot(7502072165)

About Project Structure: 
The code is runnable via scripts: RunEnvironmentBig.py
and RunEnvironmentSmall.py. These scripts will 
output the answers for the given questions using 
Small and Big Lakes. .gitignore is used to eliminate 
files that are associated with PyCharm.

The code is divided into multiple folders and scripts to enhance
modularity and readibility.

Folder Structure:
1. deprecated folder contains previous implementation 
   of the project prior to GitHub migration. it is added here 
   for bookkeeping, not being used in the answers.

2. environment folder contains the code which realizes
    the environment. The classes are separated into their 
    respective scripts. Modifications are made to 
    FrozenLake.py as requested by the assignment. 
    Auxilary_Functions.py is used as a support script
    to construct the grid map in FrozenLake.py.

3. learning_methods folder contains three separate types
    of implementations.  TabularModelBasedMethods.py 
    script realizes policy iteration, value iteration, policy
    evaluation and policy improvement respectively. 
   TabularModelFreeMethods.py script implement Tabular, 
   Model based Q learning and SARSA algorithms respectively.
   LinearWrapper.py script realizes model free 
   and non-tabular implementation of Q-learning and SARSA 
   algorithms, using Linear Function Approximation.

How to run the main function:

RunEnvironmentBig.py and RunEnvironmentSmall.py
will run every learning method which is realized in 
learning_methods folder.

The code is written in such way that the outputs are given 
in the terminal window (if any numeric output is requested).
Otherwise, Questions 2 and 5 uses RunEnvironmentBig.py 
and Question 3 uses RunEnvironmentSmall.py. Questions
1, 3 and 4 are discussion questions with no outputs available.