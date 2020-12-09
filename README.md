# ECS7002P - FrozenLake
## ECS7002P - Artificial Intelligence in Games Assignment #2
### Due: December 11, 2020

### Team Members:
Luke Abela (200588919), Hasan Emre Erdemoglu (200377106),
Suraj Gehlot (200469881)

### About Project Structure: 
The code is runnable via scripts: _**RunEnvironmentBig.py**_ 
and _**RunEnvironmentSmall.py**_. These scripts will 
output the answers for the given questions using 
Small and Big Lakes. _**.gitignore**_ is used to eliminate 
files that are associated with PyCharm.

The code is divided into multiple folders and scripts to enhance
modularity and readibility.

#### Folder Structure:
1. _**deprecated**_ folder contains previous implementation 
   of the project prior to GitHub migration. it is added here 
   for bookkeeping, not being used in the answers.

2. _**environment**_ folder contains the code which realizes
    the environment. The classes are separated into their 
    respective scripts. Modifications are made to 
    _**FrozenLake.py**_ as requested by the assignment. 
    _**Auxilary_Functions.py**_ is used as a support script
    to construct the grid map in _**FrozenLake.py**_.

3. _**learning_methods**_ folder contains three separate types
    of implementations.  _**TabularModelBasedMethods.py**_ 
    script realizes policy iteration, value iteration, policy
    evaluation and policy improvement respectively. 
   _**TabularModelFreeMethods.py**_ script implement Tabular, 
   Model based Q learning and SARSA algorithms respectively.
   _**LinearWrapper.py**_ script realizes model free 
   and non-tabular implementation of Q-learning and SARSA 
   algorithms, using Linear Function Approximation.

### How to run the main function:

_**RunEnvironmentBig.py**_ and _**RunEnvironmentSmall.py**_
will run every learning method which is realized in 
_**learning_methods**_ folder.

The code is written in such way that the outputs are given 
in the terminal window (if any numeric output is requested).
Otherwise, Questions 2 and 5 uses _**RunEnvironmentBig.py**_ 
and Question 3 uses _**RunEnvironmentSmall.py**_. Questions
1, 3 and 4 are discussion questions with no outputs available.