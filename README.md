# README 
## Introduction 
The travelling salesman problem (TSP) is considered to be an "NP-Hard" problem. This means that the solving the problem
can take exponential time and with larger cities means it is basically impossible to find. 

Ant Colony Optimisation is a Meta-Heuristic Functions whose goal it is to optimise the solution to find the shortest 
path. This program allows a user to choose a variety of parameters. It prints out a graph showing the ant colonies 
heuristic cost of the number of evaluations. The points contain the minimum cost that was obtained from the tour. 

## Prerequisites
Python 3.11 or higher\
The files "brazil58" and "burma14" should be already in the repository they are required for this project to work.
They can also be found on the ELE page.

## Getting started
**Ensure that the files brazil58 and burma14 are in the same directory as main.py for this to work**
Open up a terminal of your choice and locate to directory where package was downloaded. 
Simply run from terminal python main.py or if there are multiple versions of python are installed python3 main.py.

### Navigating the menu 
First the menu makes you a select a city choosing between either Brazil or Burma. Then the menu lets you select the 
following parameters that may affect the behaviour of the ant colony. 
- Number of Ants in a Colony
- Decay Factor
- Alpha 
- Beta

**Number of Ants In a Colony** - describes the number of ants that go on a tour per iteration [10, 50, 100]\
**Decay Factor** - describes the rate that the pheromone will decay after a tour is complete [0.1, 0.5, 0.9] \
**Alpha** - A constant factor that controls how much the pheromone influences the ants transition to the next path [0.5, 1, 3] \
**Beta**  A constant that controls how much the heuristic controls the probability of the ants transition [1, 3, 5]\ 

Once these parameters have been entered the program should run accordingly and print a graph that shows the number of
evaluations and the heuristic function. 