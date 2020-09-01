# Reinforcement Learning Notes and Solution to Exercises
### Using the [Reinforcement Learning: An Introduction, Second Edition by Sutton and Barto](http://www.incompleteideas.net/book/the-book-2nd.html). Code is written in C++ and Python.

* There is a `.pdf` in each chapter which contains the general notes of the chapter, as well as attempts to answer the exercises provided in the book. 
* For coding exercises, the outputs are usually stored in a `csvs/` subdirectory as `.csv` files. Thus, running the python code should immediately yield the relevant graphs. The C++ files should also work out of the box, and variants of the code can be obtained by tweaking the parameters to the classes.
* Note that Chapter 11 is missing. This will (hopefully) be added by the end of 2020, as I need to get on with preparing for the new academic year.


### Mini Project
* Contains policy gradient algorithms form the textbook, including vanilla policy gradient, actor-critic and actor-critic with eligiblity traces. Note that the algorithms here only deal with a discrete action space. (An attempt was made to create an actor-critic agent (A2C) for a continuous action space, but it never succeeded in OpenAI's continuous action environments)
* Tested with OpenAI Gym environment, albeit only with the basic control environments since most environments had a continuous action space.
* Includes a small directory that plays around with stock buying/selling with an actor-critic agent. Spoiler alert: it is currently too simple to perform better than passive investing since it only uses the high, low, open, close and volume as the observation space.   