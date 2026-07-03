# 1 Variables
# 2 Data Types
# 3 Operators
# 4 Conditionals (if, elif, else)
# 5 Functions
# 6 Built-in Functions
# 7 Loops (for, while)
# 8 Exception Handling (try, except, finally)
# 9 Async & Await
#10 Collections (List, Tuple, Dict, Set)
#11 Modules
#12 lambda functions (In Modules.py)
#14 Regular Expressions 
#15 classes -wip









# from typing import Any imports the Any type from Python's typing module.
# Any is used for type hints to indicate:
from typing import Any


#variables --------------------------------------------------------------


variable = 10
print(variable)



#conditionals --------------------------------------------------------------
card = 10
card2 = 20
# same as javascript

# if card > card2:
#     print("card is greater than card2")
# elif card == card2:
#     print("card is equal to card2")
# else:
#     print("card is less than card2")

#input Gets user input. --------------------------------------------------------------
name = input("Enter your name: ")
print("Hello"+name)

#len() Returns the length. --------------------------------------------------------------
print("Length of name is: "+str(len(name)))

#range() Returns a sequence of numbers, starting from 0 by default, and increments by 1 (by default), and stops before a specified number. --------------------------------------------------------------
print("Range of numbers is: "+str(range(10)))

#range() with start, stop, and step --------------------------------------------------------------
print("Range of numbers is: "+str(range(1, 10, 2)))

Employee = {
    "name": "Prakash",
    "age": 25,
    "city": "Chennai"
}

#dict.values() Returns a view object that displays a list of all the values in the dictionary. --------------------------------------------------------------
print("Values of Employee are: "+str(Employee.values()))
#dict.keys() Returns a view object that displays a list of all the keys in the dictionary. --------------------------------------------------------------
print("Keys of Employee are: "+str(Employee.keys()))
#dict.items() Returns a view object that displays a list of all the items (key-value pairs) in the dictionary. --------------------------------------------------------------
print("Items of Employee are: "+str(Employee.items()))
#len() Returns the length of the dictionary. --------------------------------------------------------------
print("Length of Employee is: "+str(len(Employee)))

#isinstance() --------------------------------------------------------------

#Returns True if the object is an instance of the specified type.
# Summary
# Code	Output
# isinstance(10, int)	True
# isinstance("Hi", str)	True
# isinstance(3.14, float)	True
# isinstance(True, bool)	True
# isinstance([1,2], list)	True
# isinstance((1,2), tuple)	True
# isinstance({"a":1}, dict)	True
# isinstance({1,2}, set)	True
# isinstance(10, str)	False
#Use it whenever your code needs to behave differently based on the type of data it receives. It's very common in Python libraries and 
# AI frameworks because functions often accept multiple kinds of input (for example, a list, tuple, or tensor) and need to check the input type before processing it.
print("Employee is a dictionary: "+str(isinstance(Employee, dict)))



# Mutable: Can be changed after creation.
# Immutable: Cannot be changed after creation.

# Summary:
# Data Type	Mutable?
# int	❌ Immutable
# float	❌ Immutable
# complex	❌ Immutable
# bool	❌ Immutable
# str	❌ Immutable
# tuple	❌ Immutable
# frozenset	❌ Immutable
# bytes	❌ Immutable
# NoneType	❌ Immutable
# list	✅ Mutable
# dict	✅ Mutable
# set	✅ Mutable
# bytearray	✅ Mutable

A = 10
B = 3.14
C = 2 + 3j
D = "Hello"
E = True
F = None
G = [1, 2, 3]
H = (1, 2, 3)
I = {"name": "Prakash"}
J = {1, 2, 3}
K = frozenset[int]([1, 2, 3])
L = range(5)
M = b"Hello"
N = bytearray(b"Hello")
O = memoryview[int](bytes(5))

variables = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O]

for item in variables:
    print("Type of "+str(item)+" is: "+str(type(item)))

#functions
def test():
    print("test")

test()

def test(name):
    print(f"Hello {name}")

test("Prakash")

#These are two of Python's most important data types. Understanding them is essential for AI and Python programming.
#(*args) A tuple is an ordered collection of items that cannot be changed after it's created (immutable).
#(**kwargs) is a dictionary stores key-value pairs.

#return args and kwargs
def test(*args ,**kwargs):
    return args, kwargs

(args, kwargs )= test(10, 20,50, name="Prakash", age=20, city="Chennai")
# print(args,"--args")
# print(kwargs,"--kwargs")

# def test(*args, **kwargs):
#     print("args:", args)
#     print("kwargs:", kwargs)

# test(10, name="Sanjay")

#A tuple is an ordered collection of items that cannot be changed after it's created (immutable).
#A dictionary stores key-value pairs.

tuple = (10, 20, 30, 40, 50)
dictionary = {"name": "Prakash", "age": 20, "city": "Chennai"}










#loops  Python has only 2 actual loop statements. --------------------------------------------------------------
#for loop
#while loop

#for loop is used to iterate over a sequence (list, tuple, string, etc.).
#while loop is used to iterate over a block of code until a condition is met.
#Nested while Loop also supported.

# Loop Control Statements
# break--	Exit the loop immediately
# continue--	Skip the current iteration
# pass--	Placeholder; does nothing
# else--	Execute after the loop completes normally

for i in range(5):
    print(f"Hello {i} times")

#while loop
Wcount = 1
while Wcount <= 5:
    print(Wcount)
    Wcount += 1

#break keywords
for i in range(5):
    if i == 3:
        break
    print(i)

#continue keyword
for i in range(5):
    if i == 3:
        continue
    print(i)

#else keyword in for loop
for i in range(5):
    print(i)
else:
    print("For loop ended")

#else keyword in while loop
Wcount = 1
while Wcount <= 5:
    print(Wcount)
    Wcount += 1
else:
    print("While loop ended")

#pass keyword Python requires every if, 
#for, while, def, and class block to contain at least one statement. If you don't have code to write yet, you use pass as a placeholder.
for i in range(5):
    pass #pass is used to do nothing
    print(f"pass keyword: "+str(i))

#pass keyword in while loop
Wcount = 1
while Wcount <= 5:
    pass
    print(Wcount)
    Wcount += 1

#Loop Through a List
fruits = ["Apple", "Banana", "Orange"]

for fruit in fruits:
    print(fruit)

#Loop Through a Tuple
fruits = ("Apple", "Banana", "Orange")

for fruit in fruits:
    print(fruit)

#Loop Through a Dictionary (Keys)
student = {
    "name": "Prakash",
    "age": 25,
    "city": "Chennai"
}

for key in student:
    print(key)

#Loop Through Dictionary Values
student = {
    "name": "Prakash",
    "age": 25,
    "city": "Chennai"
}

for value in student.values():
    print(value)

#Loop Through Dictionary Keys and Values

student = {
    "name": "Prakash",
    "age": 25,
    "city": "Chennai"
}

for key, value in student.items():
    print(key, ":", value)

#Loop with enumerate() (List or Tuple) enumerate is a built-in function that returns an enumerate object with index and value.
fruits = ["Apple", "Banana", "Orange"]

for index, fruit in enumerate[str](fruits):
    print(index, fruit) #index is the index of the item in the list or tuple.

#Loop with enumerate() (Dictionary)
student = {
    "name": "Prakash",
    "age": 25,
    "city": "Chennai"
}

for index, value in enumerate[Any](student.values()):
    print(index, value) #index is the index of the item in the dictionary.  







#Exception Types --------------------------------------------------------------
## Most Common Exceptions
# Exception-	Example
# ZeroDivisionError	10 / 0
# ValueError-	int("abc")
# TypeError-	10 + "20"
# NameError-	print(x)
# IndexError	list[5]
# KeyError-	dict["age"]
# AttributeError-	"abc".append()
# FileNotFoundError-	open("file.txt")
# ModuleNotFoundError-	import xyz
# ImportError-	from math import xyz
# OverflowError-	math.exp(1000)
# MemoryError-	Huge memory allocation
# RecursionError-	Infinite recursion
# StopIteration-	next() after the last item
# KeyboardInterrupt-	Press Ctrl + C
# AssertionError-	assert False
# PermissionError-	Access denied
# NotImplementedError-	raise NotImplementedError()
# RuntimeError-	raise RuntimeError()
# Exception-	Base class for most exceptions


# How many exception types are there?
# Python has dozens of built-in exception classes (well over 60 when including specialized subclasses). However, in everyday development you'll most frequently encounter these 10:
# ZeroDivisionError
# ValueError
# TypeError
# NameError
# IndexError
# KeyError
# AttributeError
# FileNotFoundError
# ModuleNotFoundError
# Exception

# Learning these first will prepare you for the majority of errors you'll see while writing Python programs.


#In Python, it's try → except → finally, not try → catch → finally. --------------------------------------------------------------

#Exception Handling Syntax
try:
    # Code that might cause an error
    print("Try block executed")
except Exception as e:
    # Handle the error
    print("Exception handled: "+str(e))
finally:
    # Always executes
    print("Finally block executed")


#else keyword in try except finally
try:
    print("Try block executed")
except Exception as e:
    print("Exception handled: "+str(e))
else:
    #if no exception occurred, else block is executed
    print("Else block executed")
finally:
    print("Finally block executed")


#Example: Zero Division Error
try:
    print(10/0)
except ZeroDivisionError:
    print("Error: Division by zero")
finally:
    print("Finally block executed")

#Example : Input Validation
try:
    age = int(input("Enter your age: "))
    print("Age:", age)

except ValueError:
    print("Please enter numbers only.")

finally:
    print("Thank you!")




#asyncronous & await programming with asyncio --------------------------------------------------------------

import asyncio

async def mainlog():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

 #if run with print(mainlog()) is this type of error: ERROR <coroutine object mainlog at 0x1063fe140>

#asyncronous & await programming with asyncio
async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(main())



#Modules --------------------------------------------------------------
# .py   → Python source code (you write this)
# .pyc  → Compiled Python bytecode (Python creates this)
# .pyc is reduced time to load the module it will automatically create first time you import the module.

import math #built-in module

print(math.sqrt(16))


#Modules with  wone custom module with alias
import modules #custom module

print(modules.custom_calculator["add"](10, 20))


from modules import custom_calculator as calculator
print(calculator["subtract"](20, 10))





#Regular Expressions --------------------------------------------------------------
import re

print(re.findall(r"\d+", "Age 25"))

text = "The quick brown fox jumps over the lazy dog"
pattern = r"quick"
match = re.search(pattern, text)
print(match.group())










# AI Roadmap (After Python)

# Once you're comfortable with Python, move to AI-specific topics in this order:

# NumPy
# Pandas
# Matplotlib
# Scikit-learn
# Linear Algebra basics
# Calculus basics
# Probability & Statistics
# PyTorch
# Neural Networks
# CNN
# RNN / LSTM
# Attention Mechanism
# Transformers
# GPT Architecture
# Train your own small language model



