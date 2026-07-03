def mocFunction(*args,**kwargs):
    return args,kwargs

abc =mocFunction(10,20,30,name="Prakash",age=20,city="Chennai",numbers=[1,2,3,4,5])
(args,kwargs)=abc

for i in kwargs.items():
   try:
    print(i[0],i[1])
   except KeyError:
    print("Key not found")
    break
   finally:
    print("Thank you!")
