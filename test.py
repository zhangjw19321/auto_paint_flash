
class Test():
    num = 1
    def __init__(self):
        self.count = 1

t1 = Test()
Test.num = 2
print(Test.num)
print("******")
Test.num = 3 
print(Test.num)