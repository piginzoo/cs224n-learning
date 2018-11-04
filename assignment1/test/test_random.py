import random
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e","f","g","h","i","j","k","l","m","n","o"]
    return tokens[random.randint(0, 4)], \
           [tokens[random.randint(0, 4)] for i in range(2 * C)]

t1, t2 = getRandomContext(5)
print(t1)
print(t2)