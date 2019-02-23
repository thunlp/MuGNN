import random
a = [[i for i in range(j, j+4)] for j in range(2)]


z = list(zip(*a))
random.shuffle(z)
z = list(zip(*z))


print(a)
print(z)