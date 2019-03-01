

class a(object):

    def __init__(self):
        self.box = [1,2,3,4,5]
        self.box2 = [312, 123]


def add(box):
    box.box.append(100)

aa = a()
add(aa)
print(aa.box)
