class cublit:
    def __init__(self, c1):
        # c1 list
        self.faces = c1
        # order = ['x', 'y', 'z']
        # for c in c1:
        #     index = order.index(c[1])
        #     self.faces[index] = c[0]

    def print_face(self, face):
        return self.faces[face]

    def rotate(self, axis):
        ax = self.faces.pop(axis)
        self.faces.reverse()
        self.faces.insert(axis, ax)
