import numpy as np
if __name__ == '__main__':
    a = np.array([1,2,3,4])
    b = np.array([[1,2,3,4],[2,3,4,5]])
    c = (a-b) ** 2
    print(c)
