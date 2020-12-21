import numpy as np



def load_pointcloud(filename):
    with open(filename, "rb") as p:
        for i in range(9):
            p.readline()
        num_points = int(p.readline().split()[1])
        p.readline()
        data = p.read()
        a = np.frombuffer(data[:4*num_points], dtype=np.float32)
    return a.reshape(-1, 3)
