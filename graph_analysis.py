import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


n_samples = 15

x_data_set = np.random.uniform(1, 50, n_samples)
y_data_set = np.random.uniform(1, 50, n_samples)

def make_graph(x_array, y_array, eps):
    m = x_array.shape[0]
    n = y_array.shape[0]
    
    if m != n:
        print("Invalid (x_ or y_)array size")
        return None

    adj_matrix = np.zeros([n, n], dtype='int64') 
    for i in range(35):
        for j in range(35):
            sq_eucli = (x_array[i] - x_array[j])**2 + (y_array[i] - x_array[j])**2
            if sq_eucli <= eps**2:
                adj_matrix[i][j] = adj_matrix[j][i] = int(1)
    
    return adj_matrix


np.savetxt("matrix.txt", make_graph(x_data_set, y_data_set, 13))



