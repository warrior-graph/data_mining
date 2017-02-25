import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys



n_samples = int(sys.argv[1])
eps = float(sys.argv[2])
x_data_set = np.random.uniform(1, 50, n_samples)
y_data_set = np.random.uniform(1, 50, n_samples)

def make_graph(x_array, y_array, eps):
    m = x_array.shape[0]
    n = y_array.shape[0]

    if m != n:
        print("Invalid (x_ or y_)array size")
        return None
    
    G = nx.Graph()

    adj_matrix = np.zeros([n, n], dtype='int32')
    for i in range(n):
        for j in range(i + 1, n):
            sq_eucli = (x_array[i] - x_array[j])**2 + (y_array[i] - y_array[j])**2
            if sq_eucli <= eps**2:
                adj_matrix[j][i] = adj_matrix[i][j] = 1
                G.add_edges_from([(i, j), (j, i)])
            else:
                G.add_nodes_from([i, j])
    return adj_matrix, G

adj_g, G = make_graph(x_data_set, y_data_set, eps)


print(adj_g)

n = adj_g.shape[0]

fig = plt.figure()

ax = fig.add_subplot(111)

plt.plot(x_data_set, y_data_set, 'ko')

for i, xy in enumerate(zip(x_data_set, y_data_set)):
    ax.annotate('(%d)' % i, xy=xy)



for i in range(n):
    for j in range(i + 1, n):
        if adj_g[i][j] == 1:
            plt.plot([x_data_set[i], x_data_set[j]],[y_data_set[i], y_data_set[j]],
            ls='dashed', color='green', marker='o', mfc='red')


plt.xlim(0, 60)
plt.ylim(0, 60)

plt.figure()
nx.draw(G)
plt.show()

