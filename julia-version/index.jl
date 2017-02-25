using NetworkViz
using LightGraphs
using Colors

g = CompleteGraph(10)
c = Color[parse(Colorant,"#00004d") for i in 1:nv(g)]
n = NodeProperty(c,0.2,0)
e = EdgeProperty("#ff3333",1)
drawGraph(g,node=n,edge=e,z=1) #Draw using a Graph object (3D).

am = full(adjacency_matrix(g))
drawGraph(am,node=n,edge=e,z=0) #Draw using an adjacency matrix (2D).

dgraph = bfs_tree(g,1)
drawGraph(dgraph,z=1) #Draw a Digraph.
