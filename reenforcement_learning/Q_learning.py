import networkx as nx
import numpy as np 
import pylab as pl

edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
         (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
         (8, 9), (7, 8), (1, 7), (3, 9)]


nodes = 10
graph = nx.Graph(edges)
graph.add_node(nodes)
nx.draw(graph, with_labels=True)
pl.show()

MATRIX_SIZE = 11
M = np.matrix(np.ones(shape =(MATRIX_SIZE, MATRIX_SIZE))) 
M *= -1

for point in edges: 
	print(point) 
	if point[1] == nodes: 
		M[point] = 100
	else: 
		M[point] = 0

	if point[0] == nodes: 
		M[point[::-1]] = 100
	else: 
		M[point[::-1]]= 0
		# reverse of point 

M[nodes, nodes]= 100
print(M) 
# add goal point round trip 
