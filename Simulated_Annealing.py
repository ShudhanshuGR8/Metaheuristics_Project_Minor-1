# -*- coding: utf-8 -*-
"""
@author: Shudhanshu
"""
"""
annealing is the process of repeatedly heating metal to a certain point and then immediately cooling it. This makes the metal achieve new state with new properties. 
#Similiarly simulated annealing is simulation of the above mentioned annealing process. We define a temperature variable(initially set to high) and then we allow it to cool as the algorithm runs.
"""


#Implementation
#%%
#Importing the necessary libs
import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import time

#Importing the data of city and travel for the TSP
#%%

df = pd.read_excel(r'C:\Users\Lenovo\Documents\Minor-1_Project\tsp_data.xlsx', sheet_name='Data', header=None)
data = df.values

df = pd.read_excel(r'C:\Users\Lenovo\Documents\Minor-1_Project\tsp_data.xlsx', sheet_name='Distance_Matrix', header=None)
dist_matrix = df.values

#%%
Num_Nodes = len(data[:,1])
x_cord = np.array(data[:,1])
y_cord = np.array(data[:,2])

#%%
#Simulated Annealing Algorithm
#We Will follow the below steps in order to implement the SA Algorithm

#Step - 1: Defining  the Parameters

#Step - 2: Initializing the Solution X (random initial solution is chosen here)

#Step - 3: Calculating the Fitness F(X)

#Step - 4: Looping across the algorithm untill -> 
                                            # Tc (current Temperature)  < Tmin (minimum Temperature specified) in order to save the time of simulations, as with low temperatures the optimization differences are almost not visible.
#Step - 5: Output 

#%%
#Defining some functions we will need first

#fitness_calculation function F(X)
def fitness_SA(sol, dist_matrix):
    
    dist = np.zeros([1,1])
    for i in range(0, len(sol[0,:]) - 1):
        depart_node = int(sol[0][i])
        next_node = int(sol[0][i+1])
        dist += dist_matrix[depart_node][next_node]
        
        fitness = 1/dist #inversely proportional relation
        
    return fitness
#Neighbor functions
#swap search function
def swap_Search(sol): #as the name suggest in this we swap the nodes and create the new sol
    #randomly selecting 2 nodes
    swap_nodes = rand.sample(range(1, len(sol[0,:])-1), 2)
    swap_1 = swap_nodes[0]
    swap_2 = swap_nodes[1]
    
    #now we are swaping the values
    new_sol = np.array(sol)
    temp = np.array(new_sol[0][swap_1])
    new_sol[0][swap_1] = new_sol[0][swap_2]
    new_sol[0][swap_2] = temp
    return new_sol

#insertion search function
def insertion_Search(sol): #in this , we insert one node right behind the other to create the new sol
    
    insertion_node = rand.sample(range(len(sol[0,:])-1), 2)
    while(insertion_node[0] - insertion_node[1] == 1 or insertion_node[0] - insertion_node[1] == 0):
        insertion_node = rand.sample(range(len(sol[0,:])-1), 2)
    
    insert_1 = insertion_node[0]
    insert_2 = insertion_node[1] #-> target node (this is the node whose right behind we will insert the node 1)
    
    new_sol = np.array(sol)
    if insert_1 < insert_2:
        temp = np.array(new_sol)
        new_sol[0][insert_2-1] = new_sol[0][insert_1] #Moving insert1 node right behind the insert2 node hence the insert2-1
        if insert_1 == (insert_2 - 2):
             new_sol[0][insert_1] = temp[0][insert_1+1]
        else:
            new_sol[0][insert_1:insert_2-1] = np.array(temp[0][insert_1+1:insert_2])
    elif insert_1 > insert_2:
        temp = np.array(new_sol)
        new_sol[0][insert_1-1] = new_sol[0][insert_2] #Moving insert1 node right behind the insert2 node hence the insert2-1
        if insert_1 == (insert_2 - 2):
             new_sol[0][insert_2] = temp[0][insert_2+1]
        else:
            new_sol[0][insert_2:insert_1-1] = np.array(temp[0][insert_2+1:insert_1])
    
    return new_sol
             
#2-opt search function
def two_opt_Search(sol):
    edge_nodes = rand.sample(range(len(sol[0,:])-2), 2)
    while abs(edge_nodes[0] - edge_nodes[1]) == 1:
        edge_nodes = rand.sample(range(len(sol[0,:])-2), 2)
    
    edge_A = edge_nodes[0]
    edge_B = edge_nodes[0] + 1
    edge_C = edge_nodes[1]
    edge_D = edge_nodes[1] + 1
    
    #using the 2-opt rule: a,b,c,d -> a,c,b,d
    new_sol = np.array(sol)
    node_A = new_sol[0][edge_A]
    node_B = new_sol[0][edge_C] #a,b,c,d -> a,c,b,d
    node_C = new_sol[0][edge_B] #a,b,c,d -> a,c,b,d
    node_D = new_sol[0][edge_D]
    
    new_sol[0][edge_A] = node_A
    new_sol[0][edge_B] = node_B
    new_sol[0][edge_C] = node_C
    new_sol[0][edge_D] = node_D
    
    return new_sol
    
    
#%%  Step - 1: Defining Paramters
# Tmax = 100
# Tmin = 1
# cool_rate = 0.99
# max_iter = 1700

def sa_main(Tmax, Tmin, cool_rate, max_iter):
    start_time = time.time()
    #here we are keeping the temperature between 1-100
   
    Tc = Tmax #initializing the current Temperature to the Maximum Temperature)
    _iter = 1 #current iteration number


    # Step - 2: Initial Solution, X

    X = np.zeros([1, Num_Nodes+1], dtype=int) #Here we are padding the solution space with zeroes
    X[0, 1:-1] = np.random.permutation(Num_Nodes-1)+1 #Here we are randomly permutatiing the values for the initial solution

    # Step - 3: Fitness Function, F(X)

    #Now, an important thing to note is that there is an inverse proportion relation betweeen the distance and the fitness value.
    #If the Distance between two cities is very short, that means it's fitness value is very large and that particular path is more likely to be the optimal solution

    #So Fitness function here is:
                                # min F = 1 / Σ(i->N-1) d(i, i+1)
                                    #where F is the Fitness, d(i, i+1) is the distance between the ith and (i+1)th city

    X_fitness = fitness_SA(X, dist_matrix)


    # Step - 4: Looping through the Algorithm

    while Tc > Tmin:

        #Step - 4.1: Randomizing r
        r = np.random.random()
        
        #Step - 4.2: Creating a new solution 
        if r <= 0.33:
            X_new = swap_Search(X)
            
        elif (r > 0.33) and (r <= 0.66):
            X_new = insertion_Search(X)
            
        else:
            X_new = two_opt_Search(X)
        
        #Step - 4.3: Fitness eval of the new sol
        fitness_new = fitness_SA(X_new, dist_matrix)
        
        #Step - 4.4: Decision step (if we wanna keep the Xnew(new_sol))
        if fitness_new > X_fitness:
            X = np.array(X_new)
            X_fitness = np.array(fitness_new)

        
        #Step - 4.5: Updating the parameters (temp and iter)
        _iter += 1
        if _iter == max_iter:
            Tc *= cool_rate
            _iter = 1
        
        
    comp_time_sa = time.time() - start_time     #The comp_time is nothing but the difference between the stop time fo the algorithm and the start time of the algorithm
    print(f"-> Computational Time: {comp_time_sa} seconds")
    fig = plt.figure(figsize=(6,6))
    fig_1 = fig.add_axes([0,0,1,1])
    fig_1.scatter(x_cord[1:], y_cord[1:], c='green')
    fig_1.plot(x_cord[1:], y_cord[1:], c='blue', marker='s')

    for i in range(0, Num_Nodes):
        fig_1.plot([x_cord[X[0, i]], x_cord[X[0, i+1]]], [y_cord[X[0, i]], y_cord[X[0, i+1]]], c = 'red', zorder=4) 
        print("Destination to: ", i+1, '=', X[0, i])

    fig_1.set_xlabel('X-Cordinate')
    fig_1.set_ylabel('Y-Cordinate')
    plt.show()
    plt.savefig('sa_plot.png')

#%% Calculating the computing time

#%% Output:
# sa_main(100, 1, 0.99, 1000)


# %%
