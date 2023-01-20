# -*- coding: utf-8 -*-
"""
@author: Shudhanshu
"""
"""
These are the FIve basic Steps in Genetic Algorithm:
    Step - 1: Initial Population
    Step - 2: Fitness Fucntion eval
    Step - 3: Selection
    Step - 4: Crossover
    Step - 5: Mutation
"""


#%% Imporitng the necessary libs
import numpy as np
import pandas as pd
import time #for calculating the computational time
import random
#%% creating the cities matrix
df = pd.read_excel(r'C:\Users\Lenovo\Documents\Minor-1_Project\tsp_data.xlsx', sheet_name='Data', header=None)
cities = df.values[:,:3]

#%% Implementing the merge sort algorithm (under dev) 
# def mergeSort(pop):
#     if len(pop)>1:
#         mid = len(pop)//2
#         lefthalf = pop[:mid]
#         righthalf = pop[mid:]

#         mergeSort(lefthalf)
#         mergeSort(righthalf)

#         i=0
#         j=0
#         k=0
#         while i < len(lefthalf) and j < len(righthalf):
#             if lefthalf[i].fitness < righthalf[j].fitness:
#                 pop[k].fitness=lefthalf[i].fitness
#                 i=i+1
#             else:
#                 pop[k]=righthalf[j]
#                 j=j+1
#             k=k+1

#         while i < len(lefthalf):
#             pop[k]=lefthalf[i]
#             i=i+1
#             k=k+1

#         while j < len(righthalf):
#             pop[k]=righthalf[j]
#             j=j+1
#             k=k+1


#%% Function to find the distance between two cities
def calculate_distance(cities , solution):
    dist_cal = np.append(solution, [solution[0]], axis=0)#Here we are initially appending the first city from the solution array to the list
    distance = 0 #initially distance is assigned as zero
    next_city_cnt = 0 #this variabke is used for finding the next city
    
    for i in dist_cal: #it will hold first city indexes
        next_city_cnt += 1
        if next_city_cnt < len(dist_cal):
            next_city = dist_cal[next_city_cnt] #Now by passing the next_city_cnt as a paramter, we are pointing to the second city whose distance is to be found w.r.t. first city
            distance += np.sqrt(((cities[next_city,0]-cities[i,0])**2)+((cities[next_city,1]-cities[i,1])**2)) #Calculating the distance between the two cities important note is that the distance calculated is eucledian
            
    return distance #finally we will be able to retrieve the distance betweent the two cities

#%% Function to select the parents from the population to reproduce
def parent_selection(population, number_of_pairs_M):
    current_parents = []
    
    parent_counter = 1
    
    while parent_counter <= 2*number_of_pairs_M: #We will select twice as many parents as the desired number of parent "pairs" i.e. M, so a parent will be selected every time this loop is iterated   
        #we will perform a cumulative test in order to select the parents
        random_float = random.uniform(0,population["fitness"].sum()) #initializing a random float
        cumulative_counter = 0 #counter is set to 0 and this will be assigned to the larger number in the cumulative test
    
        for solution, fitness in population.itertuples(index=False): 
            
            cumulative_counter_copy = cumulative_counter   #this counter will be assigned to the smaller number in the cumulative test   
            cumulative_counter += fitness 
            
            if cumulative_counter_copy <= random_float <= cumulative_counter:   #If the randomly generated float number is in the cumulative range, that respective parent is selected
                
                
                #But before appending it to the solution space, we need to make sure that it is not already present in the solution space or population
                already_present = True
                for parent in current_parents:
                    if parent is solution:
                        already_present = False
                
                        
                if already_present == True: #If the solution is not found, it is appended
                    current_parents.append(solution)        
                    parent_counter += 1
    
    return current_parents #finally we get the current parents for evolution

#%% Function to crossover the Parents and form the child (two point crossover)
def crossover(current_parents, crossover_probability):
    children = [] #crossovered children will be appended to this list        
    for parent_index_holder in range(1, len(current_parents)):
        if random.uniform(0,1) < crossover_probability: #Crossover to parent pairs with the probability specified
            
            parent_1 = current_parents[parent_index_holder-1]
            parent_2 = current_parents[parent_index_holder]
            
            left_bound = random.randint(1, len(current_parents[0])) #seelcting left border or bound for the crossover
            right_bound = random.randint(left_bound, len(current_parents[0])) #seelcting right border or bound for the crossover
            
            child =np.array([]) #children array is created here
            for j in range(left_bound): #The part of the child from the beginning to the left border -> from parent 1
                child = np.append(child, parent_1[j])
            
            for k in range(left_bound,right_bound): #The part of child between left border and right border -> from parent 2
                child = np.append(child, parent_2[k])
                
            for l in range(right_bound, len(parent_1)): #The part of the child from the right border to the end border -> from parent 1
                child = np.append(child, parent_1[l])
            
            #All the parts of the child are appended into the child array as all together form the crossovered gene
            
            
            maps_list = []
            for m in range(left_bound, right_bound):
                maps_list.append([parent_1[m],parent_2[m]])
            
            #child array is appended to the children array
            children.append(child) 
            #https://media.geeksforgeeks.org/wp-content/uploads/20190620121338/twopointCrossover1-2.png -> for reference
    
    return children


#%% Function to mutate the child solutions 
#Here mutation is done for those child's with a probability, by inverting a random part of it                    

def mutate_children(children, mutation_probability):
    children_after_mutation = []
    
    for child in children:
        if random.uniform(0, 1) <= mutation_probability: #checking if it lies below the defined mutation_prob, then we mutate then
            left_bound = random.randint(0,len(child)) #seelcting left border or bound for the mutation
            right_bound = random.randint(left_bound,len(child)) #seelcting left border or bound for the mutation
            child[left_bound:right_bound] = child[left_bound:right_bound][::-1] #Here we are doing nothing but reversing the genes in the child specified from the left bound city/gene to the right bound city/gene
            children_after_mutation.append(child)
        else:
            children_after_mutation.append(child) # if the prob lies above the mutation prob, then we append them without mutation
            
    return children_after_mutation

#%%
def generation_creator(population, mutated_children, cities):

    mutated_children = [] #empty list for the mutated children
    mutated_children_fitnesses = [] #empty list for the mutated children's fitnesses 
    for child in mutated_children:
        mutated_children.append(child) #appending the mutated children
        distance = calculate_distance(cities,child) #calculating the distance
        fitness = 1/distance #here we will take the fitness to be inverse of distance, as if distance between two cities is less than it will have a high fitness value
        mutated_children_fitnesses.append(fitness) #appending the mutated children fitnesses
    
    children = pd.DataFrame(list(zip(mutated_children,mutated_children_fitnesses)),columns=['solution','fitness']) #Here we create a DataFram with the columns as solution and fitness each defining a solution from the population and it's respective fitness
    children.sort_values(by='fitness',axis=0,inplace=True,ascending=False) #Sorting that particular dataframe w.r.t. fitness values

    #Now here we choose the best half children of the population    
    choosen_children_number = round(len(children)/2) #this will result in half of the length and this variable will be acting as a parameter to choosing best half of the children
    choosen_children = children.head(choosen_children_number) # here we assign the choosen_children with only the half of the specified values as we pass choosen_children_number as a parameter

    #Now here we are discrading those solutions which are from teh worst of the population
    population = population.head(len(population)-choosen_children_number)
   
    
    new_population = pd.concat([population, choosen_children]) #by concating the choosne children to the population here, we get the new poopulation or next generation
    new_population.sort_values(by='fitness',axis=0,inplace=True,ascending=False) # again we sort this new population by fitness values

    return new_population
            
#%% Now this is the main function where we will pass the values of all the Initializations
# population = pd.DataFrame()
def main_ga(gen_num, num_of_ind, num_of_pairs, crossover_prob, mutation_prob):
    
    k = 0 #counter for the current generation number
    
    solutions = []
    fitnesses = []
    for i in range(0,num_of_ind):
        solution=np.random.permutation(len(cities))
        solutions.append(solution)
        distance = calculate_distance(cities,solution)
        fitness = 1/distance                 #The fitness value of a solution (i.e. an individual) is calculated with 1/distance
        fitnesses.append(fitness)
    population = pd.DataFrame(list(zip(solutions,fitnesses)),columns=['solution','fitness'])
    population.sort_values(by='fitness',axis=0,inplace=True,ascending=False)  #Individuals in the population are ranked in descending order of fitness values  
    
    print("Initial population: ")  #Initial population is printed
    print(population)
    
    #Genetic search starts (new generations will be produced as many as the defined generation number)
    for i in range(gen_num):
        k+=1
        
        #Initializing the parents in order to move along with the algorithm
        current_parents = parent_selection(population, num_of_pairs)  
                    
        
        #Here we get the crossovered children
        children = crossover(current_parents, crossover_prob)
        
        
        #Child solutions are mutated with pre-defined mutation probability, and if the child's mutation probability lies below the defined mutation probability then only mutation occurs
        mutated_children = mutate_children(children, mutation_prob) 
        
        
        #New Generation is created after replacement from the previous generation
        population = generation_creator(population, mutated_children, cities)
        print("Generation number: ",k )

        print(population)
        
        for solution, fitness in population.itertuples(index=False):
            print("Best solution founded at: ", np.append(solution, [solution[0]], axis=0))
            print("Cost of the founded best solution: ", calculate_distance(cities , solution) )
        
            break
            
#%%  
start_time = time.time() #initializes the starting time

main_ga(100, 100, 25, 0.7, 0.5) #(generation number, number of individuals in a generation, Number of parent "pairs" to be selected in parent selection, crossover probability for a parent pair, mutation probability for a child solution)

comp_time = time.time() - start_time     #The comp_time is nothing but the difference between the stop time fo the algorithm and the start time of the algorithm
print(f"-> Computational Time: {comp_time} seconds")

# %%
