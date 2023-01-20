#%% Importing the necessary libs and modules
from Genetic_Algorithm_2 import *
from Simulated_Annealing import *
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
import folium 

#%%
st.title("Minor Project: The Metaheuristics")
# m = folium.Map(location=[30.350, 78.296], zoom_start=12)
algos = st.sidebar.selectbox("Select Algorithm", ("Genetic Algorithm", "Simulated Annealing"))

uploaded_file = st.sidebar.file_uploader("Please upload  your TSP dataset")

#%%
df = pd.read_excel(r'C:\Users\Lenovo\Documents\Minor-1_Project\tsp_data.xlsx', sheet_name='Data', header=None)
cities = df.values[:,:3]
st.header("Default Dataset for TSP:")
st.table(df)
# st.table(cities)

#%%
# my_bar = st.sidebar.progress(0)
# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)
# st.spinner()
# with st.spinner(text='In progress'):
#     time.sleep(5)
#     st.success('Done')
# st.balloons()


#%%
def get_algos(algos):
    if algos == "Genetic Algorithm":
        gen_num = int(st.sidebar.number_input("Enter the number of generations to execute"))
        ind_num = int(st.sidebar.number_input("Enter the number of individuals in one generation"))
        par_pairs = int(st.sidebar.number_input("Enter the number of parent pairs to reproduce"))
        cross_prob = st.sidebar.number_input("Enter the crossover probability")
        mut_prob = st.sidebar.number_input("Enter the mutation probability")
        
        k = 0 #counter for the current generation number
    
        solutions = []
        fitnesses = []
        for i in range(0,ind_num):
            solution=np.random.permutation(len(cities))
            solutions.append(solution)
            distance = calculate_distance(cities,solution)
            fitness = 1/distance                 #The fitness value of a solution (i.e. an individual) is calculated with 1/distance
            fitnesses.append(fitness)
        population = pd.DataFrame(list(zip(solutions,fitnesses)),columns=['solution','fitness'])
        population.sort_values(by='fitness',axis=0,inplace=True,ascending=False)  #Individuals in the population are ranked in descending order of fitness values  
        
        st.header("Initial population: ")  #Initial population is printed
        st.write(population)
        
        #Genetic search starts (new generations will be produced as many as the defined generation number)
        for i in range(gen_num):
            k+=1
            
            #Initializing the parents in order to move along with the algorithm
            current_parents = parent_selection(population, par_pairs)  
                        
            
            #Here we get the crossovered children
            children = crossover(current_parents, cross_prob)
            
            
            #Child solutions are mutated with pre-defined mutation probability, and if the child's mutation probability lies below the defined mutation probability then only mutation occurs
            mutated_children = mutate_children(children, mut_prob) 
            
            
            #New Generation is created after replacement from the previous generation
            population = generation_creator(population, mutated_children, cities)
            st.write("Generation number: ",k )

            st.write(population)
            
            for solution, fitness in population.itertuples(index=False):
                st.write("Best solution founded at: ", np.append(solution, [solution[0]], axis=0))
                st.write("Cost of the founded best solution: ", calculate_distance(cities , solution))
                break 
    
    if algos == "Simulated Annealing":
        Tmax = int(st.sidebar.number_input("Enter the upper bound of Temperature"))
        Tmin = int(st.sidebar.number_input("Enter the lower bound of Temperature"))
        cool_rate = int(st.sidebar.number_input("Enter the cooling rate for Annealing"))
        max_iter = int(st.sidebar.number_input("Enter the number of iterations"))
        
        start_time = time.time()
        #here we are keeping the temperature between 1-100
    
        Tc = Tmax #initializing the current Temperature to the Maximum Temperature)
        _iter = 1


        #  Step - 2: Initial Solution, X

        X = np.zeros([1, Num_Nodes+1], dtype=int) #Here we are padding the solution space with zeroes
        X[0, 1:-1] = np.random.permutation(Num_Nodes-1)+1 #Here we are randomly permutatiing the values for the initial solution

        # Step - 3: Fitness Function, F(X)

        #Now, an important thing to note is that there is an inverse proportion relation betweeen the distance and the fitness value.
        #If the Distance between two cities is very short, that means it's fitness value is very large and that particular path is more likely to be the optimal solution

        #So Fitness function here is:
                                    # min F = 1 / Σ(i->N-1) d(i, i+1)
                                        #where F is the Fitness, d(i, i+1) is the distance between the ith and (i+1)th city

        X_fitness = fitness_SA(X, dist_matrix)


        #Step - 4: Looping through the Algorithm

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
            
            
        # print(comp_time_sa)
        #The comp_time is nothing but the difference between the stop time fo the algorithm and the start time of the algorithm
        
        # st.write(f"-> Computational Time: seconds")
        fig = plt.figure(figsize=(6,6))
        fig_1 = fig.add_axes([0,0,1,1])
        fig_1.scatter(x_cord[1:], y_cord[1:], c='green')
        fig_1.plot(x_cord[1:], y_cord[1:], c='blue', marker='s')

        for i in range(0, Num_Nodes):
            fig_1.plot([x_cord[X[0, i]], x_cord[X[0, i+1]]], [y_cord[X[0, i]],y_cord[X[0, i+1]]], c = 'red', zorder=4) 
            st.write("Destination to: ", i+1, '=', X[0, i])
            
        fig_1.set_xlabel('X-Cordinate of the Cities')
        fig_1.set_ylabel('Y-Cordinate of the Cities')
        plt.savefig("Sa_plot.png")
        st.pyplot(fig)
        
        
        
 #%%       
        
get_algos(algos)
        
        
        
# %%
