#MertcanÖzçelik

import numpy as np
from matplotlib import pyplot as plt
import time
import json
import random

start_time = time.time()

random.seed(8)
np.random.seed(8)

class QLearning():
    def __init__(self,alpha=0.05,gamma=0.995,epsilon=1,epsilon_decay=0.997,epsilon_min=0.3,episode=250):
        self.big_cost = 999999  # Big cost to compare learning  # Total Demand
        self.alpha = alpha # Learning rate 
        self.gamma = gamma # Discount rate 
        self.epsilon = epsilon  # 100% exploration, 0% exploitation
        self.epsilon_decay = epsilon_decay # Changing from strong exploration to strong exploitation
        self.epsilon_min = epsilon_min # 70% exploration, 30% exploitation
        self.episode = episode
        self.best_route = np.array([], dtype=int)  # Best route vector
        self.route_cost = np.array([], dtype=int)
        self.vehicle_capacity = []
        self.routecost_plotting = True
    
    def readJson(self,path):
        #Matrix
        f = open(path)
        data = json.load(f)
        MatrixData = data["matrix"]
        self.r = np.array(MatrixData)*-1  # transforming Matrixdata to negative Reward Matrix for finding minimal cost 
        self.dimension = self.r.shape[0] 
        #Demand
        json_route = data["jobs"]
        self.demand = [[0],[0],[0]] # Vehicle Demands equal to 0
        for i in range (len(data["jobs"])):
            self.demand.append(json_route[i]['delivery'])
        self.demand = np.array(self.demand)
        self.demand_total = np.sum(self.demand)
        #Capacity
        capacity = data["vehicles"]
        for i in range(len(data["vehicles"])):
            self.vehicle_capacity.append(capacity[i]['capacity'])
        self.vehicle_capacity = [j for i in self.vehicle_capacity for j in i]
        self.q = np.zeros(self.r.shape, dtype=int) # Creating Q States


    def epsilongreedy(self,epsilon):
        if (
            np.random.rand() <= epsilon
        ):  # Exploration, taking a client completely at random
            action = np.random.choice(np.where(self.r[self.state] < 0)[0])
        else:  # Exploitation, takes the customer with the highest value Q
            max_q = np.max(np.take(self.q[self.state], np.where(self.r[self.state] < 0)[0]))
            action = np.random.choice(
                np.intersect1d(
                    np.where((self.q[self.state] == max_q))[0], np.where(self.r[self.state] < 0)[0]
                )
            ) 
        return action

    # Max Future Value
    def max_sa(self,action):
        max_q_sa = np.max(np.take(self.q[action], np.where(self.r[action] < 0)[0]))
        return max_q_sa


    # Q-Learning Formmula
    def update(self,state, action, max_q_sa):
        upd = self.q[state, action] + self.alpha * (self.r[state, action] + self.gamma * max_q_sa - self.q[state, action])
        return upd

    
    
    def StartProgram(self,state=0):
        for i in range(self.episode):
            self.vehicle_size = len(self.vehicle_capacity)
            self.state = state  # Vehicle starting at the depot state = [0]
            remain_cap = max(self.vehicle_capacity)  #  Select Vehicle which has max capacity to Explore Enviroment
            d = self.demand.copy()  # Customer demand vector

            """Update Q Table after Routing in each Episode"""

            while remain_cap >= min(d[d > 0]):

                action = self.epsilongreedy(self.epsilon)  # Decision of which client to visit
                max_q_sa = self.max_sa(action)  # Higher Q value based on the action taken
                self.q[self.state, action] = self.update(self.state, action, max_q_sa)  
                if remain_cap >= d[action]: 
                    remain_cap = remain_cap - d[action]  # Reduced vehicle capacity
                    d[action] = 0  # Client demand becoming 0
                state = action  # In this case the new state is the action taken

            # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            """ Routing test """
            route = np.array([], dtype=int)
            cost_q = 0
            vehicleindexdata = [0,1,2] # Vehicle Index
            for _ in range(self.vehicle_size):
                state_q = random.choice(vehicleindexdata) # Random Vehicle Selection for Starting Route
                route = np.append(route, state_q)
                vehicleindexdata.remove(state_q)
                capac_q = self.vehicle_capacity[state_q] # Set Vehicle Capacity according to assigned vehicle
                exist_nodes = np.ones(self.dimension, dtype=int)
                exist_nodes[[0,1,2]]= 0 #Masking Couriers
                exist_nodes[route] = 0 #Masking Exist Routes
                left_nodes = np.where(exist_nodes == 1)[0]
                demand_ = self.demand.copy()

                while (len(np.intersect1d(np.where(capac_q >= demand_)[0], left_nodes)) > 0):  # As long as customers are available
                    l_max_q = np.where(self.q[state_q]== np.max(np.take(self.q[state_q],np.intersect1d(np.where(capac_q >= demand_)[0], left_nodes))))[0]
                    # Q values equal to the max Q of available nodes
                    if random.random()>0.9:
                        ax = np.random.choice(
                            np.intersect1d(
                                l_max_q, np.intersect1d(np.where(capac_q >= demand_)[0], left_nodes)
                            )
                        )
                        exist_nodes[ax] = 0
                        left_nodes = np.where(exist_nodes == 1)[0]
                    else:
                        ax = np.random.choice(
                            np.intersect1d(
                                l_max_q, np.intersect1d(np.where(capac_q >= demand_)[0], left_nodes)
                            )
                        )
                    # Choose a client that has max Q of those available
                        route = np.append(route, ax)
                        capac_q = capac_q - demand_[ax][0]
                        cost_q = cost_q - self.r[state_q, ax]
                        exist_nodes[ax] = 0
                        left_nodes = np.where(exist_nodes == 1)[0]
                        state_q = ax
                

            min_cost = cost_q
            if min_cost < self.big_cost: #save best route
                self.big_cost = min_cost
                best_route = route
                episode_ = i + 1 #found in episode 

            self.route_cost = np.append(self.route_cost,cost_q)

        """ Show results (Vehicle Routes, Demand Satisfaction, Minimum route cost, Found in Episode) """


        best_route1=best_route.ravel().tolist()


        #Find which vehicle first dispatched
        self.Priority = np.array([], dtype=int)
        self.PriorityDict= []
        for i in best_route1:
            if i == 0 or i ==1 or i==2:
                self.Priority = np.append(self.Priority,i)

        self.PriorityDict = self.Priority.ravel().tolist()
        #find vehicle index in best route 
        idx= best_route1.index(0)
        idx2= best_route1.index(1)
        idx3= best_route1.index(2)


        #masking vehicle points with zero for splitting route
        best_route[idx]= 0
        best_route[idx2]= 0
        best_route[idx3]= 0

        splitted_route = np.split(best_route, np.where(best_route == 0)[0]) #Split Route by 0 Value

        # Route Splitting 
        z = 0
        Demand_satisfaction = 0
        for l in splitted_route:
            if len(l) != 0:
                z=z+1
                demand_sas = np.sum(np.take(self.demand, l)) 
                Demand_satisfaction = Demand_satisfaction + demand_sas
                print("Vehicle",self.Priority[z-1]+1,"Route =>", l[1:]-2, ", Vehicle Capacity Satisfaction rate: %", 100*demand_sas/self.vehicle_capacity[self.Priority[z-1]])
        print("")
        print(
            "Demand satisfaction:",
            Demand_satisfaction,
            "(",
            "{0:.0%}".format(Demand_satisfaction / self.demand_total),
            "of total demand )",
        )
        print("")

        print(
            "Minimum cost obtained:",
            self.big_cost,
            end="\n\n",

        )

        print("Found in episode #", episode_)
        print("--- Execution time %s seconds ---" % (time.time() - start_time)) 


        """ Plotting """

        if self.routecost_plotting is True:
            plt.style.use("ggplot")
            plt.figure(figsize=(16, 8))

            plt.plot(
                self.route_cost, label="Total Delivery Duration", color="darkred", alpha=0.7, lw=2
            )
            plt.xlabel("episodes")
            plt.ylabel("Total Delivery Duration")
            plt.title("Q-Learning Solution")
            plt.legend()
            plt.show()

        "Json Output"

        def gettime(list,i):
            cost_q = 0
            state_q = self.Priority[i] 
            dx=list[i]
            for y in range(len(dx)):
                ax = dx[y]
                cost_q = cost_q - self.r[state_q, ax]
                state_q = ax
            return cost_q

        json_route = []
        for l in splitted_route:
            if len(l) != 0:
                json_route.append(l[1:])

        output = {}
        routes = {}
        for i in range(3):
            routes[i+1] = {"jobs":json_route[self.PriorityDict.index(i)]-2,"delivery_duration": gettime(json_route,self.PriorityDict.index(i))}
        output["total_delivery_duration"] = self.big_cost
        output["routes"] = routes
        with open("Data_File.json", 'w') as file:
              json.dump(output, file, indent=4, sort_keys=True,
              separators=(', ', ': '), ensure_ascii=False,
              cls=MyEncoder)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)




if __name__ == '__main__':
    Start = QLearning()
    Start.readJson("getir_algo_input.json")
    Start.StartProgram()
