# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import math
import os
import pickle

from scipy.spatial import distance

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
	self.index = 0
	self.my_nodes = {}

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        # raise NotImplementedError
	if self.queue:
	    priority, index, key = heapq.heappop(self.queue)
	    del self.my_nodes[index]
	    return (priority, key)
	raise Exception('Queue is empty!')	

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
	for element in self.queue:
	    if node_id == element[2]:
		del self.my_nodes[element[1]]
		self.queue.remove(element)
	#this_node = self.my_nodes.pop[node_id]
	#self.queue.remove(this_node)
	
        # raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        # raise NotImplementedError
	# priority, count, node_id
	self.index += 1
	priority, key = node
	node_list = [priority, self.index, key]
	self.my_nodes[self.index] = node_list
	heapq.heappush(self.queue, node_list)


    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _,_,n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]

def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
    best_path = []
    counter = 0
    explored = {}
    parent_dict = {}
    parent_node = None
    children_dict = {}
    
    if start == goal:
	return best_path    
    # initialize a frontier object
    frontier = PriorityQueue()
    # add start node to frontier
    frontier.append((counter, start))
    parent_dict[start] = parent_node
    # explore graph with a while loop
    while True:
	if frontier.queue == []:
	    return []
	counter += 1
	node = frontier.pop()
	parent_node = node[1]
	children_dict = graph[parent_node]
	# add current parent node to explored list
	explored[parent_node] = node
	for child in children_dict:
	    if child not in parent_dict:
		parent_dict[child] = parent_node
	    if child == goal:
		while parent_node:
		    best_path.append(child)
		    child = parent_node
		    parent_node = parent_dict[child]
		best_path.append(child)
		return best_path[::-1]
	    if child not in explored:
		frontier.append((counter, child))	


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if start == goal:
	return []
    best_path = []
    parent_node = None
    explored = {}
    # key: current node, value: (parent node, cost)
    # cost is the total cost from start to the current node
    node_cost = {} # this is the link between child and parent, as well as storing cost
    # initialization
    frontier = PriorityQueue()
    # frontier: (cost, parent node), less cost pop first
    frontier.append((0,start))
    node_cost[start] = (0,None)
    # iterative explore graph
    while True:
	if len(frontier.queue) == 0:
	    return []
	node = frontier.pop()
	if node[1] == goal:
	    tmp_node = node[1]
	    # go back to the start
	    parent_node = node_cost[tmp_node][1]
	    while parent_node:
		best_path.append(tmp_node)
		tmp_node = parent_node
		parent_node = node_cost[tmp_node][1]
	    best_path.append(tmp_node)
	    return best_path[::-1]
	
	elif node[1] in explored:
	    continue
	cur_cost,cur_node = node
	children_dict = graph[cur_node]
	explored[cur_node] = node
	for child in children_dict:
	    # cost from start to this child node
	    child_cost = children_dict[child]['weight'] + cur_cost
	    if child not in node_cost or child_cost < node_cost[child][0]:
		node_cost[child] = (child_cost,cur_node)		
	    if child not in explored:
		frontier.append((child_cost, child))

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if v == goal:
	return 0
    v_pos = graph.node[v]['pos']
    goal_pos = graph.node[goal]['pos']
    return distance.euclidean(v_pos, goal_pos)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if start == goal:
	return []
    explored = {}
    node_cost = {}
    best_path = []
    parent_node = None
    frontier = PriorityQueue()
    frontier.append((0,start))
    # key: current node, value:(cost from start to current node, parent node)
    node_cost[start] = (0, None)
    while True:
	if len(frontier.queue) == 0:
	    return []
	node = frontier.pop()
	if node[1] == goal:
	    tmp_node = node[1]
	    # go back to the start
	    parent_node = node_cost[tmp_node][1]
	    while parent_node:
		best_path.append(tmp_node)
		tmp_node = parent_node
		parent_node = node_cost[tmp_node][1]
	    best_path.append(tmp_node)
	    return best_path[::-1]
	elif node[1] in explored:
	    continue
	parent_node = node[1]
	children_dict = graph[parent_node]
	explored[parent_node] = node
	parent_cost = node_cost[parent_node][0]
	for child in children_dict:
	    # cost from start to this child node
	    g = children_dict[child]['weight'] + parent_cost
	    # estimate cost from this child node to goal
	    h = euclidean_dist_heuristic(graph, child, goal)
	    if child not in node_cost or g < node_cost[child][0]:
		# only store g, not g+h
		node_cost[child] = (g, parent_node)		
	    if child not in explored:
		# store g+h to the priority queue
		frontier.append((g+h, child))
	
	


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
   
    if start == goal:
	return []
    explored_start, explored_goal = {}, {}
    node_cost_start, node_cost_goal = {}, {} 
    best_path = []
    parent_node = None
    
    # best node meet so far 
    # (current node, sum cost of current node to start and to goal)  
    best_meet = (None, float('inf'))
    
    # create two priority queue
    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()
    # initialize two Priority Queue
    frontier_start.append((0,start))
    frontier_goal.append((0,goal))
    node_cost_start[start] = (0,None)
    node_cost_goal[goal] = (0,None)
    
    while True:
	if frontier_start.size() == 0 or frontier_goal.size() == 0:
	    return []
	# from start
	cur_cost, cur_node = frontier_start.pop()
	if cur_node in explored_goal:
	    explored_goal = merge_explored_frontier(explored_goal, frontier_goal, node_cost_goal)
	    explored_start[cur_node] = node_cost_start.pop(cur_node)
	    best_meet = (cur_node, cur_cost + explored_goal[cur_node][0])
	    best_meet = update_best_meet(best_meet, explored_goal, explored_start)
	    break
	elif cur_node not in explored_start:
	    children_dict = graph[cur_node]
	    explored_start[cur_node] = node_cost_start.pop(cur_node)
	    #explored_start[cur_node] = (cur_cost, cur_node)
	    parent_cost = explored_start[cur_node][0]
	    for child in children_dict:
		tmp_cost = children_dict[child]['weight'] + parent_cost
		if child in node_cost_start and tmp_cost < node_cost_start[child][0]:
		    del node_cost_start[child]
		if child not in explored_start and child not in node_cost_start:
		    node_cost_start[child] = (tmp_cost, cur_node)
		    frontier_start.append((tmp_cost, child))
	# from goal 
	cur_cost, cur_node = frontier_goal.pop()
	if cur_node in explored_start:
	    explored_start = merge_explored_frontier(explored_start, frontier_start, node_cost_start)
	    explored_goal[cur_node] = node_cost_goal.pop(cur_node)
	    best_meet = (cur_node, cur_cost + explored_start[cur_node][0])
	    best_meet = update_best_meet(best_meet, explored_start, explored_goal)
	    break
	elif cur_node not in explored_goal:
	    children_dict = graph[cur_node]
	    explored_goal[cur_node] = node_cost_goal.pop(cur_node)
	    #explored_goal[cur_node] = (cur_cost, cur_node)
	    parent_cost = explored_goal[cur_node][0]
	    for child in children_dict:
		tmp_cost = children_dict[child]['weight'] + parent_cost
		if child in node_cost_goal and tmp_cost < node_cost_goal[child][0]:
		    del node_cost_goal[child]
		if child not in explored_goal and child not in node_cost_goal:
		    node_cost_goal[child] = (tmp_cost, cur_node)
		    frontier_goal.append((tmp_cost, child))
    # get the best path
    parent_node = best_meet[0]
    while parent_node:
	best_path.insert(0,parent_node) 
	parent_node = explored_start[parent_node][1]
    parent_node = explored_goal[best_meet[0]][1]
    while parent_node:
	best_path.append(parent_node)
	parent_node = explored_goal[parent_node][1]	
    return best_path



# helper for bidirectional_ucs    
def update_best_meet(best_meet, total_explored, this_explored):
    this_cost = float('inf')
    for key in this_explored:
	if key in total_explored:
	    this_cost = this_explored[key][0] + total_explored[key][0]
	    if this_cost < best_meet[1]:
		best_meet = (key, this_cost)
    return best_meet

# helper for bidirectional_ucs    
def merge_explored_frontier(explored, frontier, node_cost):
    while frontier.size() > 0:
	node = frontier.pop()
	tmp_node = node[1]
	if tmp_node not in explored:
	    explored[tmp_node] = node_cost[tmp_node]
    return explored



def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if start == goal:
	return []
    explored_start, explored_goal = {}, {}
    node_cost_start, node_cost_goal = {}, {} 
    best_path = []
    parent_node = None
    
    # best node meet so far 
    # (current node, sum cost of current node to start and to goal)  
    best_meet = (None, float('inf'))
    
    # create two priority queue
    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()
    # initialize two Priority Queue
    frontier_start.append((0,start))
    frontier_goal.append((0,goal))
    node_cost_start[start] = (0,None)
    node_cost_goal[goal] = (0,None)
    
    while True:
	if frontier_start.size() == 0 or frontier_goal.size() == 0:
	    return []
	# from start
	cur_cost, cur_node = frontier_start.pop()
	if cur_node in explored_goal:
	    explored_goal = merge_explored_frontier(explored_goal, frontier_goal, node_cost_goal)
	    explored_start[cur_node] = node_cost_start.pop(cur_node)
	    best_meet = (cur_node, cur_cost + explored_goal[cur_node][0])
	    best_meet = update_best_meet(best_meet, explored_goal, explored_start)
	    break
	elif cur_node not in explored_start:
	    children_dict = graph[cur_node]
	    explored_start[cur_node] = node_cost_start.pop(cur_node)
	    parent_cost = explored_start[cur_node][0]
	    for child in children_dict:
		g = children_dict[child]['weight'] + parent_cost
		h = euclidean_dist_heuristic(graph, child, goal)
		if child in node_cost_start and g < node_cost_start[child][0]:
		    del node_cost_start[child]
		if child not in explored_start and child not in node_cost_start:
		    node_cost_start[child] = (g, cur_node)
		    frontier_start.append((g+h, child))
	# from goal 
	cur_cost, cur_node = frontier_goal.pop()
	if cur_node in explored_start:
	    explored_start = merge_explored_frontier(explored_start, frontier_start, node_cost_start)
	    explored_goal[cur_node] = node_cost_goal.pop(cur_node)
	    best_meet = (cur_node, cur_cost + explored_start[cur_node][0])
	    best_meet = update_best_meet(best_meet, explored_start, explored_goal)
	    break
	elif cur_node not in explored_goal:
	    children_dict = graph[cur_node]
	    explored_goal[cur_node] = node_cost_goal.pop(cur_node)
	    parent_cost = explored_goal[cur_node][0]
	    for child in children_dict:
		g = children_dict[child]['weight'] + parent_cost
		h = euclidean_dist_heuristic(graph, child, start)
		if child in node_cost_goal and g < node_cost_goal[child][0]:
		    del node_cost_goal[child]
		if child not in explored_goal and child not in node_cost_goal:
		    node_cost_goal[child] = (g, cur_node)
		    frontier_goal.append((g+h, child))
    # get the best path
    parent_node = best_meet[0]
    while parent_node:
	best_path.insert(0,parent_node) 
	parent_node = explored_start[parent_node][1]
    parent_node = explored_goal[best_meet[0]][1]
    while parent_node:
	best_path.append(parent_node)
	parent_node = explored_goal[parent_node][1]	

    return best_path


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # raise NotImplementedError

    if goals[0] == goals[1] and goals[0] == goals[2]:
        return []
    
    # Explored dictionaries have entries of (cost from origin, parent)
    frontier = [PriorityQueue(), PriorityQueue(), PriorityQueue()]
    explored = [{}, {}, {}]
    node_data = [{}, {}, {}]
    # False until the initial meeting of two searches occurs
    meet = False

    path = [[], []]

    # Best node met at so far (state, cost)
    best_meet = None

    for i in range(0,3):
        frontier[i].append((0, goals[i]))
        node_data[i][goals[i]] = (0, None)

    while not meet:
        # Alternate, exploring forward and backward
        for i in range(0,3):
            cost, exploring = frontier[i].pop()
            if exploring in explored[(i + 1) % 3]:
                other_idx = (i + 1) % 3
                cont_idx = (i + 2) % 3
            elif exploring in explored[(i + 2) % 3]:
                other_idx = (i + 2) % 3
                cont_idx = (i + 1) % 3
            elif exploring not in explored[i]:
                children = graph[exploring]
                explored[i][exploring] = node_data[i].pop(exploring)
		parent_cost = explored[i][exploring][0]
                for state in children:
                    g = children[state]['weight'] + parent_cost
                    if state in node_data[i] and node_data[i][state][0] > g:
                        del node_data[i][state]
                    if state not in explored[i] and state not in node_data[i]:
                        node_data[i][state] = (g, exploring)
                        frontier[i].append((g, state))
                continue
            else:
                continue

            # Add exploring to the dict of explored states
            explored[i][exploring] = node_data[i].pop(exploring)
            # Create the union of the frontier and the explored sets for the set we connect with
            tmp_explored = merge_explored_frontier(explored[other_idx], frontier[other_idx], node_data[other_idx])
	    explored[other_idx] = tmp_explored
            # Provide the initial meeting point between both explored sets
            best_meet = (exploring, cost + explored[other_idx][exploring][0])
            # Iterate through to see if there is a better meeting point
            best_meet = update_best_meet(best_meet, explored[other_idx], explored[i])
            # Write to the path
            parent = best_meet[0]
            while parent:
                path[0].insert(0, parent)
                parent = explored[i][parent][-1]
            parent = explored[other_idx][best_meet[0]][-1]
            while parent:
                path[0].append(parent)
                parent = explored[other_idx][parent][-1]
            # Union the current explored set with the frontier to aid in finding shortest path for search continuing
            explored[i] = merge_explored_frontier(explored[i], frontier[i], node_data[i])
            meet = True
            break

    # Now continue search for the non-intersected explored set
    # while the state exploring is not in either other stopped explored sets
    while True:
        cost, exploring = frontier[cont_idx].pop()
        # If the state being explored is found in one of the sets, search it to find the lowest cost path
        # Also, search the other explored set to see if it has a lower cost path
        if exploring not in explored[cont_idx]:
            children = graph[exploring]
            explored[cont_idx][exploring] = node_data[cont_idx].pop(exploring)
	    parent_cost = explored[cont_idx][exploring][0]
            for state in children:
                g = children[state]['weight'] + parent_cost
                if state in node_data[cont_idx] and node_data[cont_idx][state][0] > g:
                    del node_data[cont_idx][state]
                if state not in explored[cont_idx] and state not in node_data[cont_idx]:
                    node_data[cont_idx][state] = (g, exploring)
                    frontier[cont_idx].append((g, state))

        if exploring in explored[(cont_idx + 1) % 3]:
            other_idx = (cont_idx + 1) % 3
            best_meet = (exploring, cost + explored[other_idx][exploring][0])
            best_meet = update_best_meet(best_meet, explored[other_idx], explored[cont_idx])
            best_meet = update_best_meet(best_meet, explored[other_idx], node_data[cont_idx])
            next_best_meet = update_best_meet(best_meet, explored[(cont_idx + 2) % 3], explored[cont_idx])
            next_best_meet = update_best_meet(next_best_meet, explored[(cont_idx + 2) % 3], node_data[cont_idx])
            if next_best_meet[-1] < best_meet[-1]:
                best_meet = next_best_meet
                other_idx = (cont_idx + 2) %3
        elif exploring in explored[(cont_idx + 2) % 3]:
            other_idx = (cont_idx + 2) % 3
            best_meet = (exploring, cost + explored[other_idx][exploring][0])
            best_meet = update_best_meet(best_meet, explored[other_idx], explored[cont_idx])
            continue
        else:
            continue
        parent = best_meet[0]
        while parent:
            path[1].insert(0, parent)
            if(parent in explored[cont_idx]):
                parent = explored[cont_idx][parent][-1]
            else:
                parent = node_data[cont_idx][parent][-1]
        parent = explored[other_idx][best_meet[0]][-1]
        while parent:
            path[1].append(parent)
            if parent in path[0]:
                if parent in explored[(cont_idx + 2) % 3] and parent in explored[(cont_idx + 1) % 3]:
                    if explored[(cont_idx + 2) % 3][parent][0] < explored[(cont_idx + 1) % 3][parent][0]:
                        other_idx = (cont_idx + 2) % 3
                    else:
                        other_idx = (cont_idx + 1) % 3
            parent = explored[other_idx][parent][-1]
        break

    if set(path[1]).issubset(set(path[0])):
        return path[0]
    elif set(path[0]).issubset(set(path[1])):
        return path[1]
    if path[0][-1] == path[1][-1]:
        path[1].reverse()
    elif path[0][0] == path[1][-1]:
        path[1].pop()
        return path[1] + path[0]
    path[1].pop(0)
    for state in path[1]:
        path[0].append(state)
    return path[0]



def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # raise NotImplementedError

    if goals[0] == goals[1] and goals[0] == goals[2]:
        return []
    
    # Explored dictionaries have entries of (cost from origin, parent)
    frontier = [PriorityQueue(), PriorityQueue(), PriorityQueue(), PriorityQueue()]
    # The nodes that have already been regarded, prevents re-exploration
    # Holds the children of each pre-explored state
    regarded_nodes = {}

    explored = [{}, {}, {}, {}]
    node_data = [{}, {}, {}, {}]

    path = [[], []]

    # Holds tuples of the start and goals of each A* search
    # Structure is (start, goal)
    start_goal = [None, None, None, None]
    # Booleans to indicate whether fwd/bwd search has resolved
    stopped = [False, False]

    goal_sorted = [None, None, None]
    for i in range(0, 3):
        h1 = heuristic(graph, goals[i], goals[(i + 1) % 3])
        h2 = heuristic(graph, goals[i], goals[(i + 2) % 3])
        if h1 < h2:
            goal_sorted[i] = (h1, h2, goals[i])
        else:
            goal_sorted[i] = (h2, h1, goals[i])

    goal_sorted.sort()

    # Create start and goal states between center and edge states (start, goal)
    start_goal[0] = (goal_sorted[1][-1], goal_sorted[0][-1])
    start_goal[1] = (goal_sorted[0][-1], goal_sorted[1][-1])
    start_goal[2] = (goal_sorted[2][-1], goal_sorted[0][-1])
    start_goal[3] = (goal_sorted[0][-1], goal_sorted[2][-1])

    # Initialize node_data and frontier
    for i in range(0, 4):
        # Initialize frontier by appending start state with cost of 0
        frontier[i].append((0, start_goal[i][0]))
        # Initialize node data with start state and cost of 0
        node_data[i][start_goal[i][0]] = (0, None)

    # goal_sorted no longer necessary
    del goal_sorted

    while not stopped[0] or not stopped[1]:
        for i in range(0, 4, 2):
            pair_idx = int(i / 2)
            # If this search resolved, stop searching
            if stopped[pair_idx]:
                continue
            cost, exploring = frontier[i].pop()
            other_idx = i + 1
            if exploring in explored[other_idx]:
                stopped[pair_idx] = True
                # Stop condition
                tmp_explored = merge_explored_frontier(explored[other_idx], frontier[other_idx], node_data[other_idx])
		explored[other_idx] = tmp_explored
                explored[i][exploring] = node_data[i].pop(exploring)
                best_meet = (exploring, cost + explored[other_idx][exploring][0])
                best_meet = update_best_meet(best_meet, explored[other_idx], explored[i])
                best_meet_other = update_best_meet(best_meet, explored[i], explored[(i + 2) % 4])
                if best_meet[-1] > best_meet_other[-1]:
                    best_meet = best_meet_other
                    other_idx = (i + 2) % 4

                parent = best_meet[0]
                while parent:
                    path[pair_idx].insert(0, parent)
                    parent = explored[i][parent][-1]
                parent = explored[other_idx][best_meet[0]][-1]
                while parent:
                    path[pair_idx].append(parent)
                    parent = explored[other_idx][parent][-1]
                continue
            elif exploring not in explored[i]:
                if exploring in regarded_nodes:
                    children = regarded_nodes[exploring]
                else:
                    children = graph[exploring]
                    regarded_nodes[exploring] = children
                explored[i][exploring] = node_data[i].pop(exploring)
                parent_cost = explored[i][exploring][0]
                for state in children:
                    g = children[state]['weight'] + parent_cost
                    h = heuristic(graph, state, start_goal[i][-1])
                    if state in node_data[i] and node_data[i][state][0] > g:
                        del node_data[i][state]
                    if state not in explored[i] and state not in node_data[i]:
                        node_data[i][state] = (g, exploring)
                        frontier[i].append((g + h, state))

            i += 1
            cost, exploring = frontier[i].pop()
            other_idx = i - 1
            if exploring in explored[other_idx]:
                stopped[pair_idx] = True
                # Stop condition
                tmp_explored = merge_explored_frontier(explored[other_idx], frontier[other_idx], node_data[other_idx])
		explored[other_idx] = tmp_explored
                explored[i][exploring] = node_data[i].pop(exploring)
                best_meet = (exploring, cost + explored[other_idx][exploring][0])
                best_meet = update_best_meet(best_meet, explored[other_idx], explored[i])
                parent = best_meet[0]
                while parent:
                    path[pair_idx].insert(0, parent)
                    parent = explored[other_idx][parent][-1]
                parent = explored[i][best_meet[0]][-1]
                while parent:
                    path[pair_idx].append(parent)
                    parent = explored[i][parent][-1]
                continue
            elif exploring not in explored[i]:
                if exploring in regarded_nodes:
                    children = regarded_nodes[exploring]
                else:
                    children = graph[exploring]
                    regarded_nodes[exploring] = children
                explored[i][exploring] = node_data[i].pop(exploring)
                parent_cost = explored[i][exploring][0]
                for state in children:
                    g = children[state]['weight'] + parent_cost
                    h = heuristic(graph, state, start_goal[i][-1])
                    if state in node_data[i] and node_data[i][state][0] > g:
                        del node_data[i][state]
                    if state not in explored[i] and state not in node_data[i]:
                        node_data[i][state] = (g, exploring)
                        frontier[i].append((g + h, state))
    

    if set(path[1]).issubset(set(path[0])):
        return path[0]
    elif set(path[0]).issubset(set(path[1])):
        return path[1]
    if path[0][-1] == path[1][-1]:
        path[1].reverse()
    elif path[0][0] == path[1][-1]:
        path[1].pop()
        return path[1] + path[0]
    elif path[1][0] == path[0][0]:
        path[0].reverse()
    path[1].pop(0)
    return path[0] + path[1]


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    #  raise NotImplementedError
    return 'Bian Du'

# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
