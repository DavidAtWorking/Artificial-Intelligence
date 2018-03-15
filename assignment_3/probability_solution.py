"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW.
'''

from random import randint, random
from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    # TODO: finish this function    
    # raise NotImplementedError
    # create five variables
    A_node  = BayesNode(0,2,name='alarm')
    F_A_node = BayesNode(1,2,name='faulty alarm')
    G_node  = BayesNode(2,2,name='gauge')
    F_G_node = BayesNode(3,2,name='faulty gauge')
    T_node  = BayesNode(4,2,name='temperature')
    # connect nodes
    # T
    T_node.add_child(G_node)
    T_node.add_child(F_G_node) 
    # G
    G_node.add_child(A_node)
    G_node.add_parent(T_node)
    G_node.add_parent(F_G_node)
    # F_G
    F_G_node.add_child(G_node)
    F_G_node.add_parent(T_node)
    # A
    A_node.add_parent(G_node)
    A_node.add_parent(F_A_node)
    # F_A
    F_A_node.add_child(A_node)
    # append to nodes list
    nodes.append(A_node)
    nodes.append(F_A_node)
    nodes.append(G_node)
    nodes.append(F_G_node)
    nodes.append(T_node)

    return BayesNet(nodes)


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]
    # TODO: set the probability distribution for each node
    # raise NotImplementedError    
    # T
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([],[])
    T_distribution[index] = [0.8,0.2]
    T_node.set_dist(T_distribution)
    # F_A
    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([],[])
    F_A_distribution[index] = [0.85,0.15]
    F_A_node.set_dist(F_A_distribution)
    # G
    dist = zeros([F_G_node.size(), T_node.size(), G_node.size()], dtype=float32)
    dist[0,0,:] = [0.95, 0.05]
    dist[0,1,:] = [0.05, 0.95]
    dist[1,0,:] = [0.2, 0.8]
    dist[1,1,:] = [0.8, 0.2]
    G_distribution = ConditionalDiscreteDistribution(nodes=[F_G_node, T_node, G_node], table=dist)
    G_node.set_dist(G_distribution)
    # F_G
    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)   
    dist[0,:] = [0.95, 0.05]  
    dist[1,:] = [0.2, 0.8]  
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node,F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)
    # A
    dist = zeros([F_A_node.size(), G_node.size(), A_node.size()], dtype=float32)
    dist[0,0,:] = [0.9, 0.1]
    dist[0,1,:] = [0.1, 0.9]
    dist[1,0,:] = [0.55, 0.45]
    dist[1,1,:] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[F_A_node, G_node, A_node], table=dist)
    A_node.set_dist(A_distribution)
    
    return bayes_net


def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    # TODO: finish this function
    # raise NotImplementedError
    A_node = bayes_net.get_node_by_name("alarm")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings],range(Q.nDims))
    alarm_prob = Q[index]
    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    # TOOD: finish this function
    # raise NotImplementedError
    G_node = bayes_net.get_node_by_name("gauge")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot],range(Q.nDims))
    gauge_prob = Q[index]
    return gauge_prob


def get_temperature_prob(bayes_net, temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    # raise NotImplementedError
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_G_node] = False 
    engine.evidence[F_A_node] = False 
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot],range(Q.nDims))
    temp_prob = Q[index] 
    return temp_prob

def make_game_net():
    nodes = []
    # TODO: fill this out
    # raise NotImplementedError    
    A_node = BayesNode(0,4,name='A')
    B_node = BayesNode(1,4,name='B')
    C_node = BayesNode(2,4,name='C')
    AvB_node = BayesNode(3,3,name='AvB')
    BvC_node = BayesNode(4,3,name='BvC')
    CvA_node = BayesNode(5,3,name='CvA')
    # A
    A_node.add_child(AvB_node)
    A_node.add_child(CvA_node)
    # B
    B_node.add_child(BvC_node)
    B_node.add_child(AvB_node)
    # C
    C_node.add_child(BvC_node)
    C_node.add_child(CvA_node)
    # AvB
    AvB_node.add_parent(A_node)
    AvB_node.add_parent(B_node)
    # BvC
    BvC_node.add_parent(B_node)
    BvC_node.add_parent(C_node)
    # CvA
    CvA_node.add_parent(C_node)
    CvA_node.add_parent(A_node)
    # Append nodes to node list
    nodes.append(A_node)
    nodes.append(B_node)
    nodes.append(C_node)
    nodes.append(AvB_node)
    nodes.append(BvC_node)
    nodes.append(CvA_node)

    return BayesNet(nodes)


def set_game_probability(bayes_net):
    A_node = bayes_net.get_node_by_name('A')
    B_node = bayes_net.get_node_by_name('B')
    C_node = bayes_net.get_node_by_name('C')
    AvB_node = bayes_net.get_node_by_name('AvB')
    BvC_node = bayes_net.get_node_by_name('BvC')
    CvA_node = bayes_net.get_node_by_name('CvA')
    # All AvB, BvC, CvA can use the dist below
    dist = zeros([4, 4, 3], dtype=float32)
    dist[0,0,:] = [0.10,0.10,0.80]
    dist[0,1,:] = [0.20,0.60,0.20]
    dist[0,2,:] = [0.15,0.75,0.10]
    dist[0,3,:] = [0.05,0.90,0.05]
    dist[1,0,:] = [0.60,0.20,0.20]
    dist[1,1,:] = [0.10,0.10,0.80]
    dist[1,2,:] = [0.20,0.60,0.20]
    dist[1,3,:] = [0.15,0.75,0.10]
    dist[2,0,:] = [0.75,0.15,0.10]
    dist[2,1,:] = [0.60,0.20,0.20]
    dist[2,2,:] = [0.10,0.10,0.80]
    dist[2,3,:] = [0.20,0.60,0.20]
    dist[3,0,:] = [0.90,0.05,0.05]
    dist[3,1,:] = [0.75,0.15,0.10]
    dist[3,2,:] = [0.60,0.20,0.20]
    dist[3,3,:] = [0.10,0.10,0.80]
    # A
    A_dist = DiscreteDistribution(A_node)
    index = A_dist.generate_index([],[])
    A_dist[index] = [0.15, 0.45, 0.30, 0.10]
    A_node.set_dist(A_dist)
    # B
    B_dist = DiscreteDistribution(B_node)
    index = B_dist.generate_index([],[])
    B_dist[index] = [0.15, 0.45, 0.30, 0.10]
    B_node.set_dist(B_dist)
    # C
    C_dist = DiscreteDistribution(C_node)
    index = C_dist.generate_index([],[])
    C_dist[index] = [0.15, 0.45, 0.30, 0.10]
    C_node.set_dist(C_dist)
    # AvB
    AvB_node = bayes_net.get_node_by_name('AvB')
    AvB_dist = ConditionalDiscreteDistribution(nodes=[A_node,B_node,AvB_node], table=dist)
    AvB_node.set_dist(AvB_dist)

    # BvC
    BvC_node = bayes_net.get_node_by_name('BvC')
    BvC_dist = ConditionalDiscreteDistribution(nodes=[B_node,C_node,BvC_node], table=dist)
    BvC_node.set_dist(BvC_dist)

    # CvA
    CvA_node = bayes_net.get_node_by_name('CvA')
    CvA_dist = ConditionalDiscreteDistribution(nodes=[C_node,A_node,CvA_node], table=dist)
    CvA_node.set_dist(CvA_dist)
    
    return bayes_net
    

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    return set_game_probability(make_game_net())



def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    # raise NotImplementedError
    AvB_node = bayes_net.get_node_by_name('AvB')
    BvC_node = bayes_net.get_node_by_name('BvC')
    CvA_node = bayes_net.get_node_by_name('CvA')
    engine = EnumerationEngine(bayes_net)
    engine.evidence[AvB_node] = 0
    engine.evidence[CvA_node] = 2 
    for i in range(3):
	Q = engine.marginal(BvC_node)[0]
	index = Q.generate_index([i],range(Q.nDims))
	posterior[i] = Q[index]
    
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    A= bayes_net.get_node_by_name("A")      
    AvB= bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    # TODO: finish this function
    # raise NotImplementedError
    # initialization if initial_state is none
    if not initial_state:
	initial_state = [randint(0,3), randint(0,3), randint(0,3), 0, randint(0,2), 2]
    random_index = 3
    # randomly generate a variable to update
    while random_index == 3 or random_index == 5:
	random_index = randint(0,5)
    tmp_dist = []
    if random_index < 3:
	# update A, B, C
	# store posterior distribution
	tmp_dist = [0, 0, 0, 0]
	for i in range(4):
	    # the posterior probability of the chosen variable = i
	    initial_state[random_index] = i
	    tmp_dist[i] = team_table[initial_state[0]] * team_table[initial_state[1]] * team_table[initial_state[2]] * match_table[initial_state[0], initial_state[1], initial_state[3]] * match_table[initial_state[1], initial_state[2],initial_state[4]] * match_table[initial_state[2], initial_state[0], initial_state[5]]
    else:
	tmp_dist = [0,0,0]
	for i in range(3):
	    # the posterior probability of the chosen variable = i
	    initial_state[random_index] = i
	    tmp_dist[i] = team_table[initial_state[0]] * team_table[initial_state[1]] * team_table[initial_state[2]] * match_table[initial_state[0], initial_state[1], initial_state[3]] * match_table[initial_state[1], initial_state[2], initial_state[4]] * match_table[initial_state[2], initial_state[0], initial_state[5]]
    # generate a sample from tmp_dist
    p = random()
    cumulate = 0
    s = float(sum(tmp_dist))
    for i in range(len(tmp_dist)):
	cumulate += (tmp_dist[i] / s)
	if p < cumulate:
	    initial_state[random_index] = i
	    break
    sample = tuple(initial_state)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A= bayes_net.get_node_by_name("A")      
    AvB= bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    if not initial_state:
	initial_state = [randint(0,3), randint(0,3), randint(0,3), 0, randint(0,2), 2]
    # candidate state and candidate probability
    candidate_state = [randint(0,3), randint(0,3), randint(0,3), 0, randint(0,2), 2]
    this_probability = team_table[initial_state[0]] * team_table[initial_state[1]] * team_table[initial_state[2]] * match_table[initial_state[0], initial_state[1], initial_state[3]] * match_table[initial_state[1], initial_state[2], initial_state[4]] * match_table[initial_state[2], initial_state[0], initial_state[5]]
    candidate_probability = team_table[candidate_state[0]] * team_table[candidate_state[1]] * team_table[candidate_state[2]] * match_table[candidate_state[0], candidate_state[1], candidate_state[3]] * match_table[candidate_state[1], candidate_state[2], candidate_state[4]] * match_table[candidate_state[2], candidate_state[0], candidate_state[5]] 
    if candidate_probability > this_probability:
	return tuple(candidate_state)
    else:
	p = random()
	alpha = candidate_probability / this_probability
	if p < alpha:
	    return tuple(candidate_state)
    sample = tuple(initial_state)    
     
    return sample


def compare_sampling(bayes_net, initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    # raise NotImplementedError        
    # successive iterations to calculate to compare with delta
    N = 10
    burn_in = 20000
    iteration = 1000
    # Gibbs Sampling
    # first burn in
    Gibbs_count = burn_in
    sample = list(initial_state)
    for i in range(burn_in):
	sample = Gibbs_sampler(bayes_net, list(sample))
	Gibbs_convergence[sample[4]] += 1
    burn_in_converge = [x / float(Gibbs_count) for x in Gibbs_convergence]
    # N = 10, diff < delta
    converge = False
    successive_count = 0
    while not converge:
	Gibbs_count += 1
	sample = Gibbs_sampler(bayes_net, list(sample))
	Gibbs_convergence[sample[4]] += 1
	this_converge = [x / float(Gibbs_count) for x in Gibbs_convergence]
	diff = [abs(x-y) for x,y in zip(burn_in_converge,this_converge)]
	burn_in_converge = this_converge
	if all(d < delta for d in diff):
	    successive_count += 1
	    converge = (successive_count >= N and (Gibbs_count - burn_in) > iteration)
	else:
	    successive_count = 0
    Gibbs_convergence = burn_in_converge
 
    # MH Sampling
    # first burn in 
    MH_count = burn_in
    sample = initial_state
    for i in range(burn_in):
	tmp = MH_sampler(bayes_net, sample)
	if tmp == sample:
	    MH_rejection_count += 1
	sample = tmp
	MH_convergence[sample[4]] += 1
    burn_in_converge = [x / float(MH_count) for x in MH_convergence]
    # N = 10, diff < delta
    converge = False
    successive_count = 0 
    while not converge:
	MH_count += 1
	tmp = MH_sampler(bayes_net, sample)
	if tmp == sample:
	    MH_rejection_count += 1
	sample = tmp
	MH_convergence[sample[4]] += 1	
	this_converge = [x / float(MH_count) for x in MH_convergence]
	diff = [abs(x-y) for x,y in zip(burn_in_converge,this_converge)]
	burn_in_converge = this_converge
	if all(d < delta for d in diff):
	    successive_count += 1
	    converge = (successive_count >= N and (Gibbs_count - burn_in) > iteration)
	else:
	    successive_count = 0
    MH_convergence = burn_in_converge
    
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1 
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.15
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    # raise NotImplementedError
    return 'Bian Du'
