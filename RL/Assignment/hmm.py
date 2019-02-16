import random
import numpy as np

trans = np.array([[0.7, 0.3], [0.3, 0.7]])
obsv = np.array([[0.9, 0.1], [0.2, 0.8]])

"""
This is strictly coded for the bayes network given in homework exercise 2.
It can be extended for other networks
"""

class Node:
  #Each node in Bayes Net class
  def __init__(self, parent, node_type="state", label=""):
    """
    parent is a Node is the parent of this particular object. Empty if it has no parent

    the cpt is given as follows:
    {"parent_true":{"p_true":0.7, "p_false":0.3}, "parent_false":{"p_true":0.3, "p_false":0.7}}
    if parent is None (start node); {"p_true":0.5, "p_false":0.5}

    node_type="state"|"obsv"

    label is just for output; Rain or R_t (for states) or Umbrella or U_t (for observations) in this case
    """

    self.parent = parent
    if parent == None:
      self.node_type = "state"
      #since no prior probability was given for the start state, we assume a pr of 0.5
      self.cpt = {"p_true":0.5, "p_false":0.5}
    else:
      self.node_type = node_type

      #again, cpt is hardcoded since this code is strictly for the network in exercise 2
      if (node_type == "state"):
        self.cpt = {"parent_true":{"p_true":0.7, "p_false":0.3}, "parent_false":{"p_true":0.3, "p_false":0.7}}

      elif (node_type == "obsv"):
        self.cpt = {"parent_true":{"p_true":0.9, "p_false":0.1}, "parent_false":{"p_true":0.2, "p_false":0.8}}

    self.label = label

    #some default value for the sample; it will be changed in any case
    self.value = False

  def calc_value(self):
    if (self.parent == None):
      #random truth value of equal chance for start state since pr both ways is 0.5
      self.value = True if random.uniform(0.0, 1.0) < 0.5 else False
    else:
      """
      getting a sample value by case, conditioned on parent's value
      gets the truth probability from the cpt, depending on the value of the parent
      then generates a random number between 0 and 1 and checks if it falls within the probability range
      eg. a pr of 0.7 has a higher chance of a random number btw 0 and 1 falling within it than a pr of 0.3
      """
      if self.parent.value == True:
        pr = self.cpt["parent_true"]["p_true"]
        self.value = True if random.uniform(0.0, 1.0) <= pr else False
      else:
        pr = self.cpt["parent_false"]["p_true"]
        self.value = True if random.uniform(0.0, 1.0) <= pr else False


class BN:
  #Bayes Net class
  def __init__(self, length):
    self.length = length
    self.start_node = Node(None, "state", "Start")
    self.start_node.calc_value()

    self.nodes = [] #this will contain as list of lists [[],[]]. each inner list is a state-obsv pair [state_t, obsv_t]
    parent_state = self.start_node

    for t in range(length):
      state_t = Node(parent_state, "state", "R_"+str(t+1))
      obsv_t = Node(state_t, "obsv", "U_"+str(t+1))
      self.nodes.append([state_t, obsv_t])
      parent_state = state_t


def prior_sample(bn):
  """
  *each sequence is a bayes network
  *required: generation of 15 sequences of samples, with each sequence being of length 20
  *should return [[true, false, ..., 20th value], ..., no_sequences_th list]
  """
  samples = []
  for state_obsv in bn.nodes:
    state_obsv_sample = []
    for node in state_obsv:
      node.calc_value()
      state_obsv_sample.append(node.value)

    samples.append(state_obsv_sample)

  return samples


def forward(t, trans, obsv, pr_x_z_prev):
  """
  pr_x_z_prev = the probability distribution till the previous state. 1*2 np array
  trans = the transition model 2*2 np array
  obsv = the observation model 2*2 np array
  """
  if t == 0:
    return np.array([0.5, 0.5])

  if pr_x_z_prev.size > 0:
    pr_xt_z_prev = trans.dot(pr_x_z_prev)
    pr_x_z = obsv[:,0]*pr_xt_z_prev
    pr_x_z = pr_x_z/pr_x_z.sum()
    return pr_x_z

  if t > 0 and pr_x_z_prev.size == 0:
    pr = np.array([])
    for i in range(0, t+1): #t+1 here is because range does not include the upper bound
      pr = forward(i, trans, obsv, pr)
    return pr


def _forward(prev_f, event_obsv):
  f = trans.dot(prev_f)
  f = obsv[:,0]*f if event_obsv == True else obsv[:,1]*f
  f = f/f.sum()
  return f


"""
Test for forward()
tr = np.array([[0.7, 0.3], [0.3, 0.7]])
ob = np.array([[0.9, 0.1], [0.2, 0.8]])
print (forward(10, tr, ob, np.array([])))
exit()
"""

def backward(k, t, trans, obsv, pr_z_x):
  """
  pr_z_x = the probability distribution till the previous state. 1*2 np array
  trans = the transition model 2*2 np array
  obsv = the observation model 2*2 np array
  """
  if k == t:
    return np.ones(2)

  if pr_z_x.size > 0:
    pr_z_x = obsv[:,0] * pr_z_x
    pr_z_x = trans.dot(pr_z_x)
    return pr_z_x

  if t > k and pr_z_x.size == 0:
    pr = np.array([])
    for i in range(t, k-1, -1): #k-1 here is because range does not include the upper bound
      pr = backward(i, t, trans, obsv, pr)
    return pr


def _backward(next_b, event_obsv):
  b = obsv[:,0]*next_b if event_obsv == True else obsv[:,1]*next_b
  b = trans.dot(b)
  return b

"""
Test for backward()
tr = np.array([[0.7, 0.3], [0.3, 0.7]])
ob = np.array([[0.9, 0.1], [0.2, 0.8]])
print (backward(1, 2, tr, ob, np.array([])))
exit()
"""

def normalize(vec):
  return vec/vec.sum()

def forward_backward(events, prior):
  t = len(events)

  fv = [[0.0, 0.0] for _ in range(len(events)+1)]
  sv = [[0.0, 0.0] for _ in range(len(events)+1)]
  fv[0] = prior
  b = np.ones(2)

  for i in range(1, t+1):
    fv[i] = _forward(fv[i-1], events[i-1])

  for i in range(t, 0, -1):
    sv[i] = list(normalize(fv[i]*b))
    b = _backward(b, events[i-1])

  return sv




#creating a sequence of length 20
bn = BN(5)

#printing heading for understanding of the result
states = []
for node in bn.nodes:
  states.append([node[0].label, node[1].label])
print(states)

"""
*each sequence is a bayes network (as created above. 20 required, but I limited it to length 5, for easy reading)
*required: generation of 15 sequences of samples, with each sequence being of length 20. 7 sequences are created below
"""
#creating one sequence sample.
for i in range(1):
  #recalculating initial state for each sequence
  bn.start_node.calc_value()
  print("sequence "+str(i+1)+"; R_0: "+str(bn.start_node.value))

  samples = prior_sample(bn)
  print(samples)
  print("Observations:")
  temp = np.array(samples)
  print(temp[:,1])
  print("Forward Backward:")
  temp = forward_backward(temp[:,1], np.array([0.5, 0.5]))
  print(temp)
