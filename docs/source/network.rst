Networks and State Spaces
=========================
A MCRoute analysis requires two key operating components: A state space, and a
network. State spaces must be defined first, as they are the context under which
all transition probabilities are built.

State Space
===========
A state space consists of an ordered sequence (lists) of values and labels which
represent the various possible states an object can take in a MCRoute network.
These states can be relatively arbitrary, though they typically represent some
sort of value-associated state such as schedule delay or travel time. 

.. note::
    Values associated with states are considered the **lower bound** of a given 
    state. This arbitrary decision is required to support the construciton of
    transition probability matrices from probability distributions.

.. autoclass:: mcroute.StateSpace
    :members:

Network
=======
A network consists of a set of nodes, edges, and a state space under which to
perform Markov-chain related activities. Nodes must be instantiated first, and 
subsequently connected by edges.

.. autoclass:: mcroute.Network
    :members:


