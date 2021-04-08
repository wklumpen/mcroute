.. MCRoute documentation master file, created by
   sphinx-quickstart on Wed Apr  7 11:03:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Generalized Markov Chain Modelling for (Transportation) Networks
================================================================

MCRoute is a Python package that enables the creation, manipulation, and study
of networks in which movement along networks is characterized by a Makrov chain
process.

MCRoute extends (and restricts) the 
`NetworkX <https://networkx.org/documentation/stable/>`_ module to require
that nodes and edges have associated transition probability matrices. Networks
consist of directional graphs (NetworkX's DiGraphs).

With MCRoute you can rapidly prototype Markov chain networks and transition 
probability matrices using theoretical distributions or by sampling directly
from data.

Audience
========
This module was originally built to support a theoretical definition and
reserach paper, but is intended to allow both researchers and practitioners,
specifically in the field of transportation, to create and analyze Markov chain
networks. 

Free software
=============
MCRoute is free software; you can redistribute it under the terms of the GNU
General Public License, version 3. We welcome contributors, join us on 
`GitHub <https://github.com/wklumpen/mcroute>`_

.. toctree::
   :maxdepth: 2

   installation
   examples
   reference
   network
   matrix

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
