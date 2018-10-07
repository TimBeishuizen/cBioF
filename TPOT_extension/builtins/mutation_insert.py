# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import random

from inspect import isclass

def mutInsert(individual, pset):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """

    # Find  possible locations for inserts
    possible_indices = []

    for j in range(len(individual)):
        if str(individual[j].ret) == "<class 'numpy.ndarray'>":
            possible_indices.append(j)

    # Select location for insert
    index = random.choice(possible_indices)
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,