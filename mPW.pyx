# cython: profile=True
# cython: infer_types=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language = c++
# -*- coding: utf-8 -*-

"""
@author: Barbara Ikica
"""

from collections import defaultdict
from itertools import count
from timeit import default_timer as timer

cimport cython
from libc.stdlib cimport rand, RAND_MAX
from cpython.array cimport array, zero

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef list hashing(int [::1] seq):
    """
    Maps a sequence of labels to a sequence of consecutive natural numbers
    (starting from 0).
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    seq : a sequence of labels
    """
    cdef int el

    mapping = defaultdict(count().__next__)

    return [mapping[el] for el in seq]

@cython.cdivision(True)
cdef double randnum():
    """Generates a random float in [0, 1]."""
    cdef double r
    r = rand()/(RAND_MAX*1.0)
    return r

@cython.cdivision(True)
cdef int randint(int k):
    """Generates a random integer in [1, k]."""
    cdef int i
    i = rand() / (RAND_MAX / (k + 1) + 1)
    return i

def sliding_var(arr, int window_size):
    """
    Calculates sliding-window sample means and variances of the elements of a given array.
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    arr : an array of elements
    window_size : length of the sliding-window
    """
    cdef:
        int j = 0
        int var = 0
        int n = window_size+j-1
        double mu = 0.0, mu_old = 0.0
        double v = 0.0
        vector[double] mu_list = []
        vector[double] var_list = []

    for j in range(0, window_size):
        mu += arr[j]
        v += arr[j]**2

    mu /= window_size
    mu_list.push_back(mu)

    v = 1.0/(window_size-1) * (v - window_size*mu**2)
    var_list.push_back(v)

    for j in range(window_size, len(arr)):
        mu_old = mu
        mu += 1.0/window_size * (arr[j] - arr[j-window_size])
        mu_list.push_back(mu)
        v += 1.0/(window_size-1) * ((arr[j] - arr[j-window_size]) * (arr[j] + arr[j-window_size] - mu - mu_old))
        var_list.push_back(v)

    return mu_list, var_list

cdef int bisect_right(double [::1] a, double x, int hi):
    """Assuming that the list a is sorted, return the index at which the item x
    should be inserted in this list. The returned value lo is such that e <= x
    holds for all e in a[:lo] and e > x holds for all e in a[lo:]. Thus, if x
    already appears in the list, the new item would be inserted right after the
    rightmost x already there. Optional argument hi bounds the slice of a to be
    searched."""
    cdef int lo = 0

    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1

    return lo

@cython.cdivision(True)
def mPW(g, int k, int maxStep=0, Py_ssize_t window_size=0, tol=0.001, int w=6,
                 int fine_tun1=0, int fine_tun2=0):
    """
    Returns a division of the vertices of a given graph into clusters generated
    by the modified Petford-Welsh algorithm.
    ===========================================================================
    Parameters
    ---------------------------------------------------------------------------
    g : an undirected (possibly weighted) graph
    k : initial number of colours (clusters)
    maxStep : maximum number of iterations (default value: 0)
    window_size : length of the sliding window (default value: 0)
    tol : tolerance on the sliding-window variance in the number of bad edges (default value: 0.001)
    w : weight in the probability of recolouring (default value: 6)
    fine_tun1 : flag indicating whether to recolour singleton clusters with the most frequent colours in their respective neighbourhoods (default value: 0)
    fine_tun2 : flag indicating whether to recolour every vertex with the most frequent colour in its neighbourhod (default value: 0)
    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    c : a list of vertex colours (clusters)
    badVert : a list of the number of bad vertices over time
    badEdges : a list of of the number of bad edges over time
    times : time elapsed during the initialisation phase, main iterative loop, and the fine-tuning procedures
    """

    g.simplify()

    if (maxStep == 0) & (window_size == 0):
        window_size = len(g.vs)

    cdef dict times = {}
    times['init'] = 0
    times['main'] = 0
    times['fin'] = 0

    cdef int i, j, vi, max_col, flag, deg_current
    cdef double freq
    cdef Py_ssize_t n = len(g.vs)
    cdef Py_ssize_t m = len(g.es)
    cdef int randv, c_old, current_clr, randid, ni
    cdef int badEdges_step = 0
    cdef double total = 0
    cdef int badVert_step = 0
    cdef int step = 0

    cdef double t1, t2, x

    cdef double mu = 0.0
    cdef double mu_old = 0.0
    cdef double var = 0.0

    t1 = timer()

    cdef vector[int] neig_v_view

    cdef vector[int] badVert_view = []
    cdef vector[int] badEdges_view = []

    cdef array[int] badVert_list_nonzero = array('i', (0 for i in range(n)))
    cdef int [::1] badVert_list_nonzero_view = badVert_list_nonzero

    cdef array[int] badNeigh_list = array('i', (0 for i in range(n)))
    cdef int [::1] badNeigh_list_view = badNeigh_list

    cdef array[int] c = array('i', (0 for i in range(n)))
    cdef int [::1] c_view = c

    # generate an initial k-colouring c (a random list of integers in [0,k-1])
    for i in range(n):
        c_view[i] = randint(k-1)

    cdef array[int] f_vNeig_c = array('i', (0 for i in range(k)))
    cdef int [::1] f_vNeig_c_view = f_vNeig_c

    cdef array[int] col_freq = array('i', (0 for i in range(k)))
    cdef int [::1] col_freq_view = col_freq

    cdef unordered_map[int, vector[int]] clusters

    cdef array[int] degrees = array('i', (0 for i in range(n)))
    cdef int [::1] degrees_view = degrees

    j = 0
    cdef unordered_map[int, vector[int]] map_neigh
    for i in range(n):
        g.vs[i]['id'] = g.vs[i].index
        map_neigh[i] = g.neighbors(i)
        flag = 1

        for ni in map_neigh[i]:
            degrees_view[i] += 1

            if c_view[i] != c_view[ni]: # if the colours of the endvertices i and j are different, classify both as bad vertices
                badNeigh_list_view[i] += 1

                if i < ni:
                    badEdges_step += 1

                if flag == 1:
                    badVert_step += 1
                    badVert_list_nonzero_view[j] = i
                    j += 1
                    flag = 0

    cdef int deg_max = max(degrees_view)

    cdef array[double] cum_weights = array('d', (0 for i in range(deg_max)))
    cdef double [::1] cum_weights_view = cum_weights

    cdef array[int] ind = array('i', (0 for i in range(deg_max)))
    cdef int [::1] ind_view = ind

    t2 = timer()
    times['init'] += t2-t1

    t1 = timer()

    cdef int flag_loop = 1

    cdef double m_squared = m*m * tol*tol

    while flag_loop:

        randid = randint(badVert_step-1)
        randv = badVert_list_nonzero_view[randid]

        deg_current = degrees_view[randv]
        neig_v_view = map_neigh[randv]

        t1 = timer()

        max_col = 0

        for i in range(deg_current):
            ni = neig_v_view[i]
            current_clr = c_view[ni]

            if current_clr > max_col:
                max_col = current_clr

            f_vNeig_c_view[current_clr] += 1 # frequencies of neighbouring colours

        c_old = c_view[randv]

        total = 0
        j = 0

        for i in range(max_col+1):
            freq = f_vNeig_c_view[i]
            f_vNeig_c_view[i] = 0
            if freq != 0:
                total += w ** freq
                cum_weights_view[j] = total
                ind_view[j] = i
                j += 1

        x = randnum() * total
        i = bisect_right(cum_weights_view, x, j-1)
        c_view[randv] = ind_view[i]

        for i in range(deg_current):
            ni = neig_v_view[i]

            if c_view[ni] == c_view[randv]:
                if c_view[ni] != c_old:
                    badEdges_step -= 1

                    badNeigh_list_view[ni] -= 1
                    badNeigh_list_view[randv] -= 1

            else:
                if c_view[ni] == c_old:
                    badEdges_step += 1

                    badNeigh_list_view[ni] += 1
                    badNeigh_list_view[randv] += 1

        badVert_step = 0
        j = 0
        for i in range(n):
            if badNeigh_list_view[i] > 0:
                badVert_step += 1
                badVert_list_nonzero_view[j] = i
                j += 1

        badVert_view.push_back(badVert_step)
        badEdges_view.push_back(badEdges_step)

        if window_size != 0:
            if step < window_size-1:
                mu += badEdges_step
                var += badEdges_step**2
            elif step == window_size-1:
                mu += badEdges_step
                mu /= window_size

                var += badEdges_step**2
                var = 1.0/(window_size - 1) * (var - window_size * mu**2)
            else:
                mu_old = mu
                mu += 1.0/window_size * (badEdges_step - badEdges_view[step-window_size])
                var += 1.0/(window_size-1) * ((badEdges_step - badEdges_view[step-window_size]) * (badEdges_step + badEdges_view[step-window_size] - mu - mu_old))

                if var < m_squared:
                    flag_loop = 0

        if maxStep != 0:
            if step >= maxStep-1:
                flag_loop = 0

        if badVert_step == 0:
            flag_loop = 0
            print('No bad vertices left at step', step)
            print('Remaining number of colours:', len(set(c)))

        step += 1

    t2 = timer()
    times['main'] += t2-t1

    t1 = timer()

    cdef list values = hashing(c_view)
    cdef dict dictionary = dict(zip(c_view, values))

    for i in range(n):
        c_view[i] = dictionary[c_view[i]]

    cdef cmax = max(c_view)

    for i in range(cmax+1):
        h = g.subgraph(g.vs.select([j for j in range(n) if c_view[j] == i]))
        h_conn_comp = list(h.clusters())

        if len(h_conn_comp) > 1:
            for comp in h_conn_comp:
                current_clr = max(c_view)+1

                for v in h.vs[comp]:
                    c_view[v['id']] = current_clr

    cmax = max(c_view)

    if fine_tun1:
        for i in range(n):
            current_clr = c_view[i]
            col_freq_view[current_clr] += 1

        for i in range(cmax+1):
            if col_freq_view[i] == 1:
                vi = c.index(i)

                neig_v_view = map_neigh[vi]
                deg_current = degrees_view[vi]

                for j in range(deg_current):
                    ni = neig_v_view[j]
                    current_clr = c_view[ni]

                    f_vNeig_c_view[current_clr] += 1

                max_col = 0
                for j in range(cmax+1):
                    if f_vNeig_c_view[j] > max_col:
                        max_col = f_vNeig_c_view[j]
                        c_view[vi] = j
                    f_vNeig_c_view[j] = 0

    if fine_tun2:
        for i in range(n):
            zero(f_vNeig_c)
            neig_v_view = map_neigh[i]
            deg_current = degrees_view[i]

            for j in range(deg_current):
                ni = neig_v_view[j]
                current_clr = c_view[ni]

                f_vNeig_c_view[current_clr] += 1

            max_col = max(f_vNeig_c_view)
            c_view[i] = f_vNeig_c.index(max_col)

    t2 = timer()
    times['fin'] += t2-t1

    return list(c), list(badVert_view), list(badEdges_view), times