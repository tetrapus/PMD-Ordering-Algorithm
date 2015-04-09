#! /usr/bin/python2.7
"""
Usage: %(name)s [options] <algorithm> <ratings> <users>

Options:
    -h --help                           Show this message.
    -n RUNS, --nruns=RUNS               Number of runs [default: 25]
    -d DROP, --drop=drop                Number of entries to drop

Available algorithms: rand clust est
"""

try:
    import nimfa
    import scipy.sparse as sp
    import docopt
except:
    print("nimfa, scipy and numpy are required to run this script.")
import random
import sys
import time


def clustered_select(ratings, users, rank=9, user=None):
    matrix = sp.dok_matrix((len(users), len(users)))
    for k, v in ratings.items():
        matrix[users[k[0]], users[k[1]]] = v
        matrix[users[k[1]], users[k[0]]] = v
    # Run sparse matrix factorisation
    factor = nimfa.mf(matrix, seed="random_c", rank=rank, method="snmf", max_iter=12, initialize_only=True, version='r', eta=1., beta=1e-4, i_conv=10, w_min_change=0)
    result = nimfa.mf_run(factor)
    if len(ratings) >= len(users)**2:
        return # all items expanded
    if user is None:
        # Pick a user to expand
        user = min(users, key=lambda u: len([i for i in ratings if u in i]))
    # Pick a cluster
    clusters = result.basis()
    # Select all rated users
    user_rated = {i[0]: ratings[i] for i in ratings if user == i[1]}
    user_rated.update({i[1]: ratings[i] for i in ratings if user == i[0]})
    cluster = max(range(rank), key=lambda x: (sum(r * clusters[users[u], x] for u, r in user_rated.items())+1)/(len(user_rated)+1)) # Maximise A_u(c)
    # Find the user with the highest affinity to cluster
    candidates = {i for i in users if i not in user_rated}
    candidate = max(candidates, key=lambda x: clusters[users[x], cluster])
    return user, candidate

def probability_select(ratings, users, rank=9, user=None):
    matrix = sp.dok_matrix((len(users), len(users)))
    for k, v in ratings.items():
        matrix[users[k[0]], users[k[1]]] = v
        matrix[users[k[1]], users[k[0]]] = v
    # Run sparse matrix factorisation
    factor = nimfa.mf(matrix, seed="random_c", rank=rank, method="snmf", max_iter=12, initialize_only=True, version='r', eta=1., beta=1e-4, i_conv=10, w_min_change=0)
    result = nimfa.mf_run(factor)
    if len(ratings) >= len(users)**2:
        return # all items expanded
    if user is None:
        # Pick a user to expand
        user = min(users, key=lambda u: len([i for i in ratings if u in i]))

    # Clusters (F)
    clusters = result.basis()

    # Matrix (M)
    recommendations = result.fitted()

    # All rated users (U)
    user_rated = {i[0]: ratings[i] for i in ratings if user == i[1]}
    user_rated.update({i[1]: ratings[i] for i in ratings if user == i[0]})

    # Affiliations (A)
    caff = [(sum(r * clusters[users[u], x] for u, r in user_rated.items())+1)/(len(user_rated)+1) for x in range(rank)]
    
    # Confidence (d)
    conf = sum(sum(clusters[users[u], x] for u in user_rated) for x in range(rank))/clusters.sum()

    # Cluster confidences (C)
    sums = clusters.sum(axis=0).tolist()[0]
    cconf = [sum(clusters[users[u], x] for u in user_rated)/sums[x] for x in range(rank)]
    cconf_norm = max(cconf) or 1
    cconf = [i/cconf_norm for i in cconf]

    # Find the user with the highest affinity to cluster
    candidates = {i for i in users if i not in user_rated}
    candidate = max(candidates, key=lambda x: conf * recommendations[users[user], users[x]] + (1-conf) * (sum((1-cconf[i])*caff[i]*clusters[users[x], i] for i in range(rank))/rank))
    return user, candidate

def random_select(ratings, users, rank=None, user=None):
    if user is None:
        # Pick a user to expand
        user = min(users, key=lambda u: len([i for i in ratings if u in i]))
    user_rated = {i[0]: ratings[i] for i in ratings if user == i[1]}
    user_rated.update({i[1]: ratings[i] for i in ratings if user == i[0]})
    candidates = {i for i in users if i not in user_rated}

    return user, random.choice(list(candidates))

def max_guess_select(ratings, users, rank=9, user=None):
    matrix = sp.dok_matrix((len(users), len(users)))
    for k, v in ratings.items():
        matrix[users[k[0]], users[k[1]]] = v
        matrix[users[k[1]], users[k[0]]] = v
    # Run sparse matrix factorisation
    factor = nimfa.mf(matrix, seed="random_c", rank=rank, method="snmf", max_iter=12, initialize_only=True, version='r', eta=1., beta=1e-4, i_conv=10, w_min_change=0)
    result = nimfa.mf_run(factor)
    if user is None:
        # Pick a user to expand
        user = min(users, key=lambda u: len([i for i in ratings if u in i]))
    recommendations = result.fitted()
    rval = max([i for i in users if (i, user) not in ratings and (user, i) not in ratings], key=lambda x: recommendations[users[user], users[x]])
    return user, rval
    

def test(true_ratings, matrix, users, f, test=25):
    cpy = dict(matrix)
    # calculate error of inference 
    res = []
    while len(res) < test:
        # test n items
        u1, u2 = f(cpy, users)
        key = tuple(sorted([u1, u2]))
        if key in true_ratings:
            v = true_ratings[key]
            res.append(v)
            yield v
        else:
            v = 0
        cpy[u1, u2] = v
        cpy[u2, u1] = v
    
def test_one(ratings, users, f, user=None):
    pass


if __name__ == "__main__":
    algos = {"rand": random_select, "clust": clustered_select, "est": max_guess_select, "prob": probability_select}
    d = docopt.docopt(__doc__ % {"name": sys.argv[0]})
    true_ratings, users, f = eval(open(d["<ratings>"]).read()), eval(open(d["<users>"]).read()), algos[d["<algorithm>"]]
    # new user
    #user = "lyucit"
    #matrix = dict(true_ratings)
    #drop = random.sample(list(users), int(d["--drop"]))
    #for u in drop:
    #    del matrix[u, user]
    #    if u != user:
    #        del matrix[user, u]
    ## empty
    ##matrix = {(i, i): 1 for i in users}
    ##for i in users:
    ##    matrix[u"lyucit", i] = true_ratings[u"lyucit", i]
    ##    matrix[i, u"lyucit"] = true_ratings[i, u"lyucit"]
    ### drop 50% of the data
    drop = random.sample({tuple(sorted(i)) for i in true_ratings}, 83**2/4)
    matrix = dict(true_ratings)
    for i in drop:
        del matrix[i]
        if i[1] != i[0]:
            del matrix[tuple(i[::-1])]
    res = test(true_ratings, matrix, users, f, test=int(d["--drop"]))
    for i in res:
        print(i)
        time.sleep(1)
    print("Average: %f" % (sum(res)/len(res)))

