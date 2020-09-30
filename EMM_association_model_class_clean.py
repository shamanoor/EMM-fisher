import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.0.2"
os.environ["PATH"] = r"C:\Program Files\R\R-4.0.2\bin\x64" + ";" + os.environ["PATH"]

import pandas as pd
import numpy as np
from scipy import stats
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

stats = importr('stats')

df = pd.read_csv("./data/contraceptive.csv")
abs_omega = len(df)
print("number of fields: %d" % abs_omega)

print(df.head())

#features that are used to find subgroups with
features = ['wage', 'weducation', 'heducation', 'numborn', 'wwork', 'hocc', 'sol', 'good_media_exposure']
targets = ['contraceptive', 'wismuslim']

####################################################################################################################################
# EMM framework

import heapq

#
class BoundedPriorityQueue:
    """
    Ensures uniqueness
    Keeps a maximum size (throws away value with least quality)
    """

    def __init__(self, bound):
        self.values = []
        self.bound = bound
        self.entry_count = 0

    def add(self, element, quality, **adds):
        if any((set(e) == set(element) for (_, _, e, _) in self.values)):
            return  # avoid duplicates
        #if any((self.desc_intersect(element, e) for (_,_,e) in self.values)):
        #    return
        new_entry = (quality, self.entry_count, element, adds)
        if (len(self.values) >= self.bound):
            heapq.heappushpop(self.values, new_entry)
        else:
            heapq.heappush(self.values, new_entry)

        self.entry_count += 1

    def get_values(self):
        for (q, _, e, x) in sorted(self.values, reverse=True):
            yield (q, e, x)

    def show_contents(self):  # for debugging
        print("show_contents")
        for (q, entry_count, e) in self.values:
            print(q, entry_count, e)

#
class Queue:
    """
    Ensures uniqness
    """

    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        if item not in self.items:
            self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def get_values(self):
        return self.items

    def add_all(self, iterable):
        for item in iterable:
            self.enqueue(item)
            self.enqueue(item)

    def clear(self):
        self.items.clear()

#
def EMM(w, d, q, eta, satisfies_all, eval_quality, catch_all_description):
    """
    w - width of beam
    d - num levels
    q - max results
    eta - a function that receives a description and returns all possible refinements
    satisfies_all - a function that receives a description and verifies whether it satisfies some requirements as needed
    eval_quality - returns a quality for a given description. This should be comparable to qualities of other descriptions
    catch_all_description - the equivalent of True, or all, as that the whole dataset shall match
    """
    resultSet = BoundedPriorityQueue(q)
    candidateQueue = Queue()
    candidateQueue.enqueue(catch_all_description)
    for level in range(d):
        print("level : ", level)
        beam = BoundedPriorityQueue(w)
        for seed in candidateQueue.get_values():
            print("    seed : ", seed)
            for desc in eta(seed):
                if satisfies_all(desc):
                    quality,n = eval_quality(desc)
                    print("        desc : ", desc, ", quality : ", quality, "n: ", n)
                    resultSet.add(desc, quality, n=n)
                    beam.add(desc, quality)
        # beam.show_contents()
        #candidateQueue.clear()
        candidateQueue = Queue()
        candidateQueue.add_all(desc for (_, desc, _) in beam.get_values())
    return resultSet

####################################################################################################################################

def refine(desc, more):
    copy = desc[:]
    copy.append(more)
    return copy


def eta(seed):
    print("eta ", seed)
    if seed != []:              #we only specify more on the elements that are still in the subset
        d_str = as_string(seed)
        ind = df.eval(d_str)
        df_sub = df.loc[ind, ]
    else:
        df_sub = df
    for f in features:
        column_data = df_sub[f]
        if (df_sub[f].dtype == 'float64'): #get quantiles here instead of intervals for the case that data are very skewed
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1,6): #determine the number of chunks you want to divide your data in
                x = np.percentile(dat,100/i) #
                candidate = "{} <= {}".format(f, x)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'object'):
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
                candidate = "{} != '{}'".format(f, i)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'int64'):
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1,6): #determine the number of chunks you want to divide your data in
                x = np.percentile(dat,100/i) #
                candidate = "{} == {}".format(f, x)
                if not candidate in seed:  # if not already there
                    yield refine(seed, candidate)
                # candidate = "{} <= {}".format(f, x)
                # if not candidate in seed: # if not already there
                #     yield refine(seed, candidate)
                # candidate = "{} > {}".format(f, x)
                # if not candidate in seed: # if not already there
                #     yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'bool'):
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == {}".format(f, i)
                if not candidate in seed: # if not already there
                    yield refine(seed, candidate)
        else:
            assert False

def as_string(desc):
    return ' and '.join(desc)

def satisfies_all(desc):
    d_str = as_string(desc)
    #print("satisfies_all: ",d_str)
    ind = df.eval(d_str)
    return sum(ind) > 5

def eval_quality(desc):
    print("desc before as_string: ", desc)
    d_str = as_string(desc) # this gives issues if the target is in fact boolean
    ind = df.eval(d_str)

    subgroup_targets = df[targets].loc[ind] #this is where you choose your target
    n = len(subgroup_targets)
    crosstab = pd.crosstab(subgroup_targets[targets[0]], subgroup_targets[targets[1]]) #create crosstab for subgroup targets
    crosstab = np.array(crosstab)
    print("crosstab: ", crosstab)

    res = stats.loglin(crosstab, [1, 2], fit=True, param=True)
    deviance = np.array(res[0])[0]

    score = deviance
    return score, n


# EMM_res = EMM(100, 3, 100, eta, satisfies_all, eval_quality, []) # second parameter is d (the depth)
EMM_res = EMM(10, 2, 10, eta, satisfies_all, eval_quality, []) # second parameter is d (the depth)

headers = ["Quality","Description", "n" ]

exc_results = []

#we have to write some code to automatically create some csv files including all the details
for (q,d, adds) in EMM_res.get_values():
    exc_results.append([q,d,adds["n"]])
    #new_row = pd.DataFrame({"Quality": q, "Description": d, "Mean": adds["mean"], "Std": adds["std"], "t": adds["t"], "n": adds["n"]})
    print(q,d, adds)

pd.DataFrame(exc_results, columns=headers).to_csv("./data/results_idk_what_target_is_numborn.csv",index=False, sep=";")