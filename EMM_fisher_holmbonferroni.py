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
# Initialize stats priorityqueue which will be a min-heap NVM SHOULD BE DONE AT EACH LEVEL SO THAT IT IS EMPTY

alpha = 0.05
####################################################################################################################################
# EMM framework

import heapq


########################################### FOR THESIS: CLASS UNBOUNDED PRIORITY QUEUE #############################################
class UnboundedPriorityQueue:
    """
    Ensures uniqueness
    """

    def __init__(self):
        self.values = []

    def add(self, pval, quality, desc, n):
        if any((set(e) == set(desc) for (_, e, _) in self.values)):
            return  # avoid duplicates of exactly the same description

        # NOW ALSO AVOID DUPLICATES OF descriptions where each individual element is the same but their odering is different

        new_entry = [(pval, -1*quality), desc, n]
        heapq.heappush(self.values, new_entry)

    def get_values(self):
        for (q, e, n) in sorted(self.values, reverse=True):
            yield (q, e, n)

    def show_contents(self):  # for debugging
        print("show_contents")
        for (q, entry_count, e) in self.values:
            print(q, entry_count, e)

    def pop(self):
        return heapq.heappop(self.values)

####################################################################################################################################

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
        if any((set(e) == set(element) for (_, _, e, _) in self.values)): #use set to ensure that different orderings are also picked upon, e.g. [A=a, B=b] == [B=b, A=a]
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
    Ensures uniqueness
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
        stats_queue = UnboundedPriorityQueue()
        for seed in candidateQueue.get_values():
            print("    seed : ", seed)
            for desc in eta(seed):
                if satisfies_all(desc):
                    ### compute p-value between significance between the desc and the complement's targets
                    pval = fisher_exact(desc) # given a subgroup description desc, compute pval between the subgroup and the complement
                    quality, n = eval_quality(desc)

                    stats_queue.add(pval, quality, desc, n)

        # store the number of hypotheses we have tested, which is equal to the number of elements in our stats queue
        m = len(stats_queue.values)
        rank = 1;

        # here the multiple testing procedure starts
        while stats_queue.values != []:
            candidate = stats_queue.pop()
            pval = candidate[0][0]
            desc = candidate[1]
            n = candidate[2]
            quality = -1*candidate[0][1] # multiply back to positive
            # do multiple testing correction according to holm-bonferroni
            print("candidate: ", candidate)

            if (pval > alpha / (m - rank + 1)):
                break  # no longer significant so we exit the while loop

            resultSet.add(desc, quality, n=n) # as the description was deemed significant, we add it back to the resultSet
            beam.add(desc, quality)             # as the description was deemed significant, we add it to the beam
            rank += 1   # increase rank as we go to the next p-value in line

        for a, b, c in beam.get_values():
            print("a, b, c in beam.get_values(): ", a, b, c)
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
                # candidate = "{} != {}".format(f, i)
                # if not candidate in seed: # if not already there
                #     yield refine(seed, candidate)
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
    d_str = as_string(desc) # this gives issues if the target is in fact boolean
    ind = df.eval(d_str)

    subgroup_targets = df[targets].loc[ind] #this is where you choose your target
    n = len(subgroup_targets)
    crosstab = pd.crosstab(subgroup_targets[targets[0]], subgroup_targets[targets[1]]) #create crosstab for subgroup targets
    crosstab = np.array(crosstab)
    print("crosstab for desc: ", crosstab, desc)

    res = stats.loglin(crosstab, [1, 2], fit=True, param=True)
    deviance = np.array(res[0])[0]
    print("deviance: ", deviance, "description: ", d_str, "n: ", n)

    score = deviance
    return score, n

#### FISHER EXACT FUNCTION
def fisher_exact(desc):
# create dataframe with columns: group (subgroup/complement), target (v1*t1, v1*t2, v2*t1, v2*t2 etc for targets v and t)
    d_str = as_string(desc) # this gives issues if the target is in fact boolean
    ind = df.eval(d_str)

    df_subgroup_complement = df[targets].copy()
    df_subgroup_complement['group'] = '' #initialize empty column to store whether the group belongs to subgroup or complement

    # prepare column to hold all possible combinations of target values
    for target in targets: # there will be two targets
        df_subgroup_complement[target] = df_subgroup_complement[target].astype('str')

    df_subgroup_complement['targets'] = df_subgroup_complement[[targets[0], targets[1]]].agg('-'.join, axis=1)
    df_subgroup_complement.drop(columns = targets, inplace=True) #remove targets as the information is not contained in the targets column

    # indicate whether group belongs to subgroup or to complement
    df_subgroup_complement['group'].loc[ind] = 'subgroup'
    df_subgroup_complement['group'].loc[~ind] = 'complement'

    # create crosstab
    crosstab = pd.crosstab(df_subgroup_complement['targets'], df_subgroup_complement['group'])
    crosstab = np.array(crosstab)

    # finally compute pvalue, keep workspace at least 2e9 for 2x6 table, or simulate by using: simulate_p_value = True
    pval = stats.fisher_test(crosstab, simulate_p_value = True, B=50000)[0][0]

    return pval



# EMM_res = EMM(100, 3, 100, eta, satisfies_all, eval_quality, []) # second parameter is d (the depth)
EMM_res = EMM(30, 2, 40, eta, satisfies_all, eval_quality, []) # second parameter is d (the depth)

headers = ["Quality","Description", "n" ]

exc_results = []

#we have to write some code to automatically create some csv files including all the details
for (q,d, adds) in EMM_res.get_values():
    exc_results.append([q,d,adds["n"]])
    #new_row = pd.DataFrame({"Quality": q, "Description": d, "Mean": adds["mean"], "Std": adds["std"], "t": adds["t"], "n": adds["n"]})
    print(q,d, adds)

pd.DataFrame(exc_results, columns=headers).to_csv("./data/results_fisher_holmbonferroni.csv",index=False, sep=";")

print("******************************************************************")