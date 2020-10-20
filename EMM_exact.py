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

dataset = 'diabetes'

if dataset == 'contraceptive':
    df = pd.read_csv("./data/contraceptive_02_oct.csv")
    abs_omega = len(df)
    print("number of fields: %d" % abs_omega)

    print(df.head())

    #features that are used to find subgroups with
    features = ['weducation', 'heducation', 'wismuslim', 'wwork', 'hocc', 'sol', 'contraceptive', 'good_media_exposure', 'wage', 'numborn']
    targets = ['contraceptive', 'wismuslim']

if dataset == 'diabetes':

    df = pd.read_csv("./data/diabetes.csv")
    abs_omega = len(df)
    print("number of fields: %d" % abs_omega)

    print(df.head())

    #features that are used to find subgroups with
    features = ['Age', 'IsFemale', 'Polyuria', 'Polydipsia', 'sudden_weight_loss',
           'weakness', 'Polyphagia', 'Genital_thrush', 'visual_blurring',
           'Itching', 'Irritability', 'delayed_healing', 'partial_paresis',
           'muscle_stiffness', 'Alopecia', 'Obesity', 'class']
    targets = ['IsFemale', 'class']

if dataset == 'cars':
    df = pd.read_csv("./data/car.csv")
    abs_omega = len(df)
    print("number of fields: %d" % abs_omega)

    print(df.head())

    # features that are used to find subgroups with
    features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    targets = ['acceptability', 'maint']

features = [element for element in features if element not in targets]  # remove target features from feature list
alpha = 0.05

'''
w - width of beam
d - num levels
q - max results
'''

width = 30
depth = 2
num_results = 1000

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

class UnboundedPriorityQueue:
    """
    Ensures uniqueness
    """

    def __init__(self):
        self.values = []

    def add(self, pval, quality, desc, n):
        if any((set(e) == set(desc) for (_, e, _) in self.values)):
            return  # avoid duplicates of exactly the same description, regardless of ordering of the descriptors

        new_entry = [(pval, quality), desc, n]
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

        candidateQueue = Queue()
        candidateQueue.add_all(desc for (_, desc, _) in beam.get_values())
    return resultSet

def EMM_fisher_holmbonferroni(w, d, q, eta, satisfies_all, eval_quality, catch_all_description):
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

                    # multiply quality by -1 so that we can find the lowest pvalue with the highest score
                    stats_queue.add(pval, -1*quality, desc, n)

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

def EMM_fisher_benjaminihochberg(w, d, q, eta, satisfies_all, eval_quality, catch_all_description):
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
    stats_queue = UnboundedPriorityQueue() # we can put stats queue here as the whole queue will be emptied by the end of the procedure

    for level in range(d):
        print("level : ", level)
        beam = BoundedPriorityQueue(w)
        largest_rank_found = False # used to correct for multiple testing in each search level
        for seed in candidateQueue.get_values():
            print("    seed : ", seed)
            for desc in eta(seed):
                if satisfies_all(desc):
                    ### compute p-value between significance between the desc and the complement's targets
                    pval = fisher_exact(desc) # given a subgroup description desc, compute pval between the subgroup and the complement
                    quality, n = eval_quality(desc)
                    # multiply pval with -1 because we are building a maxheap
                    # multiply quality by -1 so that we can find the lowest pvalue with the highest score
                    stats_queue.add(-1*pval, -1*quality, desc, n)

        # store the number of hypotheses we have tested, which is equal to the number of elements in our stats queue
        m = len(stats_queue.values)
        rank = len(stats_queue.values) # we start at the last hypothesis so it will be equal to the total size

        # here the multiple testing procedure starts
        while stats_queue.values != []:
            candidate = stats_queue.pop()
            pval = -1*candidate[0][0] # multiple by -1 again to get a positive pval
            desc = candidate[1]
            n = candidate[2]
            print("candidate: ", candidate)
            quality = -1*candidate[0][1] # multiply back to positive
            # do multiple testing correction according to holm-bonferroni
            if largest_rank_found:
                # once it is found we continue simply adding all remaining elements from stats into our beam and resultset queues
                resultSet.add(desc, quality, n=n)  # as the description was deemed significant, we add it back to the resultSet
                beam.add(desc, quality)  # as the description was deemed significant, we add it to the beam
            elif (pval <= alpha / (m - rank + 1)):
                largest_rank_found = True # we found our largest rank for which it is significant
                # add the current candidate to the beam and resultset
                resultSet.add(desc, quality, n=n)  # as the description was deemed significant, we add it back to the resultSet
                beam.add(desc, quality)  # as the description was deemed significant, we add it to the beam
            else:
                rank -= 1   # decrement rank by one as we are traversing in descending order, we only continue doing this for as long as we haven't found a significant description

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
            for i in range(1,6): # determine the number of chunks you want to divide your data in
                x = np.percentile(dat,100/i) #
                candidate = "{} <= {}".format(f, x)
                if not candidate in seed and not ["{} <=".format(f) in elem for elem in seed]: # if not already there
                    identical = covers_same_data(seed, candidate)   # check if the improvement is not redundant
                    if not identical:
                        yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if not candidate in seed and not ["{} >".format(f) in elem for elem in seed]: # if not already there
                    identical = covers_same_data(seed, candidate)   # check if the improvement is not redundant
                    if not identical:
                        yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'object'):
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in seed and not ["{} !=".format(f) in elem for elem in seed]: # if not already there
                    identical = covers_same_data(seed, candidate)   # check if the improvement is not redundant
                    if not identical:
                        yield refine(seed, candidate)
                # According to the paper this must be in here so let's leave it in
                candidate = "{} != '{}'".format(f, i)
                if not candidate in seed and not ["{} ==".format(f) in elem for elem in seed]: # if not already there
                    identical = covers_same_data(seed, candidate)   # check if the improvement is not redundant
                    if not identical:
                        yield refine(seed, candidate)

        elif (df_sub[f].dtype == 'int64'):
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1,6): #determine the number of chunks you want to divide your data in
                x = np.percentile(dat,100/i) #
                candidate = "{} <= {}".format(f, x)
                if not candidate in seed and not ["{} <=".format(f) in elem for elem in seed]: # if not already there
                    identical = covers_same_data(seed, candidate)   # check if the improvement is not redundant
                    if not identical:                               # only allow refinement if there it is not redundant
                        yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if not candidate in seed and not ["{} >".format(f) in elem for elem in seed]: # if not already there
                    identical = covers_same_data(seed, candidate) # check if the improvement is not redundant
                    if not identical:
                        yield refine(seed, candidate)
        elif (df_sub[f].dtype == 'bool'):
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == {}".format(f, i)
                if not candidate in seed: # if not already there
                    if candidate == []:
                        yield refine(seed, candidate)
                    identical = covers_same_data(seed, candidate)   # check if the improvement is not redundant
                    if not identical:
                        yield refine(seed, candidate)
        else:
            assert False

def as_string(desc):
    return ' and '.join(desc)

def covers_same_data(seed, candidate):
    print("seed, candidate: ", seed, ", ", candidate)

    if seed == []:
        # if this is the first seed, then include the description
        return False

    d_refinement = refine(seed, candidate) # refine subgroup
    d_refinement_str = as_string(d_refinement) # prepare to select seed and refinement df

    ind_refinement = df.eval(d_refinement_str) # This is then new subgroup of which we want to test if it is the same
    # as the seed
    seed_str = as_string(seed)
    ind_seed = df.eval(seed_str) # this is the previous subgroup of which we have the refinement

    df_refinement = df.loc[ind_refinement]

    df_seed = df.loc[ind_seed]# select dataframes
    print("df_refinement.equals(df_seed): ", df_refinement.equals(df_seed))
    print("lengths of df_refinement and df_seed: ", len(df_refinement.index), len(df_seed.index))

    if df_seed.equals(df):
        # if our improvement returns the entire dataset and it is not the first seed,
        # then it will not improve the description because it does not add any additional filtering properties
        return True

    return df_refinement.equals(df_seed) # only if it returns the same dataset



def satisfies_all(desc):
    d_str = as_string(desc)

    ind = df.eval(d_str)
    subgroup_size = sum(ind)
    complement_size = len(df) - subgroup_size
    print("subgroup_size, complement_size: ", subgroup_size, complement_size)

    # only satisfy constraints if both subgroup and complement hold at least 5 values
    return (sum(ind) > 5 and complement_size > 5)

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

    complement_mask = df_subgroup_complement['group'] == 'complement'
    subgroup_mask = df_subgroup_complement['group'] == 'subgroup'

    # check if subgroup is empty, if it is, then return 1
    if df_subgroup_complement[complement_mask].empty:
        return 1 # return largest possible pvalue
    if df_subgroup_complement[subgroup_mask].empty:
        return 1 # return largest possible pvalue

    # check if complement is empty, if it is, then return 1
    # create crosstab
    crosstab = pd.crosstab(df_subgroup_complement['targets'], df_subgroup_complement['group'])
    crosstab = np.array(crosstab)

    # finally compute pvalue, keep workspace at least 2e9 for 2x6 table, or simulate by using: simulate_p_value = True
    pval = stats.fisher_test(crosstab, workspace=2e9)[0][0]

    return pval

def eval_quality(desc):
    print("desc before as_string: ", desc)
    d_str = as_string(desc) # this gives issues if the target is in fact boolean
    ind = df.eval(d_str)

    subgroup_targets = df[targets].loc[ind] #this is where you choose your target
    n = len(subgroup_targets)
    crosstab = pd.crosstab(subgroup_targets[targets[0]], subgroup_targets[targets[1]]) #create crosstab for subgroup targets
    crosstab = np.array(crosstab)

    res = stats.loglin(crosstab, [1, 2], fit=True, param=True)
    deviance = np.array(res[0])[0]

    score = deviance
    print("deviance: ", deviance, "description: ", d_str, "n: ", n)
    return score, n


# FIX THE MESS THAT IS DOWN BELOW......

headers = ["Quality","Description", "n" ]

# DO RUN FOR EACH MODEL
print("****************************** REGULAR ASSOCIATION MODEL ******************************")
EMM_association = EMM(width, depth, num_results, eta, satisfies_all, eval_quality, []) # second parameter is d (the depth)

exc_results_association = []

#we have to write some code to automatically create some csv files including all the details
for (q,d, adds) in EMM_association.get_values():
    exc_results_association.append([q,d,adds["n"]])
    print(q,d, adds)

# save to csv
pd.DataFrame(exc_results_association, columns=headers).to_csv("./results/association_model_{}_{}_{}_{}_{}.csv".format(targets[0], targets[1], width, depth, num_results),index=False, sep=";")


print("****************************** HOLM BONFERRONI ******************************")
EMM_fisher_holmbonferroni = EMM_fisher_holmbonferroni(width, depth, num_results, eta, satisfies_all, eval_quality, []) # second parameter is d (the depth)

exc_results_holmbonferroni = []

#we have to write some code to automatically create some csv files including all the details
for (q,d, adds) in EMM_fisher_holmbonferroni.get_values():
    exc_results_holmbonferroni.append([q,d,adds["n"]])
    print(q,d, adds)

# save to csv
pd.DataFrame(exc_results_holmbonferroni, columns=headers).to_csv("./results/holm_bonferroni_{}_{}_{}_{}_{}.csv".format(targets[0], targets[1], width, depth, num_results),index=False, sep=";")

print("****************************** BENJAMINI HOCHBERG ******************************")
EMM_fisher_benjaminihochberg = EMM_fisher_benjaminihochberg(width, depth, num_results, eta, satisfies_all, eval_quality, []) # second parameter is d (the depth)

exc_results_benjaminihochberg = []

#we have to write some code to automatically create some csv files including all the details
for (q,d, adds) in EMM_fisher_benjaminihochberg.get_values():
    exc_results_benjaminihochberg.append([q,d,adds["n"]])
    print(q,d, adds)

# save to CSV
pd.DataFrame(exc_results_benjaminihochberg, columns=headers).to_csv("./results/benjamini_hochberg_{}_{}_{}_{}_{}.csv".format(targets[0], targets[1], width, depth, num_results),index=False, sep=";")