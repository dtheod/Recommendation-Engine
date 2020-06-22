try:
    import pandas as pd
    import numpy as np
    import collections
    import itertools
    import time
    import os
    from collections import defaultdict
    import csv
    import inspect
    import psutil
    from functools import wraps
except ImportError as e:
    print("{} cannot be imported".format(e))

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("[" + fn.__name__ + "] took " + str(t2 - t1) + " seconds")
        return result

    return measure_time


class Association_Rules(object):
    def __init__(self, url, support):
        self.data = pd.read_csv(url)
        self.data = self.data[['Order ID', 'Product ID']]
        self.support = support
        self.initial_transactions = self.testing()
        self.ones = self.one_occur(self.initial_transactions)

    @timefn
    def testing(self):
        dic = defaultdict(list)
        print(self.data.shape)
        val_data = self.data.values
        for row in val_data:
            dic[row[0]].append(row[1])
        val_dic = list(dic.values())
        dic = None
        del dic
        del val_data
        print("The nested transactions are done")
        return val_dic

    @timefn
    def filtering(self, reduced_trans, number):
        print("The filtering for more than product is Done")
        return [twos for twos in self.initial_transactions if len(twos) > number]

    @timefn
    def one_occur(self, transactions):
        items = {}
        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1
        for key in list(items.keys()):
            if items[key] < self.support * (len(transactions) / 100):
                del items[key]
        print("The support for all one products is done")
        return items

    @timefn
    def computation(self, sec_support=None):
        di1 = {}
        filters = self.filtering(self.initial_transactions, 1)
        print("The computation is started")
        for trans in filters:
            if len(trans) == 2 and trans[0] != trans[1]:
                sort_trans = sorted(trans)
                if tuple(sort_trans) in di1:
                    di1[(sort_trans[0], sort_trans[1])] += 1
                else:
                    di1[(sort_trans[0], sort_trans[1])] = 1
        for key in list(di1.keys()):
            if di1[key] < 2:
                del di1[key]

        filters2 = self.filtering(filters, 2)
        for key in di1.keys():
            for trans in filters2:
                if set(key).issubset(set(trans)):
                    di1[key] += 1

        print("The two product support is done")

        return di1

    def products_list(self):
        return sorted(list(set(self.data['Product ID'])))

    @timefn
    def confidence(self):
        dic = {}
        twos = self.computation()
        products = sorted(list(set(self.data['Product ID'])))
        frequency_matrix = np.zeros(shape=(len(products), len(products)))
        print("The confidence starts")
        for index, key in enumerate(products):
            dic[key] = index
        for k2, v2 in twos.items():
            for k1, v1 in self.ones.items():
                if set(k2).issuperset(set((k1,))):
                    frequency_matrix[dic[k2[0]], dic[k2[1]]] = v2 / v1
        print(frequency_matrix.shape)
        transition_matrix = np.divide(frequency_matrix, frequency_matrix.sum(axis=1)[:, None])
        transition_dataframe = pd.DataFrame(data=transition_matrix[0:, 0:], index=self.products_list(),
                                            columns=self.products_list())
        return transition_dataframe





















