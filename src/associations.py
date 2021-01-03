try:
    import pandas as pd
    import numpy as np
    import collections
    import itertools
    import time
    import os
    from collections import defaultdict
    import csv
except ImportError as e:
    print("{} cannot be imported".format(e))


class Association_Rules(object):

    def __init__(self, data, support):
        self.data = data
        self.data = self.data[['InvoiceNo', 'StockCode']]
        self.support = support
        self.initial_transactions = self.first_operation()
        self.ones = self.one_occur()
        self.twos = self.computation()

    def first_operation(self):
        dic = defaultdict(list)
        val_data = self.data.values
        for row in val_data:
            dic[row[0]].append(row[1])
        val_dic = list(dic.values())
        dic = None
        del dic
        del val_data
        print("The nested transactions are done")
        return [twos for twos in val_dic if len(twos) > 1]

    def one_occur(self):
        items = {}
        for transaction in self.initial_transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1
        #for key in list(items.keys()):
        #    if items[key] < self_support * (len(transactions) / 100):
        #        del items[key]
        print("The support for all one products is done")
        lens = len(self.initial_transactions)
        return {k:v/lens for k,v in items.items()}

    def computation(self):
        di1 = {}
        print("The computation is started")
        for trans in self.initial_transactions:
            if len(trans) == 2 and trans[0] != trans[1]:
                sort_trans = sorted(trans)
                if tuple(sort_trans) in di1:
                    di1[(sort_trans[0], sort_trans[1])] += 1
                else:
                    di1[(sort_trans[0], sort_trans[1])] = 1

        for key in list(di1.keys()):
            if di1[key] < 2:
                del di1[key]

        for key in di1.keys():
            for trans in self.initial_transactions:
                if set(key).issubset(set(trans)):
                    di1[key] += 1

        print("The two product support is done")
        lens = len(self.initial_transactions)
        return {k:v/lens for k,v in di1.items()}
        
    def lift(self):
        lift_dic = {}
        for k2, v2 in self.twos.items():
            try:
                lift_dic[k2] = v2/(self.ones[k2[0]] * self.ones[k2[1]])
            except:
                KeyError
        return {k:v for k,v in lift_dic.items() if v != 0}





















