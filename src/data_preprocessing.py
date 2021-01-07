import numpy as np
import pandas as pd
import math

def enumerate_dict(i_u_list):
    dictionary = {}
    for i, i_u_list in enumerate(i_u_list):
        dictionary[i_u_list] = i
    return dictionary


def bisection_method(counts):
    ratings = []
    uniques_ordered = sorted(list(set(counts)))
    mini = min(uniques_ordered)
    maxi = max(uniques_ordered)
    mid_point = math.ceil(np.median(uniques_ordered))
    min_betwe_mid = math.ceil(np.median([mini, mid_point]))
    max_betwe_mid = math.ceil(np.median([mid_point, maxi]))
    first = list(range(mini, int(min_betwe_mid)))
    second = list(range(int(min_betwe_mid), int(mid_point)))
    third = list(range(int(mid_point), int(max_betwe_mid)))
    fourth = list(range(int(max_betwe_mid), int(maxi)))
    if len(counts) == 1:
        ratings = 5
        return ratings
    for elements in counts:
        if elements == max(uniques_ordered):
            ratings.append(5)
        elif elements in first:
            ratings.append(1)
        elif elements in second:
            ratings.append(2)
        elif elements in third:
            ratings.append(3)
        elif elements in fourth:
            ratings.append(4)
    return ratings


def create_utility_matrix(df):

    diff_dates = lambda x: abs(x[3] - x[4])

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    selected_df = df[['CustomerID', 'StockCode', 'InvoiceDate']]
    aggr_data = pd.DataFrame(selected_df.groupby(['CustomerID', 'StockCode'])
                             .agg({'StockCode': np.size, 'InvoiceDate': np.max}))
    del selected_df
    aggr_data.rename(columns={'InvoiceDate': 'max_cust_date',
                              'StockCode': 'frequency'}, inplace=True)
    aggr_data = aggr_data.reset_index()
    max_dates_data = aggr_data[['CustomerID', 'max_cust_date']].\
        groupby('CustomerID').\
        aggregate(np.max).reset_index()
    max_dates_data.columns = ['CustomerID', 'max_sku_date']
    new_dataframe = pd.merge(aggr_data, max_dates_data, on='CustomerID', how='left')

    new_dataframe = new_dataframe[['CustomerID',
                                   'StockCode',
                                   'frequency',
                                   'max_cust_date',
                                   'max_sku_date']]

    new_dataframe.columns = ['customer', 'sku', 'frequency', 'max_cust_date', 'max_sku_date']
    new_dataframe['new_date'] = new_dataframe.apply(diff_dates, axis=1)

    new_dataframe['normalised'] = (new_dataframe['new_date'] - new_dataframe['new_date']
                                   .mean()) / (new_dataframe['new_date'].std()) + 1

    new_dataframe['pre_rating'] = round(new_dataframe['frequency'] / new_dataframe['normalised'])
    new_dataframe = new_dataframe[['customer',
                                   'sku',
                                   'pre_rating']]
    new_dataframe['pre_rating'] = new_dataframe['pre_rating'].astype(np.int32)

    # TODO Change the 5 start bucket to include more data

    customers = new_dataframe.customer.unique()
    items = new_dataframe.sku.unique()
    i_dic = enumerate_dict(items)
    u_dic = enumerate_dict(customers)
    u_map = lambda x: u_dic[x]
    i_map = lambda x: i_dic[x]

    for idx, customers in enumerate(customers):
        if idx == 0:
            initial = np.matrix([0, 0, 0])
        new_data = new_dataframe[new_dataframe['customer'] == customers]
        new_data['rating'] = bisection_method(list(new_data['pre_rating']))
        if idx == 0:
            final_data = new_data[['customer', 'sku', 'rating']].values
            final_data1 = np.concatenate((initial, final_data), axis=0)
            final_data1 = final_data1[1, :]
        else:
            final_data = new_data[['customer', 'sku', 'rating']].values
            final_data1 = np.concatenate((final_data1, final_data), axis=0)

    final_data = pd.DataFrame(final_data1, columns=['customer', 'sku', 'rating'])
    final_data['customer'] = final_data['customer'].apply(u_map)
    final_data['sku'] = final_data['sku'].apply(i_map)
    print(final_data)

    return final_data


