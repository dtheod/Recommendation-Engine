import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from data_preprocessing import create_utility_matrix
from als_matrix_training import ALS_training

def main():
    df = pd.read_csv("/Users/danistheodoulou/Recomment-Engine/order_products_train.csv",
                     parse_dates=['InvoiceDate'],
                     error_bad_lines=False,
                     low_memory=False)

    df = df.dropna()
    data = create_utility_matrix(df)
    print("Utility matrix")
    # TODO Put users and items in two-dim list

    n_users = data.customer.unique().shape[0]
    n_items = data.sku.unique().shape[0]
    train_data, test_data = cv.trainls_test_split(data, test_size=0.10)

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # TODO Change the values to see which give better results.

    als_init = ALS_training(lamda = 0.01,
                            num_epochs=10,
                            dimensions=10,
                            train_data = train_data,
                            test_data = test_data,
                            n_users = n_users,
                            n_items = n_items)


    return als_init.als_training()


if __name__ == '__main__':
    main()


