import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import create_utility_matrix
from als_matrix_training import ALS_training
import associations
from associations import Association_Rules


def main():

    df = pd.read_csv("~/Projects/Recommendation-Engine/data/data.csv",encoding='unicode_escape')
    print(df.head())
#    asso_class = Association_Rules(df, 0.00005)
#    lifts_reco = asso_class.lift()
#    print(lifts_reco)

    df = df.dropna()
    data = create_utility_matrix(df)
    print("Utility matrix")
    # TODO Put users and items in two-dim list

    n_users = data.customer.unique().shape[0]
    n_items = data.sku.unique().shape[0]
    train_data, test_data = train_test_split(data, test_size=0.20)

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # TODO Change the values to see which give better results.

    als_init = ALS_training(lamda = 0.01,
                            num_epochs=8,
                            dimensions=10,
                            train_data = train_data,
                            test_data = test_data,
                            n_users = n_users,
                            n_items = n_items)

    als_init.als_training()
#    return als_init.als_training()


if __name__ == '__main__':
    main()










