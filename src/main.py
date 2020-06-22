from associations import  Association_Rules
aso = Association_Rules(url="/Users/danistheodoulou/Market Basket Analysis Python/data/transactions_test/Orders-Table 1.csv",
                        support=1)

print(aso.confidence())





























