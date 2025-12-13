import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# PART A: DATA PREPARATION

# Transaction dataset
data = {
    'Transaction_ID': [1,2,3,4,5,6,7,8,9,10],
    'Items': [
        ['Bread', 'Milk', 'Eggs'],
        ['Bread', 'Butter'],
        ['Milk', 'Diapers', 'Beer'],
        ['Bread', 'Milk', 'Butter'],
        ['Milk', 'Diapers', 'Bread'],
        ['Beer', 'Diapers'],
        ['Bread', 'Milk', 'Eggs', 'Butter'],
        ['Eggs', 'Milk'],
        ['Bread', 'Diapers', 'Beer'],
        ['Milk', 'Butter']
    ]
}

df = pd.DataFrame(data)

print("FIRST 10 TRANSACTIONS:")
print(df.head(10))

# Transaction Encoding
te = TransactionEncoder()
te_ary = te.fit(df['Items']).transform(df['Items'])
transactions = pd.DataFrame(te_ary, columns=te.columns_)

print("\nONE-HOT ENCODED TRANSACTIONS:")
print(transactions)
# PART B: APRIORI ALGORITHM

# Applying Apriori with minimum support = 0.2
frequent_items = apriori(transactions, min_support=0.2, use_colnames=True)

print("\nFREQUENT ITEMSETS (support >= 0.2):")
print(frequent_items)

# Generating association rules (confidence >= 0.5)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)
# Selecting important columns only
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

print("\nASSOCIATION RULES (support, confidence, lift):")
print(rules)

# PART C: INTERPRETATION

# Sorting rules by Lift (descending)
strongest_rules = rules.sort_values(by='lift', ascending=False).head(3)

print("\nTOP 3 STRONGEST RULES (BASED ON LIFT):")
print(strongest_rules)

print("\nBUSINESS RECOMMENDATIONS:")
print("1. 'Product Placement Strategy'The supermarket should place Beer and Diapers close to each other or within the same shelves. This strategic placement can increase cross-selling opportunities and encourage customers to buy both items in a single visit.")
print("2. Implement bundle pricing for {Bread, Butter, Milk} and {Diapers, Beer} based on their frequent co-purchase patterns.")



