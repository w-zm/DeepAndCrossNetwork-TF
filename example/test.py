import pandas as pd

a = pd.DataFrame({'id':['a', 'b', 'c', 'a']})
us = a['id'].unique()
b = dict(zip(us, range(0, len(us)+0)))
print(b)
a[col].map(self.feat_dict.feat_dict[col])