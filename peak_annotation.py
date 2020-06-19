import pandas as pd


r = pd.read_table("90-100-1.txt", sep='\t').iloc[:, 1].values
peak = [74, 97, 118, 140, 161, 183, 205, 226, 248, 269]
plot(r)
for j in peak:
    annotate('*', (j, r[j]), size=10)
