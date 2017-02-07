from __future__ import print_function
import scipy.sparse as sp
import csv
import os
from coclust.CoclustSpecMod import CoclustSpecMod

file_path = os.getcwd()
file_name = file_path + "/datasets/classic32.csv"
csv_file = open(file_name, 'rb')
csv_reader = csv.reader(csv_file, delimiter=",")

nb_row, nb_col, nb_clusters = map(int, csv_reader.next())
X = sp.lil_matrix((nb_row, nb_col))

for row in csv_reader:
    i, j, v = map(int, row)
    X[i, j] = v

model = CoclustSpecMod(n_clusters=nb_clusters)
model.fit(X)

predicted_row_labels = model.row_labels_

for i in range(nb_clusters):
    number_of_rows, number_of_columns = model.get_shape(i)
    print("Cluster", i, "has", number_of_rows, "rows and",
          number_of_columns, "columns.")