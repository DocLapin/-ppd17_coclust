import os
from scipy.io import loadmat
from coclust.CoclustMod import CoclustMod

file_path = os.getcwd()
file_name = file_path + "/datasets/cstr.mat"
matlab_dict = loadmat(file_name)
X = matlab_dict['fea']

model = CoclustMod(n_clusters=4)
model.fit(X)

print(model.modularity)
predicted_row_labels = model.row_labels_
predicted_column_labels = model.column_labels_
print(predicted_row_labels)
print(predicted_column_labels)