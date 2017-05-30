import matplotlib.pyplot as plt
import numpy as np, scipy.sparse as sp, scipy.io as io
import json
import scipy.sparse
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.sampledata.les_mis import data
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import pandas as pd
from coclust.io.data_loading import load_doc_term_data
from coclust.visualization import (plot_reorganized_matrix,
                                  plot_cluster_top_terms,
                                  plot_max_modularities)
from coclust.evaluation.internal import best_modularity_partition
from coclust.coclustering import CoclustMod


# Lecture des données
path = '../datasets/cstr_coclustFormat.mat'
matlab_dict = io.loadmat(path)
doc_term_data = load_doc_term_data(path)

# doc_term_matrix contient la matrice terme-document avec les valeurs associées
X = matlab_dict['doc_term_matrix']


# Application de l'algo de coclustering
clusters_range = range(2, 6)
model, modularities = best_modularity_partition(X, clusters_range, n_rand_init=1)


# Renvoie les clusters associés aux colonnes (termes)
predicted_column_labels = model.column_labels_


# Renvoie les clusters associés aux lignes (documents)
predicted_row_labels = model.row_labels_


# Renvoie les libellés de colonnes (termes)
namcol = matlab_dict['term_labels']


# Renvoie les libellés de lignes (documents)
true_row_labels = matlab_dict['doc_labels']


# Récupération des clusters associés aux colonnes
clustcol = predicted_column_labels

### Modification du tableau des labels de lignes pour avoir des labels uniques (prérequis Bokeh)


namrow = np.char.mod('%d', true_row_labels[0])
rowlabels = [row + " - " + str(i) for row, i in zip(namrow, range(0,len(true_row_labels[0])))]

clustrow = predicted_row_labels


#Conversion des tableaux de labels associés aux clusters en JSON
nodescol = json.loads(json.dumps([{'name':n,'group':c} for n, c in zip(namcol, clustcol)]))
nodesrow = json.loads(json.dumps([{'name':n,'group':c} for n, c in zip(rowlabels, clustrow)]))


# Définition des colonnes de la matrice  d'entrée
source = X.indices.tolist()
target = X.indptr.tolist()
value = X.data.tolist()

#Conversion de la matrice d'entrée en représentation COOrdinate pour faciliter la conversion en Panda Dataframe
coo = X.tocoo()

#Conversion de la matrice d'entrée en dataframe pour faciliter la conversion en JSON
links = pd.DataFrame({'source': coo.row, 'target': coo.col, 'value': coo.data}
                 )[['source', 'target', 'value']].sort_values(['source', 'target']
                 ).reset_index(drop=True)

				 

# Conversion du dataframe en JSON pour traitement Bokeh
linkstab2 = json.loads(json.dumps(json.loads(links.reset_index().to_json(orient='records'))))
#linkstab = links.to_json(orient='records')[1:-1]

### Traitement Bokeh 

namescol = [node['name'] for node in sorted(nodescol, key=lambda x: x['group'])]
namesrow = [node['name'] for node in sorted(nodesrow, key=lambda x: x['group'])]


M = len(nodesrow)
N = len(nodescol)
counts = np.zeros((N, M))

for link in linkstab2:
    counts[link['target'], link['source']] = link['value']
	
    if link['target'] < M:
        counts[link['source'], link['target']] = link['value']
	
colormap = ["#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
            "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]

xname = []
yname = []
color = []
alpha = []
for i, node1 in enumerate(nodescol):
    for j, node2 in enumerate(nodesrow):

        xname.append(node1['name'])
        yname.append(node2['name'])
		
        alpha.append(min(counts[i,j]/4.0, 0.9) + 0.1)

        if node1['group'] == node2['group']:
            color.append(colormap[node1['group']])
        else:
            color.append('lightgrey')

source = ColumnDataSource(data=dict(
    xname=xname,
    yname=yname,
    colors=color,
    alphas=alpha,
    count=counts.flatten()
))


p = figure(title="Co-clustering",
           x_axis_location="above", tools="hover,save",
           x_range=namescol, y_range=list(reversed(namesrow)), webgl=True)

p.plot_width = 5000
p.plot_height = 5000
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi/3

p.rect('xname', 'yname', 0.9, 0.9, source=source,
       color='colors', alpha='alphas', line_color=None,
       hover_line_color='black', hover_color='colors')

	   
p.select(HoverTool).tooltips = [
    ('namesrow', '$y, $x'),
    ('count', '@count'),
]


output_file("coclustering.html", title="Coclustering.py")

show(p)