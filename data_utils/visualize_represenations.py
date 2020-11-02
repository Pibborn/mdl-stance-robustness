# https://towardsdatascience.com/visualization-of-word-embedding-vectors-using-gensim-and-pca-8f592a5d3354
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

import json

def compare_datasets(pooling, testset):
    df_all = pd.DataFrame()
    feat_columns = []
    for dataset in DATASETS:
        data = np.load(FOLDER+REPR_FILE_FORMAT.format(dataset, testset, pooling))
        #data = data[:100]
        feat_columns = [str(i) for i in range(len(data[0]))]
        df = pd.DataFrame(data, columns=feat_columns)
        df['y'] = [dataset for i in range(len(data))]
        #df = pd.DataFrame(data=data, index=[dataset for i in range(len(data))], columns=["labels"]+[str(i) for i in range(len(data[0]))])
        #df = pd.DataFrame(data=data, columns=["labels"]+[str(i) for i in range(len(data[0]))])
        if len(df_all) == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    #https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df_all[feat_columns].values)

    df_all['pca-one'] = pca_result[:,0]
    df_all['pca-two'] = pca_result[:,1]
    df_all['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_all,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(FOLDER+"all_{}_1024dim_pca_{}.png".format(testset, pooling))

def compare_dataset_with_attacks(pooling, dataset, set):
    df_all = pd.DataFrame()
    feat_columns = []
    for s in attacksets+[set]:
        data = np.load(FOLDER+REPR_FILE_FORMAT.format(dataset, s, pooling))
        #data = data[:100]
        feat_columns = [str(i) for i in range(len(data[0]))]
        df = pd.DataFrame(data, columns=feat_columns)
        df['y'] = [s for i in range(len(data))]
        #df = pd.DataFrame(data=data, index=[dataset for i in range(len(data))], columns=["labels"]+[str(i) for i in range(len(data[0]))])
        #df = pd.DataFrame(data=data, columns=["labels"]+[str(i) for i in range(len(data[0]))])
        if len(df_all) == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    #https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df_all[feat_columns].values)

    df_all['pca-one'] = pca_result[:,0]
    df_all['pca-two'] = pca_result[:,1]
    df_all['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 4),
        data=df_all,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(FOLDER+"{}_w_attacks_1024dim_pca_{}.png".format(dataset, pooling))

def _compare_repr_distance(pooling, dataset, adv_set):
    data_test = np.load(FOLDER+REPR_FILE_FORMAT.format(dataset, "test", pooling))
    data_attack = np.load(FOLDER+REPR_FILE_FORMAT.format(dataset, adv_set, pooling))
    assert len(data_test) == len(data_attack), "Attack and Test set don't have the same length"

    dist = 0
    for a, b in zip(data_test, data_attack):
        dist += np.linalg.norm(a-b)
    dist = dist / len(data_attack)
    return dist

def compare_all_dataset_repr_distances(pooling):
    result = {}
    for dataset in DATASETS:
        for attack in attacksets:
            result[dataset+"->"+attack] = _compare_repr_distance(pooling, dataset, attack)
    return result




attacksets = "test_spelling,test_negation,test_paraphrase".split(",")
DATASETS = "arc,argmin,fnc1,ibmcs,iac1,perspectrum,semeval2016t6,semeval2019t7,scd,snopes".split(",")
CALC_PCA = True
CALC_REPR_DISTANCE = True

# todo modify
FOLDER = "../checkpoints/stance_detection_models/mt-dnn-arc,argmin,fnc1,ibmcs,iac1,perspectrum,semeval2016t6,semeval2019t7,scd,snopes_ST_seed0_ep5_mt_dnn_large_answer_opt1_PRETRAINED_MTDNN_LARGE_2019-06-26T1116/"
REPR_FILE_FORMAT = "{}_{}_seq_ouput_{}.npy"
pooling = "pooled" # "pooled" or "cls", used later on for the REPR_FILE_FORMAT

if __name__ == '__main__':
    """
    Takes the represenations from the checkpoint folder FOLDER for all DATASETS. 
    Change representation file format REPR_FILE_FORMAT if necessary. Resulting files are stored
    in the given checkpoint folder
    """
    if CALC_PCA:
        # Visualizes the test sets of all used datasets in a single figure
        compare_datasets(pooling, "test")

        # Visualizes the spelling adversarial set of all used datasets in a single figure
        compare_datasets(pooling, "test_spelling")

        # Visualizes the negation adversarial set of all used datasets in a single figure
        compare_datasets(pooling, "test_negation")

        # Visualizes the paraphrase adversarial set of all used datasets in a single figure
        compare_datasets(pooling, "test_paraphrase")

        # Visualizes all adversarial and the test set of all datasets separately
        for dataset in DATASETS:
            compare_dataset_with_attacks("pooled", dataset, "test") #last param "test" or "dev"

    if CALC_REPR_DISTANCE:
        # calculate the distances between a test set sample representation and its adv set repr for all samples, then avg
        result = compare_all_dataset_repr_distances(pooling)
        with open(FOLDER+"representation_distances.json", "w") as out_f:
            json.dump(result, out_f, indent=4, sort_keys=True)
