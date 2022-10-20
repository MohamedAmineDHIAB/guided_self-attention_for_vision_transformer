'''
explore the dataset and plot classes histogram

to check if the dataset is balanced.
'''
"""
Example of usage :

    python3 ./src/data/explore.py -dp './data/processed/GRAZ/data.tsv' -sp './reports/figures/exploration.jpg'

"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from argparse import ArgumentParser

sns.set(rc={'axes.facecolor':'white',"axes.edgecolor":"black","axes.grid":False,"axes.labelsize":30,"xtick.labelsize":15,"ytick.labelsize":15})



# data table path
global data_path
# saving path
global save_path


def parse_arguments():
    global data_path
    global save_path

    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('-dp', '--data_path', help='data file',
                        required=True)
    parser.add_argument('-sp', '--save_path',
                        help='exploration plot save path',
                        required=False)


    args = parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path





if __name__ == '__main__':
    parse_arguments()
    df=pd.read_csv(data_path)
    fig, axes = plt.subplots(nrows = 1, ncols =1, figsize = (15, 7),facecolor='white')
    sns.countplot(x="class",data=df,palette=["#777777", "#988ed5","#8FD5FF","#FF96A3"],edgecolor="k")
    plt.xlabel("Gland's Class")
    plt.ylabel('Count')
    plt.savefig(save_path)
