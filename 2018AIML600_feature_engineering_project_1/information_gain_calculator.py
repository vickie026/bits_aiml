import pandas as pd 
import numpy as np
import math

def descritizer(spliter, data_column) :
    spliter = sorted(spliter)
    first_val = np.min(data_column)
    second_val = 0
    val = 1
    cat_list = ['NONE' for i in range(len(data_column))]
    for i in range(len(spliter)) :       
        second_val = spliter[i]
        for j in range(len(data_column)) :
            if float(data_column[j]) >= first_val and float(data_column[j]) < second_val :
                cat_list[j] = 'L' + str(val)
        val += 1
        first_val = spliter[i]
    second_val = np.max(data_column)
    for i in range(len(data_column)) :
        if float(data_column[i]) >= first_val and float(data_column[i]) <= second_val :
            cat_list[i] = 'L'+str(val)
    return cat_list
def calculate_contigency(dataset, feature_name, label_name) :
    rows = dataset[label_name].unique();
    cols = dataset[feature_name].unique()
    cntgnc_lst = []
    size = len(dataset[label_name])
    for i in range(len(rows)) :
        for j in range(len(cols)) :
            counter = 0
            for k in range(size) :
                if dataset[feature_name][k] == cols[j] and dataset[label_name][k] == rows[i] :
                    counter += 1
            cntgnc_lst.append(counter)
    cntgncy = np.asarray(cntgnc_lst)
    if np.sum(cntgncy) != len(dataset[feature_name]) :
        print (' Something wrong with contigency table values happens.')
    cntgnc_lst = cntgncy.reshape(len(rows),len(cols))
    labels = np.empty(len(rows), dtype = int)
    for i in range(len(labels)) :
        counter = 0
        for j in range(size) :
            if dataset['class'][j] == rows[i] :
                counter += 1
        labels[i] = counter
    return np.transpose(cntgnc_lst), labels

def calculate_abs_entropy(dataset) :
    total = np.sum(dataset)
    res = 0
    for i in dataset :
        val = i / total
        res -= val*math.log(val,2)
    return round(res, 3)

def calculate_condit_entropy(contigency) :
    grand = np.sum(contigency)
    sums = [np.sum(contigency[i]) for i in range(len(contigency))]
    result = 0
    for i in range(len(contigency)) :
        temp_res = 0
        for j in range(len(contigency[i])) :
            val = contigency[i][j]/sums[i]
            if val > 0 :
                temp_res += val*math.log(val,2)
        result -= (sums[i] * temp_res)
    result /= grand
    return round(result, 3)

def calculate_information_gain(contigency, labels) :
    ig = calculate_abs_entropy(labels) - calculate_condit_entropy(contigency)
    return round(ig, 4)

def main() :
    process_data()

def information_theoritic_measurement(feature_name, label_name, data_column, dataset, spliter) :

    data_column = descritizer(spliter, data_column)
#    print(' AFTER ', data_column)
    dataset[feature_name] = data_column
    contigency, labels = calculate_contigency(dataset, feature_name, label_name) 
    ig = calculate_information_gain(contigency, labels)
    return ig

def process_data() :
    dataset = pd.read_csv('glass_features.csv')
    dataset1 = pd.read_csv('glass_target.csv')
    feature_names = list(dataset)
    label_name = 'class'
    ig_dict = {}
    dataset[label_name] = dataset1[label_name]
    spliter = [[1.51, 1.517, 1.52, 1.524, 1.525], [11, 11.5, 12, 12.5, 13, 13.5], [1.5, 2.8, 3.5], [0.9, 1.3, 1.36, 1.5, 1.82, 2, 2.55, 2.8 ], [72, 72.7, 73 ], [0.15, 0.25, 0.32, 0.36, 0.4, 0.55, 0.6, 0.65], [7.5, 8, 8.5, 9, 10 ], [0.1, 0.5, 1, 1.5], [0.2, 0.3, 0.35, 0.4]]
    x = 0
    for feature_name in feature_names :
        data_column = dataset[feature_name].values
        ig = information_theoritic_measurement(feature_name, label_name, data_column, dataset, spliter[x])
        if(feature_name == 'refractive_index') :
            feature_name = 'ri'
        ig_dict[feature_name.upper()] = ig
        x += 1
    sorted_ig_list = sorted(ig_dict.items(), key = lambda x : (x[1], x[0]), reverse = True)
    for x in sorted_ig_list :
        print(str(x[0]).upper()+'\t'+str(x[1]))
    print(' completed ')


if __name__== "__main__":
    main()

        
