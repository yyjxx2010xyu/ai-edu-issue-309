
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import datetime,time
import concurrent.futures
from math import sqrt
from joblib import Parallel, delayed

from HelperClass2.NeuralNet_2_0 import *

train_data_name = ".\data\ch09.train.npz"
test_data_name = ".\data\ch09.test.npz"

def train(hp, folder):
    net = NeuralNet_2_0(hp, folder)
    net.train(dataReader, 50, True)
    print("Accuracy: ", net.Test(dataReader), "eta: ", hp.eta)
    
    trace = net.GetTrainingHistory()
    return trace


def ShowLossHistory(folder, file1, hp1, file2, hp2, file3, hp3, file4, hp4):
    lh = TrainingHistory_2_0.Load(file1)
    axes = plt.subplot(2,2,1)
    lh.ShowLossHistory4(axes, hp1)
    
    lh = TrainingHistory_2_0.Load(file2)
    axes = plt.subplot(2,2,2)
    lh.ShowLossHistory4(axes, hp2)

    lh = TrainingHistory_2_0.Load(file3)
    axes = plt.subplot(2,2,3)
    lh.ShowLossHistory4(axes, hp3)

    lh = TrainingHistory_2_0.Load(file4)
    axes = plt.subplot(2,2,4)
    lh.ShowLossHistory4(axes, hp4)

    plt.show()


def try_hyperParameters(folder, n_hidden, batch_size, eta):
    hp = HyperParameters_2_0(1, n_hidden, 1, eta, 10000, batch_size, 0.001, NetType.Fitting, InitialMethod.Xavier)
    filename = str.format("{0}\\{1}_{2}_{3}.pkl", folder, n_hidden, batch_size, eta).replace('.', '', 1)
    file = Path(filename)
    if file.exists():
        return file, hp
    else:
        lh = train(hp, folder)
        lh.Dump(file)
        return file, hp

#  根据超参数来载入已经训练好的文件
def load_hyperParameters(folder, n_hidden, batch_size, eta):
    filename = str.format("{0}\\{1}_{2}_{3}.pkl", folder, n_hidden, batch_size, eta).replace('.', '', 1)
    file = Path(filename)
    return file

#  得到训练的最终的正确率
def load_accuracy(file):
    lh=TrainingHistory_2_0.Load(file)
    return lh.accuracy_val[-1]

if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    # 需调节参数列表    
    ne_list = [4, 6, 8]
    batch_list = [10, 15, 20]
    eta_list = [0.1, 0.3, 0.5, 0.7]

    max_acc = 0.0
    best_ne = ne_list[0]
    best_batch = batch_list[0]
    best_eta = eta_list[0]
    
    beg_time = datetime.datetime.now()
    
    folder = "Thread"
    
    f = open("thread.ans", "w") 
    
    # try eta
    Parallel(n_jobs=2)(delayed(try_hyperParameters)(folder, best_ne, best_batch, eta) for eta in eta_list)
    
    for eta in eta_list:
        fi = load_hyperParameters(folder, best_ne, best_batch, eta)
        acc = load_accuracy(fi)
        print("ne=%d batch=%d eta=%.1f acc=%.3f" % (best_ne, best_batch, eta, acc), file = f)
        if acc > max_acc:
            max_acc = acc
            best_eta = eta
        #endif
    #endfor
    
    # try batch 
    Parallel(n_jobs=2)(delayed(try_hyperParameters)(folder, best_ne, batch, best_eta) for batch in batch_list)
    
    for batch in batch_list:
        fi = load_hyperParameters(folder, best_ne, batch, best_eta)
        acc = load_accuracy(fi)
        print("ne=%d batch=%d eta=%.1f acc=%.3f" % (best_ne, batch, best_eta, acc), file = f)
        if acc > max_acc:
            max_acc = acc
            best_batch = batch
        #endif
    #endfor
    
    # try ne
    Parallel(n_jobs=2)(delayed(try_hyperParameters)(folder, ne, best_batch, best_eta) for ne in ne_list)
    
    for ne in ne_list:
        fi = load_hyperParameters(folder, ne, best_batch, best_eta)
        acc = load_accuracy(fi)
        print("ne=%d batch=%d eta=%.1f acc=%.3f" % (ne, best_batch, best_eta, acc), file = f)
        if acc > max_acc:
            max_acc = acc
            best_ne = ne
        #endif
    #endfor
    f.close()
    
    end_time = datetime.datetime.now()
    thread_time = (end_time - beg_time).seconds
    
    f = open("single.ans", "w") 
    max_acc = 0.0
    best_ne = ne_list[0]
    best_batch = batch_list[0]
    best_eta = eta_list[0]
    
    
    beg_time = datetime.datetime.now()
    folder = "Single"
    for eta in eta_list:
        fi, hp = try_hyperParameters(folder, best_ne, best_batch, eta)
        acc = load_accuracy(fi)
        print("ne=%d batch=%d eta=%.1f acc=%.3f" % (best_ne, best_batch, eta, acc), file = f)
        if acc > max_acc:
            max_acc=acc
            best_eta=eta
        #endif
    #endfor
    
   
    
    for batch in batch_list:
        fi, hp = try_hyperParameters(folder, best_ne, batch, best_eta)
        acc = load_accuracy(fi)
        print("ne=%d batch=%d eta=%.1f acc=%.3f" % (best_ne, batch, best_eta, acc), file = f)
        if acc > max_acc:
            max_acc=acc
            best_batch=batch
        #endif
    #endfor
     
    for ne in ne_list:
        fi, hp = try_hyperParameters(folder, ne, best_batch, best_eta)
        acc = load_accuracy(fi)
        print("ne=%d batch=%d eta=%.1f acc=%.3f" % (ne, best_batch, best_eta, acc), file = f)
        if acc > max_acc:
            max_acc=acc
            best_ne=ne
        #endif
    #endfor
    f.close()
    end_time = datetime.datetime.now()
    single_time = (end_time - beg_time).seconds
    
    
    print("best_ne:",best_ne," best_batch:",best_batch," best_eta:", best_eta, "acc:", max_acc) 
    print ("Thread Training Time:", thread_time, "s")
    print ("Single Training Time:", single_time, "s")
    
