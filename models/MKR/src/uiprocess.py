from io import open
import numpy as np
import os

def preprocess(dataset,file):
    model = dataset
    moedl_path = 'data/'+model

    if os.path.exists(moedl_path):
        ratings_final = open( moedl_path + '/ratings_final.txt','r',encoding='utf-8')
    else:
        os.mkdir(moedl_path)
        ratings_final = open( moedl_path + '/ratings_final.txt','w',encoding='utf-8')
        with open(file,"r") as fw:
            lines = fw.readlines()
            for line in lines:
                if line:
                    ratings_final.write(line)

    rating_train = open( moedl_path + '/rating_train.dat','w',encoding='utf-8')
    with  open (moedl_path+'/ratings_final.txt', "r") as fw:
        lines = fw.readlines()
        for line in lines:
            if line:
                user, item, rating = line.strip().split("\t")
                rating_train.write("u" + user + "\t" + "i" + item + "\t" + rating + "\n")

    rating_test = open( moedl_path + '/rating_test.dat','w',encoding='utf-8')
    with  open (moedl_path+'/ratings_final.txt', "r") as fw:
        lines = fw.readlines()[1001:10000]
        for line in lines:
            if line:
                user, item, rating = line.strip().split("\t")
                rating_test.write("u" + user + "\t" + "i" + item + "\t" + rating + "\n")

    rating_t = ''
    with  open (moedl_path+'/ratings_final.txt', "r") as fw:
        lines = fw.readlines()[1001:10000]
        last = lines[-1]
        for line in lines:
            if line:
                user, item, rating = line.strip().split("\t")
                if line is last:
                    rating_t += "u" + user + "\t" + "i" + item + "\t" + rating
                else:
                    rating_t += "u" + user + "\t" + "i" + item + "\t" + rating + "\n"
                # rating_t.write("u" + user + "\t" + "i" + item + "\t" + rating + "\n")
    return rating_t