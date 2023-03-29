# import numpy as np
# import scipy
# model_path = 'data/movie/'
# user_emb_matrix = scipy.random.rand(6036,8)
# user_emb_matrix = np.random.random((6036,8))
# for i in range(6036):
#     line = user_emb_matrix[i]
#     for JI in range(8):
#         num = line[JI]
#         embedding.write(str(num))
#         embedding.write("\t")
#     embedding.write("\n")
# embedding = np.zeros((6036,8),dtype='float32')
#     lines = user.readlines()
#     count = 0
#     for line in lines:
#         line1 = str(line)
#         num = line1.split("\t")
#         print(len(num))
#         for i in range(len(num)-1):
#             print(num[i])
#             embedding[count][i] = (num[i])
#         count = count + 1
# # print(embedding.dtype)
# import numpy as np
# import tensorflow as tf
# x = tf.constant([1,2,3,4,5,6,81923])
# with tf.Session() as sess:
#     print(type(sess.run(x)))
# print(x)
from uiprocess import preprocess
rating_train=preprocess('book','data/book/ratings_final.txt')
train = rating_train.strip().split('\n')
print(len(train),train)