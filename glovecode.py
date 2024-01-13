#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://www.jianshu.com/p/d0cb367752e8
#https://blog.csdn.net/sinat_26917383/article/details/83029140
#https://zhuanlan.zhihu.com/p/165975230
#https://blog.csdn.net/weixin_38235865/article/details/115578458


# ### 加载词向量
import multiprocessing
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == '__main__':
    multiprocessing.freeze_support()
    glove_file = datapath('glove.6B.100d.txt')
    word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
    glove2word2vec(glove_file, word2vec_glove_file)
    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
    print(model.most_similar('banana'))

    result = model.most_similar(positive=['woman', 'king'], negative=['man'])
    print(result)
    print("{}: {:.4f}".format(*result[0]))

    # ### 训练词向量
    #https://blog.csdn.net/weixin_38235865/article/details/115578458
    #glove安装失败 https://blog.csdn.net/weixin_41596463/article/details/106781674
    #准备数据集
    # from __future__ import print_function
    import argparse
    import pprint
    import gensim
    from glove import Glove
    from glove import Corpus
    import pkuseg
    #分词
    pkuseg.test(r'不要等到毕业以后.txt', r'不要等到毕业以后分词.txt', model_name = "default", user_dict = "default", postag = False, nthread = 5)
    #1.准备数据集
    with open(r'不要等到毕业以后分词.txt','r',encoding='utf-8') as f:
        sentense = [line.replace('\n','').split(' ')  for line in f.readlines() if line.strip()!='']
    #sentense = [['你','是','谁'],['我','是','中国人']]
    corpus_model = Corpus()
    corpus_model.fit(sentense, window=10)#10
    corpus_model.save('corpus.model')
    print('Dict size: %s' % len(corpus_model.dictionary))
    #Dict size: 2485
    print('Collocations: %s' % corpus_model.matrix.nnz)
    #Collocations: 71160

    # 2.训练
    glove = Glove(no_components=100, learning_rate=0.05)#no_components 维度，可以与word2vec一起使用。
    glove.fit(corpus_model.matrix, epochs=10, no_threads=1, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    #3.corpus模型保存与加载
    corpus_model.save('corpus.model')
    corpus_model = Corpus.load('corpus.model')

    print( glove.dictionary)

    # 指定词条词向量
    print(glove.word_vectors[glove.dictionary['你']])

    # 相似词
    print(glove.most_similar('专业', number = 10))

    # 全部词向量矩阵
    print(glove.word_vectors)

    #语料协同矩阵 corpus coocurrence matrix
    print(corpus_model.matrix.todense().tolist())

