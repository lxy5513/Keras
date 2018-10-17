### **Embedding层**

> Embedding层只能作为模型的第一层

```
keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```



### Preview

#### Word2Vec和Embeddings

> Word2Vec其实就是通过学习文本来用词向量的方式表征词的语义信息，即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。Embedding其实就是一个映射，将单词从原先所属的空间映射到新的多维空间中，也就是把原先词所在空间嵌入到一个新的空间中去。

Word2Vec模型中，主要有Skip-Gram和CBOW两种模型，从直观上理解，Skip-Gram是给定input word来预测上下文。而CBOW是给定上下文，来预测input word.

 

### Keras Embedding Layer

Keras提供了一个嵌入层，适用于文本数据的神经网络。

它要求输入数据是整数编码的，所以每个字都用一个唯一的整数表示。这个数据准备步骤可以使用Keras提供的Tokenizer API来执行。

嵌入层用**随机权重进行初始化**，并将学习训练数据集中所有单词的嵌入。

它是一个灵活的图层，可以以多种方式使用，例如：

- 它可以单独使用来学习一个单词嵌入，以后可以保存并在另一个模型中使用。
- 它可以用作深度学习模型的一部分，其中嵌入与模型本身一起学习。
- 它可以**用来加载预先训练的词嵌入模型**，这是一种迁移学习。



### function

Embedding layer的作用主要在于学习词语的distributed representation并将极其稀疏的one-hot编码的词语进行降维

正整数（下标）就是one-hot编码 转化成 固定大小的向量

#### 术语

嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]*

> 此时参数 input_dim=1000, output_dim=2
>
> 假如单词表的大小为1000，词向量的维度为2. 经单词频数统计后，tom对应的id=4，而ferry对应的id=20
>
> 将上述转化后，等到一个M1000×2的矩阵，而tom对应的是该矩阵的第4行，取出该行的数据就是[0.25,0.1]
>
> > 如果输入数据不需要词的语义特征语义，简单使用Embedding层就可以得到一个对应的词向量矩阵，但如果需要语义特征，我们大可把以及训练好的词向量权重直接扔到Embedding层中即可



### Parameters

> input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1 
> output_dim：大于0的整数，代表全连接嵌入的维度 
>
> input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
>
> embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers 
>
> embeddings_regularizer: 嵌入矩阵的正则项，为Regularizer对象 
> embeddings_constraint: 嵌入矩阵的约束项，为Constraints对象 



### Example

**输入shape** 
形如（samples，sequence_length）的2D张量 (sequence_length 即是 input_length)
**输出shape** 
形如(samples, sequence_length, output_dim)的3D张量