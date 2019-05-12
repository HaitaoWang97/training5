#!/usr/bin/env python
# coding: utf-8

# In[1]:


import d2lzh as d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss 
import time

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


# In[2]:


nd.one_hot(nd.array([0, 2]), vocab_size)


# In[3]:


def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape


# In[4]:


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
#隐藏层    
    w_xh = _one((num_inputs, num_hiddens))
    w_hh = _one((num_inputs, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    #输出层
    w_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    #创建梯度
    params = [w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


# In[5]:


#定义初始模型，初始化隐藏输入为0
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


# In[10]:


#定义rnn函数，确定计算过程
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


# In[11]:


state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X, vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
len(outputs), outputs[0].shape, state_new[0].shape


# In[13]:


#定义预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
               num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


# In[14]:


predict_rnn('分开'，10, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx)


# In[15]:


#裁剪梯度
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            


# In[16]:


#定义训练模型
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                         vocab_size, ctx, corpus_indices, idx_to_char,
                         char_to_idx, is_random_iter, num_epochs, num_steps,
                         lr, clipping_theta, batch_size, pred_period, pred_len,
                          prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()
    
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X,Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                (outputs, state) = rnn(inputs, state,params)
                outputs = nd.concat(*outputs, dim=0)
                y = Y.T.reshape((-1,))
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            d2l.sgd(params, lr, l)
            l_sum += l.asscalar()* y.size
            n += y.size
        
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
            epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print('-', predict_rnn(
                prefix, pred_len, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
    


# In[17]:


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50,['分开'，'不分开']

train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                     vocab_size, ctx, corpus_indices, idx_to_char,
                     char_to_idx, True, num_epochs, num_steps, lr,
                     clipping_theta, batch_size, pred_period, pred_len,
                     prefixes)


# In[ ]:


train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                     vocab_size, ctx, corpus_indices, idx_to_char,
                     char_to_idx, False, num_epochs, num_steps,
                     pred_len, prefixes)

