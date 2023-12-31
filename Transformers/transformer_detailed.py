#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:57:46 2023

@author: soumensmacbookair
"""

# Import the libraries
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#%% Build transformer
class TransformerModel(tf.keras.Model):

    def __init__(self, num_encoding_layer = 6,
                 num_decoding_layer = 6, embedding_size = 512, num_heads = 6,
                 encoder_vocab_size = 10000, decoder_vocab_size = 10000):
        """
        Transformer consists of (x_enc, x_dec) -> encoder layer + decoder layer -> linear layer -> y
        """

        super(TransformerModel, self).__init__()
        self.num_encoding_layer = num_encoding_layer
        self.num_decoding_layer = num_decoding_layer
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size

        self.encoder_layer = TransformerEncoderLayer(num_encoding_layer=self.num_encoding_layer,
                                                     embedding_size=self.embedding_size,
                                                     num_heads=self.num_heads,
                                                     encoder_vocab_size=self.encoder_vocab_size)

        self.decoder_layer = TransformerDecoderLayer(num_decoding_layer=self.num_decoding_layer,
                                                     embedding_size=self.embedding_size,
                                                     num_heads=self.num_heads,
                                                     decoder_vocab_size=self.decoder_vocab_size)

        self.linear_layer = tf.keras.layers.Dense(units=self.decoder_vocab_size,
                                                  activation="linear") # returns logits

    def call(self, xenc_xdec_tuple):
        """
        input xenc_xdec_tuple must be (x_enc, x_dec)
        x_enc.shape = (batch_size, seq_length)
        x_dec.shape = (batch_size, seq_length)
        output y.shape = (batch_size, seq_length, decoder_vocab_size)
        """

        x_enc, x_dec = xenc_xdec_tuple
        z = self.encoder_layer(x_enc)
        y = self.decoder_layer(xz_tuple=(x_dec, z))
        y = self.linear_layer(y)

        return y

class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_encoding_layer, embedding_size, num_heads, encoder_vocab_size):
        """
        Encoder consists of x -> embedding layer -> L sequential encoding layer -> y
        """

        super(TransformerEncoderLayer, self).__init__()
        self.num_encoding_layer = num_encoding_layer
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.encoder_vocab_size=encoder_vocab_size

        self.embedding_layer = EmbeddingLayer(embedding_size=self.embedding_size,
                                              vocab_size=self.encoder_vocab_size)

        self.encoding_layers = [EncodingLayer(embedding_size=self.embedding_size,
                                              num_heads=self.num_heads)] * self.num_encoding_layer

    def call(self, x):
        """
        input x.shape = (batch_size, seq_length)
        output y.shape = (batch_size, seq_length, embedding_size)
        """

        y = self.embedding_layer(x)
        for encoding_layer in self.encoding_layers:
            y = encoding_layer(y)

        return y

class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_decoding_layer, embedding_size, num_heads, decoder_vocab_size):
        """
        Decoder consists of (x, z) -> embedding layer -> L sequential decoding layer -> y
        """

        super(TransformerDecoderLayer, self).__init__()
        self.num_decoding_layer = num_decoding_layer
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.decoder_vocab_size=decoder_vocab_size

        self.embedding_layer = EmbeddingLayer(embedding_size=self.embedding_size,
                                              vocab_size=self.decoder_vocab_size)

        self.decoding_layers = [DecodingLayer(embedding_size=self.embedding_size,
                                              num_heads=self.num_heads)] * self.num_decoding_layer

    def call(self, xz_tuple):
        """
        input xz_tuple must be (x, z)
        x.shape = (batch_size, seq_length)
        z.shape = (batch_size, seq_length, embedding_size)
        output y.shape = (batch_size, seq_length, embedding_size)
        """

        x = xz_tuple[0]
        z = xz_tuple[1]
        y = self.embedding_layer(x)
        for decoding_layer in self.decoding_layers:
            y = decoding_layer(xz_tuple=(y, z))

        return y

class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, embedding_size, vocab_size):
        """
        EmbeddingLayer consists of x -> word embedding layer -> positional embedding layer -> y
        """

        super(EmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.word_embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                              output_dim=self.embedding_size,
                                                              mask_zero=True) # mask for zero padding is propagated to all layer

    @staticmethod
    def positional_encoding(sequence_length, embedding_depth):
        """
        positional encoding is fixed, could be cached, output shape (seq_length, embedding_size)
        """

        embedding_depth = embedding_depth/2
        positions = np.arange(sequence_length)[:, np.newaxis] # equivalent to reshaping to (sequence_length, 1)
        depths = np.arange(embedding_depth)[np.newaxis, :]/embedding_depth # equivalent to reshaping to (1, embedding_depth)

        angle_rates = 1 / (10000**depths) # (1, embedding_depth)
        angle_rads = positions * angle_rates # (sequence_length, embedding_depth) [both the vectors are broadcasted to become a matrix]

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        """
        input x.shape = (batch_size, seq_length)
        output y.shape = (batch_size, seq_length, embedding_size)
        """

        y1 = self.word_embedding_layer(x)
        y = y1 + EmbeddingLayer.positional_encoding(sequence_length=x.shape[-1],
                                                    embedding_depth=self.embedding_size) # broadcasted to batch size

        return y

class EncodingLayer(tf.keras.layers.Layer):

    def __init__(self, embedding_size, num_heads):
        """
        EncodingLayer consists of
        x -> multihead non causal self attention layer -> feed forward layer -> y
        """

        super(EncodingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.mhnc_self_attention_layer = MultiheadNonCausalSelfAttentionLayer(embedding_size=self.embedding_size,
                                                                              num_heads=self.num_heads)
        self.feed_forward_layer = FeedForwardLayer(embedding_size=self.embedding_size)

    def call(self, x):
        """
        input x.shape = (batch_size, seq_length, embedding_size)
        output y.shape = x.shape
        """

        y = self.mhnc_self_attention_layer(x)
        y = self.feed_forward_layer(y)

        return y

class DecodingLayer(tf.keras.layers.Layer):

    def __init__(self, embedding_size, num_heads):
        """
        DecodingLayer consists of
        x -> multihead causal self attention layer -> multihead cross attention layer -> feed forward layer -> y
        """

        super(DecodingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.mhc_self_attention_layer = MultiheadCausalSelfAttentionLayer(embedding_size=self.embedding_size,
                                                                          num_heads=self.num_heads)

        self.mh_cross_attention_layer = MultiheadCrossAttentionLayer(embedding_size=self.embedding_size,
                                                                     num_heads=self.num_heads)

        self.feed_forward_layer = FeedForwardLayer(embedding_size=self.embedding_size)

    def call(self, xz_tuple):
        """
        input xz_tuple must be (x, z)
        x.shape = (batch_size, seq_length, embedding_size)
        z.shape = (batch_size, seq_length, embedding_size)
        output y.shape = (batch_size, seq_length, embedding_size)
        """

        x = xz_tuple[0]
        z = xz_tuple[1]

        y = self.mhc_self_attention_layer(x)
        y = self.mh_cross_attention_layer(xz_tuple=(y, z))
        y = self.feed_forward_layer(y)

        return y

class FeedForwardLayer(tf.keras.layers.Layer):

    def __init__(self, embedding_size):
        """
        FeedForwardLayer consists of
        x -> Dense of 4 * embedding_size neurons -> Linear of embedding_size neurons
        -> add layer -> norm layer -> y
        """

        super(FeedForwardLayer, self).__init__()
        self.embedding_size = embedding_size
        self.dense_layer = tf.keras.layers.Dense(units=4 * self.embedding_size,
                                                 activation="relu",
                                                 kernel_initializer="he_normal")

        self.linear_layer = tf.keras.layers.Dense(units=1 * self.embedding_size,
                                                  activation="linear",
                                                  kernel_initializer="he_normal")

        self.add_layer = tf.keras.layers.Add()
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x):
        """
        input x.shape = (batch_size, seq_length, embedding_size)
        output y.shape = x.shape
        """

        y1 = self.dense_layer(x)
        y1 = self.linear_layer(y1)
        y = self.add_layer([x, y1])
        y = self.norm_layer(y)

        return y

class MultiHeadAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, embedding_size, num_heads, isCausal):
        """
        MultiheadAttentionLayer consists of
        x -> attention layer -> y
        """

        super(MultiHeadAttentionLayer, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.key_dim = int(self.embedding_size / self.num_heads)
        self.value_dim = self.key_dim
        self.isCausal = isCausal

        self.attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                                  key_dim=self.key_dim,
                                                                  value_dim=self.value_dim) # attention mask is automatically taken

    def call(self, x_query, x_key, x_value):
        """
        input x.shape = (batch_size, seq_length, embedding_size)
        output y.shape = x.shape
        """

        y = self.attention_layer(query=x_query,
                                 key=x_key,
                                 value=x_value,
                                 return_attention_scores=False,
                                 use_causal_mask=self.isCausal)

        return y

class MultiheadNonCausalSelfAttentionLayer(MultiHeadAttentionLayer):

    def __init__(self, embedding_size, num_heads):
        """
        MultiheadNonCausalSelfAttentionLayer consists of
        x -> attention layer -> add layer -> norm layer -> y
        """

        super(MultiheadNonCausalSelfAttentionLayer, self).__init__(embedding_size=embedding_size,
                                                                   num_heads=num_heads,
                                                                   isCausal=False)
        self.add_layer = tf.keras.layers.Add()
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x):
        """
        input x.shape = (batch_size, seq_length, embedding_size)
        output y.shape = x.shape
        """

        y1 = super(MultiheadNonCausalSelfAttentionLayer, self).call(x_query=x,
                                                                    x_key=x,
                                                                    x_value=x)
        y = self.add_layer([x, y1])
        y = self.norm_layer(y)

        return y

class MultiheadCausalSelfAttentionLayer(MultiHeadAttentionLayer):

    def __init__(self, embedding_size, num_heads):
        """
        MultiheadCausalSelfAttentionLayer consists of
        x -> attention layer -> add layer -> norm layer -> y
        """

        super(MultiheadCausalSelfAttentionLayer, self).__init__(embedding_size=embedding_size,
                                                                num_heads=num_heads,
                                                                isCausal=True)
        self.add_layer = tf.keras.layers.Add()
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x):
        """
        input x.shape = (batch_size, seq_length, embedding_size)
        output y.shape = x.shape
        """

        y1 = super(MultiheadCausalSelfAttentionLayer, self).call(x_query=x,
                                                                 x_key=x,
                                                                 x_value=x)
        y = self.add_layer([x, y1])
        y = self.norm_layer(y)

        return y

class MultiheadCrossAttentionLayer(MultiHeadAttentionLayer):

    def __init__(self, embedding_size, num_heads):
        """
        MultiheadCrossAttentionLayer consists of
        x -> attention layer -> add layer -> norm layer -> y
        """

        super(MultiheadCrossAttentionLayer, self).__init__(embedding_size=embedding_size,
                                                           num_heads=num_heads,
                                                           isCausal=False)
        self.add_layer = tf.keras.layers.Add()
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, xz_tuple):
        """
        input xz_tuple must be (x, z)
        x.shape = (batch_size, seq_length, embedding_size)
        z.shape = (batch_size, seq_length, embedding_size)
        output y.shape = (batch_size, seq_length, embedding_size)
        """

        x = xz_tuple[0]
        z = xz_tuple[1]

        y1 = super(MultiheadCrossAttentionLayer, self).call(x_query=x,
                                                            x_key=z,
                                                            x_value=z)
        y = self.add_layer([x, y1])
        y = self.norm_layer(y)

        return y

#%% Test the transformer model
batch_size = 64
seq_length = 128

transformer_model = TransformerModel()

x_enc = tf.random.uniform(shape=(batch_size, seq_length), minval=0, maxval=10000, dtype=tf.int64)
x_dec = tf.random.uniform(shape=(batch_size, seq_length), minval=0, maxval=10000, dtype=tf.int64)

y_dec = transformer_model((x_enc, x_dec))
transformer_model.summary()











