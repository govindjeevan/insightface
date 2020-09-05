import sys
import os
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import symbol_utils

sys.path.append(os.path.join('..'))
from config import config


# Helpers
def dense_block(num_layers, num_feature_maps, stage_index, transition=None):
    out = nn.HybridSequential(prefix='stage%d_' % stage_index)
    with out.name_scope():
        for _ in range(num_layers):
            out.add(dense_layer(num_feature_maps))
    
    if transition == "up":
        out.add(transition_up(num_feature_maps))
    elif transition == "down":
        out.add(transition_down(num_feature_maps))
        
    return out


def dense_layer(feature_maps):
    
    dense = gluon.nn.HybridSequential()
    dense.add(nn.BatchNorm())
    dense.add(nn.LeakyReLU(0.3))
    dense.add(nn.Conv2D(feature_maps, kernel_size=3, padding=1, use_bias=False))
    dense.add(nn.Dropout(0.2))
    
    skip = gluon.contrib.nn.HybridConcurrent(1, prefix='')
    skip.add(gluon.contrib.nn.Identity())
    skip.add(dense)
    
    return skip

def transition_down(feature_maps):
    out = nn.HybridSequential(prefix='')
    out.add(nn.BatchNorm())
    out.add(nn.LeakyReLU(0.3))
    out.add(nn.Conv2D(feature_maps, kernel_size=1, use_bias=False))
    out.add(nn.Dropout(0.2))
    out.add(nn.MaxPool2D(pool_size=2))
    return out


def transition_up(feature_maps):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2DTranspose(feature_maps, kernel_size=6, padding=2, strides=2, use_bias=False))
    return out


class AttentionBlock(nn.HybridBlock):

    def __init__(self, num_feature_map, **kwargs):

        super(AttentionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.mean_block =  gluon.nn.HybridSequential()   
            self.mean_block.add(nn.AvgPool2D())
            self.mean_block.add(nn.Conv2D(num_feature_map, kernel_size=1, use_bias=False))
            self.mean_block.add(nn.LeakyReLU(0.3))
            self.mean_block.add(nn.Conv2D(num_feature_map, kernel_size=1, use_bias=False))

            self.max_block =  gluon.nn.HybridSequential()
            self.max_block.add(nn.MaxPool2D())
            self.max_block.add(nn.Conv2D(num_feature_map, kernel_size=1, use_bias=False))
            self.max_block.add(nn.LeakyReLU(0.3))
            self.max_block.add(nn.Conv2D(num_feature_map, kernel_size=1, use_bias=False))
    
    def hybrid_forward(self, F, x):
        mean_block_out = self.mean_block(x)
        max_block_out = self.max_block(x)
        
        return F.Activation(F.concat(mean_block_out, max_block_out,dim=2), 'sigmoid')
    
    
    
class DenseNet(nn.HybridBlock):

    def __init__(self, num_feature_map, **kwargs):

        super(DenseNet, self).__init__(**kwargs)
        with self.name_scope():
            self.block1 = gluon.nn.HybridSequential()
            self.block1.add(gluon.nn.Conv2D(channels=16, kernel_size=9, strides=1, use_bias=False))
            self.block1.add(dense_block(4, num_feature_map,1))

            self.block2 = gluon.nn.HybridSequential()
            self.block2.add(transition_down(num_feature_map))
            self.block2.add(dense_block(5, num_feature_map,2))

            self.block3 = gluon.nn.HybridSequential()
            self.block3.add(transition_down(num_feature_map))
            self.block3.add(dense_block(7, num_feature_map,3))

            self.block4 = gluon.nn.HybridSequential()
            self.block4.add(transition_down(num_feature_map))
            self.block4.add(dense_block(10, num_feature_map,4))
            self.block4.add(transition_up(128))
            
            self.block5 = gluon.nn.HybridSequential()
            self.block5.add(dense_block(7, num_feature_map,6))
            self.block5.add(transition_up(96))

            self.block6 = gluon.nn.HybridSequential()
            self.block6.add(dense_block(5, num_feature_map,7))
            self.block6.add(transition_up(80))
            
            self.block7 = gluon.nn.HybridSequential()
            self.block7.add(dense_block(4, num_feature_map,8))
            self.block7.add(nn.BatchNorm())
            self.block7.add(nn.LeakyReLU(0.3))

            self.cam1 = AttentionBlock(128)
            self.cam2 = AttentionBlock(96)
            self.cam3 = AttentionBlock(80)
            
    def hybrid_forward(self, F, x):
        
        b1 = self.block1(x)
        
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        
        concat1 = F.concat(b4, b3, dim=3)
        cam1_out = self.cam1(concat1)
        b5 = self.block5(cam1_out*b4+b3)
        
        concat2 = F.concat(b5, b2, dim=3)
        cam2_out = self.cam2(concat2)
        b6 = self.block6(cam2_out*b5+b2)
        
        concat3 = F.concat(b6, b1, dim=3)
        cam3_out = self.cam3(concat3)
        b7 = self.block7(cam3_out*b6+b1)
        
        return b7
def get_symbol():

    net = DenseNet(16)
    data = mx.sym.Variable(name='data')
    data = data - 127.5
    data = data * 0.0078125
    body = net(data)
    fc1 = symbol_utils.get_fc1(body, config.emb_size, config.net_output)
    return fc1
