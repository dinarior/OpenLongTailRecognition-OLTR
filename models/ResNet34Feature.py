from models.ResNetFeature import *
from utils import *
        
def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, caffe=False, test=False):
    
    print('Loading Scratch ResNet 50 Feature Model.')
    resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)
    
    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 34 Weights.' % dataset)
            resnet34 = init_weights(model=resnet34,
                                     weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet34