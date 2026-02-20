
# 코드리뷰 Main.py

이 파일은 https://github.com/Nanne/pytorch-NetVlad 의 구현 코드를 상세히 리뷰하기 위해 작성하였습니다.   


# Header 

```python
from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
```
- `from __future__ import print_function`  
이는 python 버전이 달라져도 동일한 기능을 할 수 있도록 선언한다.  
통상은 Python 버전 2와 3간의 호환성을 위해 선언한다고 한다. 
공식문서는 다음과 같으며,  
https://docs.python.org/ko/3.13/library/__future__.html  
더 도움되는 참고자료는 아래 주소가 유익하다.   
https://chasuyeon.tistory.com/entry/Python-from-future-import-printfunction  


그 외에는 python에서 자주 사용되는 라이브러리들을 미리 선언한다. 
수학 관련/난수생성/파일,경로관리에 사용되는 라이브러리들이다. 


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss
```

- `import torch`  
torch library를 불러온다. 

- `import torch.nn as nn`  / `import torch.nn.functional as F`  
네트워크를 구성하고 편집하기 위한 클래스다.  
torch API문서 : https://docs.pytorch.org/docs/stable/nn.html  
상세 설명 페이지 : https://tutorials.pytorch.kr/beginner/nn_tutorial.html  

- `import torch.optim as optim`  
Optimizer 설정을 위한 유틸리티 클래스.  
상세 설명 페이지 : https://wikidocs.net/194971  


- `from torch.autograd import Variable`
자동미분을 적용하여, 역전파/순전파에 활용하기 위함.  
torch API 문서 : https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html  

- `from torch.utils.data import DataLoader, SubsetRandomSampler`  
공식 API 문서 : https://docs.pytorch.org/docs/stable/data.html  
Data Loader : 데이터셋을 읽어와서 배치단위로 데이터를 불러오는 작업을 진행.   
참고자료 : https://hi-guten-tag.tistory.com/345  
SubsetRandomSampler : DataLoader의 사용시 파라미터에 shuffle을 지정할 수 있는데,   
`shuffle = True`로 할 경우 dataloader는 자동으로 RandomSampler를 선택한다.  
그러나, 일부 Subset에서 random하게 뽑고 싶다면 `SubsetRandomSampler` 를 사용해야 하며, 이때 `shuffle = false` 로 선언해야 한다.   
이 경우 db 데이터와 query 데이터가 나누어져 있으므로 사용한다.   
참고자료 : https://chickencat-jjanga.tistory.com/135#google_vignette 

- `from torch.utils.data.dataset import Subset`  
데이터셋을 쉽게 관리하기 위해 사용한다.   
참고자료 : https://yeko90.tistory.com/entry/pytorch-how-to-use-Subset

- `import torchvision.transforms as transforms`
이미지 데이터를 편집하기 위해 사용한다. 보통은 편집 후 텐서로 변환하여 데이터셋에 넣는다.   
공식 API문서 : https://docs.pytorch.org/vision/0.9/transforms.html  

- `import torchvision.datasets as datasets`   
이미지용 api인 torchvision에서 제공하는 dataset 유틸리티다.  
공식 API문서 : https://docs.pytorch.org/vision/main/datasets.html  

- `import torchvision.models as models`  
Pretrained 모델을 불러올때 사용하는데, 이경우에는 VGG16을 불러와야 하기때문에 사용한다.   
공식 API문서 : https://docs.pytorch.org/vision/0.9/models.html  

- `import h5py`
Train중 결과 데이터와 가중치들을 저장하기 위해 HDF5( Hierarchical Data Format) 데이터포맷을 사용하기 위해 가져온다.   
이 모듈을 통해 대량의 데이터를 쉽게 관리할 수 있다.  
공식 문서 : https://docs.h5py.org/en/latest/index.html  
참고 문서 : https://bo-10000.tistory.com/108  

- `import faiss`  
벡터들의 검색/유사도 연산에서 사용하는 툴. nearest neighber나, k-nn, cos-similarity 와 같은 연산을 수행하는 유틸리티이다. Faiss is a library for efficient similarity search and clustering of dense vectors. 

공식 git : https://github.com/facebookresearch/faiss/wiki



```python 
from tensorboardX import SummaryWriter
import numpy as np
import netvlad
```
- `from tensorboardX import SummaryWriter` 
텐서보드를 사용하기 위해 불러온다. 시각화툴이다. 있으면 좋은 것. 

- `import netvlad` 
여기서는 CNN+VLAD layer의 구성을 netvlad.py에서 선언한다.   
따라서, 해당 파일내의 함수들을 사용하기 위해 선언한다.  



# Run parameter

```python
parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--batchSize', type=int, default=4, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='/nfs/ibrahimi/data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='/nfs/ibrahimi/runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default=environ['TMPDIR'], help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='pittsburgh', 
        help='Dataset to use', choices=['pittsburgh'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
```

main.py가 실행될때 사용되는 옵션 파라미터들. 키워드들을 중심으로 보면, 이 코드가 어떤 부분에서 어떻게 튜닝할수 있도록 준비되어있는지 알수 있다.     



# Main
잠시 중간을 건너뛰고, main부분만 먼저 보도록 한다.   
사실 전문을 모두 리뷰하고자 하는건 아니나, 최대한 많은 부분들을 세세히 들여다보고자 한다. 

```python
if __name__ == "__main__":
    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'seed', 'patience']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if opt.dataset.lower() == 'pittsburgh':
        import pittsburgh as dataset
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

        train_set = dataset.get_training_query_set(opt.margin)

        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'test250k':
            whole_test_set = dataset.get_250k_test_set()
            print('===> Evaluating on test250k set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print('===> Building model')

    pretrained = not opt.fromscratch
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
            if not opt.resume: 
                if opt.mode.lower() == 'train':
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5: 
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs) 
                    del clsts, traindescs

            model.add_module('pool', net_vlad)
        elif opt.pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)
    
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = test(whole_test_set, epoch, write_tboard=False)
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score 
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else: 
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
```

  
  
  
전문을 천천히 따라가면서, 세부적으로 내용을 파악해보자. 

```python
    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'seed', 'patience']
```
위 코드는 bash 에서 main.py를 실행할때 기입한 옵션 파라미터들을 저장하는 부분이다. 

```python
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)
```
이중 resume옵션일 경우 핸들링하는 부분인데, 주요 flag들을 불러와서 학습을 재게한다.  
하단부에 'train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]` 가 있는 것으로 보아, train중에 뭔가 뭠춰야 할 경우가 자주 발생했던듯 싶다.  
현 시점에서는 굳이 이런 핸들링까지 필요하지는 않을 것으로 예상해본다.   

```python
    if opt.dataset.lower() == 'pittsburgh':
        import pittsburgh as dataset
    else:
        raise Exception('Unknown dataset')
```

데이터셋이 Pittsburgh인 경우 pittsburg.py를 불러온다.   
대소문자 실수를 방지하기 위해 문자열을 전부 lower로 바꾸었는데, 이런 배려는 좋아보인다.  

```python
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
```
No Cuda, No Torch.   
아니, 뭐 꼭 그런건 아니고...   


```python 
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
```
이 부분도 학습관련 long-term 프로젝트를 하게 되면 유용한 부분이다.  
Random함수를 사용하게 되면 매 실행마다 random값이 바뀌게 되면서 학습의 안정성을 헤치는 결과가 나올 수 있다. 따라서, Random으로 숫자를 선정하지만, seed가 주어지면, 동일한 seed에서는 random함수의 결과가 동일하게 나오도록 할 수 있다. 이를 통해 재현가능성을 올릴 수 있다.   

참고자료 : https://m.blog.naver.com/regenesis90/222363064500

여기까지 해서, 기본적으로 실행될때의 Option처리는 끝난다.   
물론 다른 옵션들도 있지만, 이는 Data load와 관련된 것들이므로 바로 이어서 알아보도록 한다. 

```python
    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

        train_set = dataset.get_training_query_set(opt.margin)

        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'test250k':
            whole_test_set = dataset.get_250k_test_set()
            print('===> Evaluating on test250k set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)
```
Data를 loading 하는 단계이다.  
Option에서 어떤 파라미터를 썼는가에 따라, test용이냐 아니면 대형 테스트용인가, 또는트레이닝용인가에 대한 부분들을 정의한다.   

1. training   

만약 Training 이라면, 
 `whole_train_set` 을 사용하며, `get_whole_training_set()`를 통해 데이터셋을 불러온다.  
 DataLoader는 다음과 같이 실행된다. 

 ```python
         whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)
 ```
`train_set = dataset.get_training_query_set(opt.margin)` 부분은 이 논문의 특징으로 발생한다.  
db이미지와 query이미지를 나누어 데이터셋을 구성한다.   
데이터셋 구성이나 논문에서도 나오지만, 실행 초기에 네트워크를 학습할때는 db이미지와 query이미지를 섞어서 학습을 하지만, 이 feature 벡터들을 이용해해서 triplet을 구성할때에는 query이미지는 anchor로 활용되고, db 이미지는 positive나 negative로 활용되어 학습을 진행한다.   
따라서, Query이미지 세트도 한세트 필요하다. 이건 데이터셋 선언하는 부분에서 좀 더 자세히 보도록 하자.  


저자의 readme.md 파일에서 실행 예제를 보면 다음과 같이 실행된다. 
```bash
python main.py --mode=train --arch=vgg16 --pooling=netvlad --num_clusters=64
```

2. test  

만약 test라면, 
`whole_test_set` 을 사용하며, `get_whole_test_set()`를 통해 데이터셋을 얻는다.   
그 이하의 내용들은 어떤 데이터셋인가에 따라 테스트용 데이터를 불러오는 과정을 거친다. 
test에서부터 val 까지 다양하게 있는걸 볼 수 있다. 마찬가지로 readme.md에 있는 사용예를 보자. 

```bash
    python main.py --mode=test --resume=runsPath/Nov19_12-00-00_vgg16_netvlad --split=test
```


그 다음엔 Pre-train된 모델을 가져온다. option명에선 `--arch` 로 되어있다. 

(26. 2. 21 ) 현재 구현해보고자 하는 것은 DataLoader까지이므로, main은 여기에서 일단 리뷰를 멈춘다. 