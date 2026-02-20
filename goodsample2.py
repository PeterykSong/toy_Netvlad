import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

# Pittsburgh 데이터셋 루트 경로.
# 원본 NetVLAD 코드처럼 절대 경로를 고정해 두었기 때문에,
# 로컬 환경에 맞게 수정하지 않으면 바로 예외를 발생시킨다.
root_dir = '/nfs/ibrahimi/data/pittsburgh/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

# 구조체(.mat) 파일과 쿼리 이미지가 있는 하위 디렉터리
struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')

def input_transform():
    # ImageNet 사전학습 백본(VGG/ResNet 등)에 맞춘 표준 정규화.
    # PIL 이미지를 Tensor(C,H,W)로 변환 후 채널별 normalize 한다.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_training_set(onlyDB=False):
    # 학습 split 전체(데이터베이스 + 쿼리) 반환.
    # onlyDB=True면 데이터베이스 이미지(numDb)만 포함한다.
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_val_set():
    # 검증 split 전체(30k) 반환.
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_val_set():
    # 검증 split 전체(250k) 반환.
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())
def get_whole_test_set():
    # 테스트 split 전체(30k) 반환.
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_test_set():
    # 테스트 split 전체(250k) 반환.
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1):
    # Triplet/Ranking 학습용 쿼리 데이터셋 반환.
    # margin은 hard negative 판별 기준(dNeg < dPos + sqrt(margin))에 사용된다.
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(), margin=margin)

def get_val_query_set():
    # 검증용 쿼리 데이터셋(30k) 반환.
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_val_query_set():
    # 검증용 쿼리 데이터셋(250k) 반환.
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

# MATLAB의 dbStruct를 Python에서 다루기 쉬운 형태로 정의.
dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    # .mat 파일의 dbStruct를 로드해 namedtuple로 변환한다.
    # 필드 의미:
    # - dbImage/qImage: 상대 경로 이미지 리스트
    # - utmDb/utmQ: GPS/UTM 좌표 (근접성 기반 positive/negative 판정에 사용)
    # - posDistThr: 평가 시 positive로 간주할 거리 임계값
    # - nonTrivPosDistSqThr: 학습 시 non-trivial positive 후보 거리^2 임계값
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        # DB 이미지는 root_dir 기준 상대 경로.
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            # 평가/추론 편의를 위해 query 이미지도 같은 배열 뒤에 붙인다.
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        # getPositives()를 호출할 때 lazy 계산/캐시한다.
        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        # PIL 로드 -> optional transform -> (tensor, index) 반환.
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # 평가(Recall@N)용 positive는 posDistThr 반경 안의 DB 이미지들.
        # 쿼리 좌표(utmQ) 각각에 대해 반경 이웃 검색을 수행한다.
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            # self.positives[i] = i번째 query의 positive DB 인덱스 배열
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
        
def collate_fn(batch):
    """학습용 샘플(query, positive, negatives)을 배치 텐서로 묶는다.
    
    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    # __getitem__에서 hard negative가 없으면 None을 반환할 수 있으므로 제거.
    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    # query/positive는 일반 collate로 (B, C, H, W)로 적재.
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    # 샘플별 negative 개수(가변 길이)를 따로 저장.
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    # negatives는 하나의 텐서로 이어붙여 (sum(n_i), C, H, W) 형태로 관리.
    negatives = torch.cat(negatives, 0)
    import itertools
    # [query_idx, pos_idx, neg_idx...] 메타 인덱스도 1차원으로 flatten.
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        # non-trivial positive 후보:
        # query 주변의 매우 쉬운 positive(너무 가까움)만 쓰지 않기 위해
        # nonTrivPosDistSqThr 반경 이웃을 positive 후보로 둔다.
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        # radius_neighbors는 정렬을 보장하지 않으므로 미리 정렬해 두어 재현성/일관성 확보.
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # positive 후보가 하나도 없는 query는 학습 불가하므로 제외.
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives:
        # posDistThr 바깥(DB 기준 충분히 먼 샘플)만 negative 후보로 유지.
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        # 이미지 임베딩을 미리 저장한 HDF5 캐시 파일 경로.
        # 외부 학습 루프에서 self.cache에 실제 경로를 주입해야 __getitem__이 동작한다.
        self.cache = None

        # query별 최근 hard negative를 저장해 다음 iteration에 재사용(마이닝 안정화).
        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        # 필터링된 query 인덱스를 원본 dbStruct query 인덱스로 복원.
        index = self.queries[index]
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            # features 배열은 [db features..., query features...] 순서라고 가정.
            qOffset = self.dbStruct.numDb 
            qFeat = h5feat[index+qOffset]

            # 해당 query의 non-trivial positive 후보 중 feature 거리상 가장 가까운 1개를 고른다.
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            # negative 후보에서 랜덤 샘플링 + 이전 hard negative 캐시를 합쳐 탐색 풀 구성.
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[list(map(int, negSample))]
            knn.fit(negFeat)

            # negative 최근접 후보를 넉넉히(10배수) 찾은 뒤 margin 위반 샘플만 채택.
            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), 
                    self.nNeg*10) # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # margin 위반 조건:
            # dNeg < dPos + sqrt(margin) 인 hard negative만 유지.
            violatingNeg = dNeg < dPos + self.margin**0.5
     
            if np.sum(violatingNeg) < 1:
                # 학습 신호가 없는 query는 배치에서 제외(None)한다.
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            # 다음 샘플링 때 재활용할 hard negative 캐시 갱신.
            self.negCache[index] = negIndices

        # 실제 이미지 로드(positive/negative는 DB에서, query는 queries_real에서).
        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(root_dir, self.dbStruct.dbImage[posIndex]))

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(root_dir, self.dbStruct.dbImage[negIndex]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        # negative 이미지를 (nNeg, C, H, W) 텐서로 스택.
        negatives = torch.stack(negatives, 0)

        # 마지막 반환값은 추적/디버깅용 인덱스 메타정보.
        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)
