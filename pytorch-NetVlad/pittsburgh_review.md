# Pittsburgh Dataset 설정 

이 파일은 pittsburgh.py의 코드 리뷰입니다.   
main.py의 실행 옵션에 따라 main 실행시 불러집니다.   


# Header
주요 라이브러리들을 불러들인다.   
헤더는 총 3부분으로 되어있다. 

```python
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
```
- `import torch`   
torch 를 불러오는 부분  

- `import torchvision.transforms as transforms`
데이터의 전처리를 위해 사용된다.   
공식 API문서 : https://docs.pytorch.org/vision/0.9/transforms.html    

- `import torch.utils.data as data` 
데이터셋을 만들기 위한 유틸리티 모듈. 대표적을 DataLoader과 Dataset이 여기 들어가있다.   
공식 튜토리얼 문서 : https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html  


```python
from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
```
파이썬에서 파일을 다루기 위해 흔하게 사용되는 라이브러리들이다.

- `from os.path import join, exists`  
경로 편집을 위한 모듈  

- `from scipy.io import loadmat`  
MATLab 데이터를 불러들이기 위한 유틸리티  

넘파이는 생략하고. 

- `from collections import namedtuple`  
조금 생경한 부분인데, collections에서 정렬관련 모듈들은 제법 본 적 있는데, namedtuple은 처음본다.  
data들이 튜플의 형태로 저장되는데, 이를 좀 더 쉽게 핸들링하기 위한 것이라고 이해하자.   
참고자료 : https://wikidocs.net/104956  

- `from PIL import Image`  
Pillow 라이브러리. 이미지의 필셀단위 조작, 마스킹, 투명도 제어나 색상 보정 등 여러가지 작업을 한다.  
참고자료 : https://wikidocs.net/153080  

```python
from sklearn.neighbors import NearestNeighbors
import h5py
```
근접 데이터 검색을 위한 라이브러리와, 
main.py에도 나왔던 대용량 캐피파일 HDF5를 핸들링하기 위한 라이버러리다. 

공식 API(sklearn) : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  
공식 문서(HDF5) : https://docs.h5py.org/en/latest/index.html  
참고 문서(HDF5) : https://bo-10000.tistory.com/108  

# Path
데이터셋이 어디 저장되어있는지 명시해야 한다. 

```python 
root_dir = '/nfs/ibrahimi/data/pittsburgh/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')
struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')
```
- `root_dir = '/nfs/ibrahimi/data/pittsburgh/'`  
root까지만 끊어서 저장하는게 여러모로 편하다.   
이후의 데이터들은 데이터셋 폴더의 실제 경로와는 제법 다르니, 유의해서 수정해야 한다.   

- `struct_dir = join(root_dir, 'datasets/')`  
데이터셋의 주요 내용이 있는 파일을 불러와야 한다. 이미지 경로, 좌표 등. 매트랩 파일로 되어있다.   

# Functions.


## input_transform() 
```python
def input_transform():
    return transforms.Compose([
        transfo리rms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
```
보통은 그냥transform이라고 많이 쓰는데, 헷갈리지 않으려고 이름을 좀 구분한 것 같다.  
dataset을 DataLoader를 통해 불러들일때 들어가는 파라미터들 중 transform이 있는데, 여기에 사용한다.  
나중에 이야기하겠지만, VGG 모델을 사용하지만 VGG모델에서 이미지분류를 할때 이미지를 224 x 224로 리사이즈하는 것에 반해, 여기선 리사이즈 없이 그냥 집어 넣는다. 

참고로 리사이즈의 공식 API문서는 다음과 같다.   
https://docs.pytorch.org/vision/main/generated/torchvision.transforms.Resize.html  

## get_whole_training_set()
데이터셋 구조가 들어있는 파일을 불러들여서,   
데이터셋 클래스를 반환하도록 되어있다. 

```python
def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)
```
- `structFile = join(struct_dir, 'pitts30k_train.mat')`  
데이터셋 구조 데이터의 경로를 불러온다. 여기엔 파일 리스트와 좌표 정보 등이 수록되어있다. 

- `WholeDatasetFromStruct(structFile,input_transform=input_transform(),onlyDB=onlyDB)`
뒤에 선언될 클래스의 프로토타입이다. Dataloader에 들어간다. main.py를 살펴보자.   

## dbStruct
데이터셋에 들어간 튜플의 구조를 선언한다. 앞서 헤더에서 언급했던 namedtuple을 사용한다. 

```python
dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])
```
- whichSet : 데이터셋의 종류. train, test, val 용으로 구분된다.
- dataset : 데이터셋의 이름. pittburgh250k인가 30k인가  등등이다. 
- dbImage : DB 이미지 파일 리스트. 경로로 되어있다. 
- utmDb : 각 DB 이미지파일의 좌표값. UTM으로 되어있어, 이스팅 xxx, 노딩 xxx로 표현되어있다.
- qImage : 쿼리 이미지 파일 리스트
- utmQ : 각 쿼리 이미지의 UTM 좌표값
- numDb : DB 이미지의 총 개수 len(dbImage)
- numQ : Query 이미지의 총 개수 len(qImage)
- posDistThr : positive이미지인지 판단하기 위한 GPS distace거리 기준값
- posDistSqThr : 위의 posDistThr값의 제곱. 빨리 갖다 쓰기 위해 만들었다. 
- nonTrivPosDistSqThr : 너무 쉬운 Positive만 고르지 않기 위해 정하는 limit값. (posDistSqThr< nonTrivPosDistSqThr)

마지막의 distance threshold값들은 training 이냐, test냐에 따라 사용되는 변수가 다르다. 추후에 알아보자. 

## parse_dbStruct(path)

```python
def parse_dbStruct(path):
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

```
mat 파일로 저장된 것들을 각각의 요소로 parsing 하는 코드다. 딱히 별건 없는데,   
중간중간 자료의 길이나 차원이 달라서 다소간의 핸들링이 필요하다.  



# WholeDatasetFromStruct(data.Dataset)

dataset의 형태를 선언하는 클래스다. 즉 이게 핵심이다.
datset은 세개의 함수로 구성해야 한다.(필수사항)
`__init__()`, `__getitem__()`,`__len__()` 이 그것이며,   
여기서는 triplet을 구성하기 위해 `getPositives()` 가 추가되어있다. 


## `__init__()`

```python
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None
```

- `def __init__(self, structFile, input_transform=None, onlyDB=False):`  
클래스를 선언하기 위해서는 mat파일에서 추출한 struct와 transform 형태를 선.....언하면 될줄 알았는데, 여기선 mat파일 자체를 넘긴다. 다시보내 내부에서 parsing한다.   
통상 tutorial들에서는 input_transform이라 안쓰고 그냥 transform이라고 쓰는 게 일반적이긴 한데 워딩을 혼동하고 싶지 않았던 듯 하다.   

- `super().__init__()`   
선언시 부모 클래스를 상속받기 위한 초기화.   
Class 정의에 있는 파라미터를 보면 `data.Dataset`이라는 클래스를 받아온다. 
따라서 없어도 동작에 무리는 없겠으나, 넣어두는것이 코딩룰에 가깝다.

- `self.input_transform = input_transform`
transform을 가져온다. 앞서서 input_transform 함수를 선언했는데, 그걸 가져온다. 

- `self.dbStruct = parse_dbStruct(structFile)`  
선언하면서 입력받은 경로의 mat파일을 불러 데이터셋을 불러온다.   
만약 커스텀 데이터셋을 쓰게 된다면 이 부분은 수정이 가해져야 할 것 같다. 

- `self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]`  
mat 파일의 경로는 000/000....jpg로 되어있어서 root경로가 빠져있다. (예 ./data/Pittsburgh250k)
이 두 문자열을 concat해줘야 한다. 

- `self.whichSet = self.dbStruct.whichSet`, `self.dataset = self.dbStruct.dataset`
무슨 데이터셋인지 정의된 내용을 가져온다. 이 값을 보고 이후 함수에서 분기하는데,   
단일 데이터셋만 사용하는 경우 굳이 필요있나 싶긴 하다.  

- `self.positives = None`, `self.distances = None`
보통 이 클래스가 선언되는 시점은 아직 네트워크 학습이 진행되지 않은 상태임은 물론,  
feature에 대한 추출이 한번도 시행되지 않은 시점이다.   
따라서, `None` 값이 저장된다. 

## __getitem__()
데이터셋에서 데이터를 가져올 쓴다. 
c++ 에선 클래스에 선언된 변수에 대해 모두 get함수를 만들어야 한다고 배웠던 것 같은데... 
여기선 해당 인덱스의 이미지만 불러서 인덱스와 함께 리턴하고 있다. 

```python
    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index
```

다른 인스턴스들에 대해 접근하고 싶다면 get함수를 만들어야 하나 싶지만, 
이미 `self.`으로 붙여놓았기 때문에 `dataset.***` 로 접근이 가능하다. 
파이썬은 이런 형태로 접근을 쉽게 해놓았다.

첨언. 만약 이걸 C++의 객체지향형 설계로 한다면. 어떻게 바꿔야 하나. 고민해보자. 

## __len__()
필수 3종 셋트에 마지막이다. 
이미지의 개수를 굳이 반환해야 할 일이 있을까 싶다가.... 아, 프로그레스바 만들려면 필요하겠구나 싶다.   

## getPositives(self):

```python
    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
```
이 논문은 triplet을 구성하는게 포인트다. 
그 triplet을 구하는데 필요한게 Positive와 negative인데, 이걸 찾기 위한 함수를 만들었다.  

- `if  self.positives is None:`
아직 Positive가 선정되지 않았다면. 

- `knn = NearestNeighbors(n_jobs=-1)` 
NN 서치 알고리즘을 선언한다. (class)  
Unsupervised learner for implementing neighbor searches.라고 되어있다. 이것도 학습 알고리즘이구나.. 
공식 API : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html   
`n_jobs = -1`은 모든 프로세서를 사용하여 병렬 연산하란 소리다.  ...... 

그러고보니 main.py에선 FAISS 를 사용했는데, 굳이 여기선 sklearn을 쓸 이유가 있나 싶다.

- `knn.fit(self.dbStruct.utmDb)`   
검색대상이 되는 객체를 등록하는단계다. DB이미지의 좌표값을 메모리에 올린다.   

- `self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,radius=self.dbStruct.posDistThr)`  
검색을 실시한다.  
radius_neighbors(X=None, radius=None, return_distance=True, sort_results=False)
함수의 결과는 neigh_distd와 neigh_index를 반환한다. 
입력된 파라미터를 보건데, utmQ를 넣고....Query를 넣네. triplet구성을 위한 수단이다. 
raidus 는 posDistThr를 사용한다. 

이제 positive를 리턴한다.   

```python
return self.positives
```

# collate_fn(batch)
클래스 밖에서 선언되었다. 
사용 위치를 찾아봤더니 main.py의 `def train()` 내부에서 사용된다.  
collate의 단어 뜻이 합친다는 뜻이다.  
학습하기 좋은 tensor형태로, 즉 (query이미지, db이미지 중 positive, db이미지중 negative)로 합치는 과정이다. 
따라서 출력값을 보면 `return query, positive, negatives, negCounts, indices`  로 되어있다.   

실행 구조 자체는 torch.utils.data에 있는 default_collate함수를 사용한다.  
상세한 참고자료는 이곳을 참조하면 도움이 된다.   
https://biomadscientist.tistory.com/170  

```python
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
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

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices
```

- `batch = list(filter (lambda x:x is not None, batch))`  
입력받은 batch에서 none을 삭제한다. 근데, none이 있을 수 있나...?  
참고자료 : https://wikidocs.net/22803 

- `query, positive, negatives, indices = zip(*batch)`  
배치를 요소에 맞게 분해한다.  
근데, 처음 실행하는데 배치가 있는가? 나중에 train함수를 더 보자. 
zip 함수의 용법에 대해서는 아래 참고자료를 보자.   
참고자료 : https://data-scientist-jeong.tistory.com/20

```python 
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
```
이 부분은 지금 모호하게 이해하고 있다. 
DataLoader 에서 collate_fn을 파라미터로 받는데,  
이때 DataLoader는 tensor를 배치 개수만큼 가져와서 차원이 늘어난 하나의 tensor로 출력한다.   
할인마트 1+1은 1개 같은 느낌이네.  
참고자료 : https://biomadscientist.tistory.com/170  

(2.21)일단 초기 데이터 로더까지 필요한건 여기까지다. main.py의 def train() 부분을 더 참고하자. 







