# Intro  

이 글은 NetVLAD의 논문과, 해당 논문을 구현한 코드를 분석하기 위해 작성한 문서입니다.   
논문 링크 : https://arxiv.org/pdf/1511.07247  
논문의 내용과 코드 실행 순서에 따라 주요 구성 및 구현 방법을 리뷰하고자 합니다.   

내용은 다음과 같이 정리합니다. 

1. 전체 Framework
2. 각 Framework의 단계별 내용과 의미와 각 언어별 구현 형태 비교

이미 pytorch에 대한 예제가 있으나, 현 시점에 맞춰 코드를 업데이트하는데 활용하고자 합니다. 


# 전체 Framework

## 1. Test  
논문에서는 VPR문제를 이미지 찾기 문제로 다시 정의하고 있습니다. 어떻게 이미지를 찾아내는지의 과정을 설명합니다. Network의 형태를 정의하고, 이미지를 입력했을때, Network의 출력값이 어떻게 되는지에 대한 설명입니다.  
이후 코드를 통해 해당 부분이 어떻게 구현되었는지 확인해봅니다.   
Test는 아래와 같이 구성되어있습니다. 

    1.1. 이미지의 입력  
    1.2. CNN Network : VGG16의 conv5 사용        
    1.3. VLAD Layer
        1.3.1. VLAD Descriptor
        1.3.2. Cluster
        1.3.2. Layer Add
    1.4. 출력 Vector를 이용한 최근접 data 탐색

## 2. Training  
이미지를 찾아내는 Network를 어떻게 학습시키는지에 대한 부분을 설명합니다.  
데이터를 불러와 Dataset을 구성하는 방법에서부터, triplet의 구성과 Network의 weight를 업데이트하는 방법에 대해 설명합니다.   
이후 코드와 함께 각 단계가 어떻게 진행되는지 확인합니다.    
    
    2.1. Dataset 구성  
    2.2. Feature의 추출  
    2.3. Triplet Dataset의 구성  
    2.4. Training 시작    
        2.4.1. Subset 구성
        2.4.2. DataLoad
        2.4.3. Training
        2.4.4. Loss의 계산
        2.4.5. Backpropergation

     

