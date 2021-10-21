# Transformer Style Transfer
 
- 트랜스포머(Transformer)를 사용한 스타일 트랜스퍼(Style Transfer)구현.
- GAN 구조 사용
- Generator를 트랜스포머로 대체, Transformer Encoder와 Decoder 모두 사용.

### Train

    train.py


#### DataSets

    /dataset/data/trainA
    /dataset/data/trainB

콘텐츠 이미지와 스타일 이미지를 각 trainA, trainB에 넣어줌

- Optimizer Adam 사용 (beta1= 0.99) <- GAN에서는 beta1을 0.5를 권장하지만 TransGAN에서는 0.0을 권장해서 가장 디폴트 값인 0.99를 사용
