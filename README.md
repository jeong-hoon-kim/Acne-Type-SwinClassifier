# Swin Transformer를 이용한 여드름 이미지 분류 프로젝트

![Swin Transformer Logo](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/swin_transformer.jpg)

## 프로젝트 개요

[cite_start]본 프로젝트는 `Swin Transformer` 모델을 활용하여 여드름 이미지 데이터를 분석하고 종류별로 여드름을 분류하는 딥러닝 모델을 개발합니다. [cite_start]여드름은 흔한 피부 질환으로, 잘못된 치료는 흉터를 남기거나 피부 문제를 악화시킬 수 있어 증상별 단계별 치료가 중요합니다. [cite_start]특히 남성의 경우 여드름 관리에 대한 인식이 저조하여  [cite_start]적절한 관리를 위한 분류 기술의 필요성이 대두되었습니다.

## 프로젝트 목표

* 여드름 이미지 데이터셋 구축 및 분석
* Swin Transformer 모델을 활용한 여드름 종류별 분류 모델 개발
* 높은 정확도와 F1 Score를 달성하여 실제 활용 가능성 검증
* 개발된 모델을 활용한 웹 서비스 구현

## 📊 데이터셋

[cite_start]본 프로젝트는 `Roboflow`를 통해 수집된 여드름 이미지 데이터셋을 활용합니다.

* [cite_start]**해상도**: 640x640 해상도의 JPG 이미지 
* [cite_start]**총 이미지 파일 수**: 8,678개 
    * [cite_start]Train 이미지: 7,830개 
    * [cite_start]Valid 이미지: 632개 
    * [cite_start]Test 이미지: 216개 
* [cite_start]**분류 클래스**: 10개의 Multi-label 데이터로 구성 
    * [cite_start]Pimples (면포성 여드름), Papular (구진성 여드름), Purulent (화농성 여드름), Blackhead (블랙헤드), Cystic (결정성 여드름), Milium (비립종), Keloid (켈로이드성 여드름), Folliculitis (모낭염), Crysanlline (낭종성 여드름), Conglobata (응괴성 여드름) 
* [cite_start]**특징**: 클래스 불균형이 존재합니다.

![레이블별 이미지 수 분포](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/label_distribution.png?raw=true)

### 여드름 유형별 이미지 및 설명

| 유형 (영어)    | 유형 (한글)      | 설명                                                                                                                                                                                                                               | 예시 이미지                                                                                                                                                                 |
| :------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Pimples`      | 면포성 여드름    | [cite_start]여드름의 전형적인 형태로, 색에 따라 화이트헤드와 블랙헤드로 불립니다[cite: 23].                                                                                                                                                     | ![Pimples Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_pimples.png?raw=true) *(실제 이미지로 변경)* |
| `Papular`      | 구진성 여드름    | [cite_start]면포, 붉은 구진, 고름이 든 농포가 많이 형성되어 있습니다[cite: 23].                                                                                                                                                                | ![Papular Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_papular.png?raw=true) *(실제 이미지로 변경)* |
| `Purulent`     | 화농성 여드름    | [cite_start]여드름이 화농하여 피부 깊숙이 농포를 만들고 있는 단계입니다[cite: 25].                                                                                                                                                            | ![Purulent Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_purulent.png?raw=true) *(실제 이미지로 변경)* |
| `Blackhead`    | 블랙헤드         | [cite_start]모공이 열려 피지와 노폐물이 산화되어 검게 보이는 여드름입니다[cite: 26].                                                                                                                                                            | ![Blackhead Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_blackhead.png?raw=true) *(실제 이미지로 변경)* |
| `Cystic`       | 결정성 여드름    | [cite_start]고름이 찬 것처럼 느껴지고 여러 개의 낭종들이 뭉쳐 크기가 더 커지기도 합니다[cite: 25].                                                                                                                                             | ![Cystic Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_cystic.png?raw=true) *(실제 이미지로 변경)* |
| `Milium`       | 비립종           | [cite_start]피부 표면에 작고 하얀 알갱이 형태로 나타나며 작은 각질 주머니가 있습니다[cite: 26].                                                                                                                                               | ![Milium Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_milium.png?raw=true) *(실제 이미지로 변경)* |
| `Keloid`       | 켈로이드성 여드름 | [cite_start]화농되는 모낭 염증과 모낭 주위염은 치유 후에도 켈로이드를 형성합니다[cite: 23].                                                                                                                                                     | ![Keloid Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_keloid.png?raw=true) *(실제 이미지로 변경)* |
| `Folliculitis` | 모낭염           | [cite_start]화장품, 피부 마찰 등 외적 요인으로 손상을 받거나, 피부 면역이 약해져 생깁니다[cite: 24].                                                                                                                                             | ![Folliculitis Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_folliculitis.png?raw=true) *(실제 이미지로 변경)* |
| `Crystanlline` | 낭종성 여드름    | [cite_start]딱딱하게 만져지는 형태로 초기 단계에 치료하지 않거나 잘못 치료하여 발생합니다[cite: 25].                                                                                                                                           | ![Crystanlline Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_crystanlline.png?raw=true) *(실제 이미지로 변경)* |
| `Conglobata`   | 응괴성 여드름    | [cite_start]여드름의 가장 심한 형태로 전문가의 상담과 장기적인 치료가 필요합니다[cite: 23].                                                                                                                                                     | ![Conglobata Example](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/example_conglobata.png?raw=true) *(실제 이미지로 변경)* |

## 개발 흐름도

![개발 흐름도](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/flowchart.png?raw=true)

개발 과정은 다음과 같습니다:
1.  [cite_start]**Acne Image Dataset**: 8678개의 이미지 데이터셋을 사용합니다.
2.  [cite_start]**Data Analysis**: 데이터 분석을 수행합니다.
3.  **Data Processing**:
    * [cite_start]`Resizing`: 224x224로 이미지 크기를 조정합니다.
    * [cite_start]`Normalization`: 데이터 정규화를 수행합니다.
    * [cite_start]`Augmentation`: 데이터 증강을 적용하여 과적합 방지 및 다중 레이블 학습 효과를 얻습니다.
4.  [cite_start]**Swin Transformer (Tiny)**: 사전 학습된 Swin Transformer (Tiny) 모델을 불러와 classification layer만 교체하여 학습합니다.
5.  [cite_start]**Output**: 분류 결과를 출력합니다.

## ⚙실험 방법

### Swin Transformer를 사용한 이유
[cite_start]Swin Transformer는 CNN과 ViT (Vision Transformer)의 장점을 조합하여 정밀한 로컬 특징 추출과 전역적인 문맥 이해를 동시에 수행할 수 있는 구조입니다. [cite_start]크기, 위치, 형태가 다양하고 복합적인 시각적 패턴을 가진 여드름 분류 프로젝트에서 기존 CNN이나 ViT보다 더 유리한 성능을 발휘할 수 있습니다.

### 기본 설정
* [cite_start]**Optimizer**: AdamW 
* [cite_start]**Learning Rate (lr)**: 3e-4 (초기), 2e-4 (최종 튜닝) 
* [cite_start]**Weight Decay**: 1e-4 
* [cite_start]**Scheduler**: CosineAnnealingLR (T_max=EPOCHS, eta_min=1e-6 -> 1e-5로 조정) 
* [cite_start]**Batch Size**: 32 
* [cite_start]**Data Augmentation**: Flip, RandomBrightnessContrast 적용 

### 성능 평가
[cite_start]각 클래스의 Train/Valid Loss를 산출하고 정확도(Accuracy)와 F1 Score를 통해 성능을 평가했습니다.

## 실험 결과

### 초기 실험 결과

| Model        | Train Acc | Val Acc | Test Acc | F1-score | 비고                                   |
| :----------- | :-------- | :------ | :------- | :------- | :------------------------------------- |
| Swin-Tiny    | 0.9991    | 0.9758  | 0.9824   | [cite_start]0.9720   | lr = 3e-4                    |
| Swin-Tiny    | 0.9977    | 0.9755  | 0.9810   | [cite_start]0.9666   | lr = 3e-4, Flip/RandomBrightnessContrast 적용  |
| Swin-Tiny    | 0.9991    | 0.9820  | 0.9884   | [cite_start]0.9813   | lr = 2e-4, Flip 적용                    |
| Swin-Tiny    | 0.9951    | 0.9672  | 0.9636   | [cite_start]0.9635   | lr = 3e-5                               |
| EfficientNetV2 | 0.9991    | 0.9847  | 0.9884   | [cite_start]0.9791   | lr = 3e-4                               |

[cite_start]*초기에는 EfficientNetV2에 비해 비슷하거나 약간 낮은 성능을 보였습니다.*

[cite_start]Transformer 계열 모델은 CNN보다 더 많은 학습 에폭이 필요할 수 있다는 분석을 통해, Early stopping 기준을 `val_loss`에서 `val_acc`로 변경하여 더 많은 에폭 학습을 진행했습니다.

### 미세 조정 후 최종 결과

* [cite_start]**Test Accuracy**: 0.9912 
* [cite_start]**F1 Score**: 0.9868 
* [cite_start]**적용된 설정**: lr = 2e-4, Flip 적용, CosineAnnealingLR scheduler 최소값을 1e-6에서 1e-5로 조정 

[cite_start]많은 에폭 학습을 통해 성공적인 성능 향상이 있었습니다.

### 학습 곡선 (미세 조정 후)
![미세 조정 후 Train/Valid Loss](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/tuned_loss.png?raw=true)
![미세 조정 후 Train/Valid Accuracy](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/tuned_accuracy.png?raw=true)

### 모델별 Test Accuracy 및 F1-score 결과
![모델별 성능 비교](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/model_comparison.png?raw=true)

### Swin-Tiny (튜닝) 레이블별 정확도
![레이블별 정확도](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/label_accuracy.png?raw=true)

## 활용 방안 구현 (웹 서비스)

[cite_start]개발된 모델을 활용하여 여드름 이미지를 업로드하고 분류 결과를 확인할 수 있는 웹 서비스를 구현했습니다.
[cite_start]사용자는 이미지를 업로드하여 여드름 유형을 분류하고, 각 유형에 대한 상세 설명을 얻을 수 있습니다.

![웹 서비스 초기 화면](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/web_service_initial.png?raw=true)
![웹 서비스 분류 예시](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/web_service_classification.png?raw=true)
![웹 서비스 다중 레이블 분류 예시 및 설명](https://github.com/jeong-hoon-kim/Acne-Type-SwinClassifier/blob/main/images/web_service_multi_label.png?raw=true)
