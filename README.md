# koELECTRA-small Korean NER Pipeline (Python + C++)

이 프로젝트는 아래 5단계를 실행할 수 있도록 구성되어 있습니다.

1. koELECTRA-small 다운로드
2. Python inference 테스트
3. C++ ONNX inference 테스트
4. `train.jsonl`, `valid.jsonl`로 파인튜닝
5. 파인튜닝 모델 C++ ONNX inference 테스트

## 0) 환경 준비

```powershell
cd c:\Users\sungw\git_repo\NER
pip install -r requirements.txt
```

## 1) koELECTRA-small 다운로드

```powershell
python scripts\download_koelectra.py `
  --base_model monologg/koelectra-small-v3-discriminator `
  --output_dir models\koelectra-small-initial
```

산출물:
- `models/koelectra-small-initial/`
- `models/koelectra-small-initial/base_labels.txt` (26개)
- `models/koelectra-small-initial/bio_labels.txt` (53개)

## 2) Python inference 테스트

```powershell
python scripts\infer_python.py `
  --model_dir models\koelectra-small-initial `
  --text "이재명 대통령은 3월 2일 서울에서 회의를 열었다." `
  --max_length 64
```

참고:
- 초기 모델은 token-classification head가 랜덤 초기화되어 있어 예측 정확도는 의미가 없습니다.
- 이 단계는 파이프라인/런타임 검증 목적입니다.

## 3) C++ ONNX inference 테스트 (초기 모델)

### 3-1) ONNX export

```powershell
python scripts\export_onnx.py `
  --model_dir models\koelectra-small-initial `
  --output_dir artifacts\onnx-initial `
  --opset 17
```

### 3-2) 입력 텐서 생성

```powershell
python scripts\prepare_onnx_inputs.py `
  --model_dir models\koelectra-small-initial `
  --text "이재명 대통령은 3월 2일 서울에서 회의를 열었다." `
  --out_dir artifacts\input-initial `
  --max_length 64
```

### 3-3) C++ 빌드

`ONNXRUNTIME_DIR`는 ONNX Runtime C++ 패키지 압축 해제 경로입니다.

```powershell
cd c:\Users\sungw\git_repo\NER\cpp
cmake -S . -B build -DONNXRUNTIME_DIR="C:/onnxruntime-win-x64-1.19.2"
cmake --build build --config Release
```

### 3-4) C++ 실행

```powershell
.\build\Release\ner_onnx_infer.exe `
  ..\artifacts\onnx-initial\model.onnx `
  ..\artifacts\input-initial\input_ids.txt `
  ..\artifacts\input-initial\attention_mask.txt `
  ..\artifacts\input-initial\token_type_ids.txt `
  ..\artifacts\input-initial\tokens.txt `
  ..\artifacts\onnx-initial\bio_labels.txt
```

## 4) 파인튜닝

```powershell
cd c:\Users\sungw\git_repo\NER
python scripts\train_ner.py `
  --model_dir models\koelectra-small-initial `
  --train_file train.jsonl `
  --valid_file valid.jsonl `
  --output_dir models\koelectra-small-finetuned `
  --max_length 64 `
  --batch_size 8 `
  --epochs 3 `
  --learning_rate 3e-5
```

라벨 alias 매핑(자동 적용):
- `CulturalAsset -> CultureSite`
- `DateDuration -> Duraion`
- `QuantityTemperature -> QunatityTemperature`
- `QuantityMoney -> QuantityPrice`

## 5) 파인튜닝 모델 C++ inference 테스트

### 5-1) ONNX export

```powershell
python scripts\export_onnx.py `
  --model_dir models\koelectra-small-finetuned `
  --output_dir artifacts\onnx-finetuned `
  --opset 17
```

### 5-2) 입력 텐서 생성

```powershell
python scripts\prepare_onnx_inputs.py `
  --model_dir models\koelectra-small-finetuned `
  --text "삼성전자는 5월 1일 서울 강남에서 신제품을 공개했다." `
  --out_dir artifacts\input-finetuned `
  --max_length 64
```

### 5-3) C++ 실행

```powershell
cd c:\Users\sungw\git_repo\NER\cpp
.\build\Release\ner_onnx_infer.exe `
  ..\artifacts\onnx-finetuned\model.onnx `
  ..\artifacts\input-finetuned\input_ids.txt `
  ..\artifacts\input-finetuned\attention_mask.txt `
  ..\artifacts\input-finetuned\token_type_ids.txt `
  ..\artifacts\input-finetuned\tokens.txt `
  ..\artifacts\onnx-finetuned\bio_labels.txt
```

## 참고

- 사용자 요청 라벨 목록의 철자(`Duraion`, `QunatityTemperature`)는 그대로 사용했습니다.
- 메모리 제약 디바이스용 배포 시 `max_length` 축소(예: 48/64), batch=1 고정을 권장합니다.
