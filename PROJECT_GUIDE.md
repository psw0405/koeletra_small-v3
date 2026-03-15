# NER 프로젝트 구조 및 실행 가이드

이 문서는 현재 워크스페이스(`c:\Users\sungw\git_repo\NER`) 기준으로
디렉토리별 파일 설명과 실행 방법을 정리한 문서입니다.

## 1) 디렉토리별 파일 설명

### 루트 디렉토리

| 경로 | 설명 |
|---|---|
| `README.md` | 전체 파이프라인(다운로드, 추론, ONNX, C++) 실행 가이드 |
| `requirements.txt` | Python 의존성 목록 (`transformers`, `torch`, `optimum`, `onnxruntime` 등) |
| `train.jsonl` | 학습 데이터(JSONL). 각 줄은 `{text, entities}` 구조 |
| `valid.jsonl` | 검증 데이터(JSONL). 형식은 `train.jsonl`과 동일 |
| `.venv/` | 로컬 가상환경 |
| `scripts/` | Python 파이프라인 스크립트 |
| `models/` | 모델 가중치, 토크나이저, 라벨 파일 |
| `artifacts/` | ONNX 내보내기 결과 및 C++ 입력 텐서 |
| `cpp/` | C++ ONNX Runtime 추론 코드 및 실행 파일 |
| `third_party/` | ONNX Runtime SDK/라이브러리 |

### `scripts/`

| 경로 | 설명 |
|---|---|
| `scripts/download_koelectra.py` | 베이스 모델 다운로드 + token-classification head 초기화 + 라벨 파일 생성 |
| `scripts/train_ner.py` | JSONL 로드, 라벨 정규화, 토크나이즈/라벨 정렬, Trainer 학습/평가/저장 |
| `scripts/infer_python.py` | Python에서 토큰 단위 예측 + BIO 엔티티 복원 출력 |
| `scripts/export_onnx.py` | Hugging Face 모델을 ONNX로 export (`optimum`) |
| `scripts/prepare_onnx_inputs.py` | 텍스트를 ONNX 입력 텐서 txt로 변환 (`input_ids`, `attention_mask` 등) |
| `scripts/ner_labels.py` | 라벨 목록, BIO 라벨 생성, 데이터셋 라벨 alias 매핑 정의 |
| `scripts/__pycache__/` | Python 바이트코드 캐시 |

### `models/` 폴더별 역할

| 경로 | 설명 |
|---|---|
| `models/koelectra-small-initial/` | 초기 모델(헤드 랜덤 초기화 상태) |
| `models/koelectra-small-finetuned/` | 본 학습 결과 모델 |
| `models/koelectra-small-finetuned-smoke/` | 소규모 smoke 학습 결과 모델 |
| `models/*/checkpoint-*/` | 중간 체크포인트(재시작/분석용) |

### `models/*` 공통 주요 파일

| 파일명 | 설명 |
|---|---|
| `config.json` | 모델 구조 및 라벨 매핑 설정 |
| `model.safetensors` | 모델 가중치 |
| `tokenizer.json` | 토크나이저 전체 정의 |
| `tokenizer_config.json` | 토크나이저 설정 |
| `special_tokens_map.json` | 특수 토큰 매핑 |
| `vocab.txt` | 어휘 사전 |
| `base_labels.txt` | 엔티티 기본 라벨 목록 |
| `bio_labels.txt` | BIO 확장 라벨 목록 |
| `dataset_label_aliases.json` | 데이터셋 라벨명 보정(alias) |
| `eval_metrics.json` | 검증 지표(`eval_f1`, `eval_accuracy` 등) |
| `training_args.bin` | 학습 하이퍼파라미터 스냅샷 |
| `optimizer.pt` | 옵티마이저 상태(체크포인트) |
| `scheduler.pt` | LR 스케줄러 상태(체크포인트) |
| `rng_state.pth` | 난수 상태(재현성) |
| `trainer_state.json` | Trainer 진행 상태 |

### `artifacts/`

| 경로 | 설명 |
|---|---|
| `artifacts/onnx-initial/model.onnx` | C++ 추론용 ONNX 모델 |
| `artifacts/onnx-initial/config.json` | ONNX export 시 복사된 설정 |
| `artifacts/onnx-initial/tokenizer*.json`, `vocab.txt`, `special_tokens_map.json` | ONNX 추론 재현용 토크나이저 파일 |
| `artifacts/onnx-initial/base_labels.txt`, `bio_labels.txt` | 추론 라벨 파일 |
| `artifacts/input-initial/input_ids.txt` | 토큰 ID 입력 |
| `artifacts/input-initial/attention_mask.txt` | Attention Mask 입력 |
| `artifacts/input-initial/token_type_ids.txt` | Segment 입력 |
| `artifacts/input-initial/tokens.txt` | 토큰 문자열 |
| `artifacts/input-initial/text.txt` | 원문 텍스트 |

### `cpp/`

| 경로 | 설명 |
|---|---|
| `cpp/CMakeLists.txt` | C++ 빌드 설정 (`ONNXRUNTIME_DIR` 필요) |
| `cpp/ner_infer.cpp` | ONNX Runtime C++ 추론 프로그램 |
| `cpp/ner_infer.exe` | 현재 폴더에 있는 빌드 결과 실행 파일 |
| `cpp/onnxruntime.dll` | 실행 시 필요한 ONNX Runtime DLL |
| `cpp/onnxruntime_providers_shared.dll` | 실행 시 필요한 provider DLL |
| `cpp/ner_infer.obj` | 오브젝트 파일(빌드 중간 산출물) |

### `third_party/onnxruntime-win-x64-1.24.3/`

| 경로 | 설명 |
|---|---|
| `third_party/onnxruntime-win-x64-1.24.3/include/` | C/C++ 헤더 |
| `third_party/onnxruntime-win-x64-1.24.3/lib/onnxruntime.lib` | 링크용 import 라이브러리 |
| `third_party/onnxruntime-win-x64-1.24.3/lib/onnxruntime.dll` | 런타임 DLL |
| `third_party/onnxruntime-win-x64-1.24.3/lib/onnxruntime_providers_shared.*` | provider 관련 라이브러리/DLL |
| `third_party/onnxruntime-win-x64-1.24.3/README.md`, `LICENSE` 등 | 서드파티 문서/라이선스 |

## 2) 실행 방법 정리 (Windows PowerShell)

### 2-1) 환경 준비

```powershell
cd c:\Users\sungw\git_repo\NER
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2-2) 초기 모델 다운로드 (최초 1회)

```powershell
python scripts\download_koelectra.py `
  --base_model monologg/koelectra-small-v3-discriminator `
  --output_dir models\koelectra-small-initial
```

### 2-3) Python 추론 테스트

```powershell
python scripts\infer_python.py `
  --model_dir models\koelectra-small-initial `
  --text "이재명 대통령은 3월 2일 서울에서 회의를 열었다." `
  --max_length 64
```

### 2-4) 초기 모델 ONNX export + 입력 텐서 생성

```powershell
python scripts\export_onnx.py `
  --model_dir models\koelectra-small-initial `
  --output_dir artifacts\onnx-initial `
  --opset 17

python scripts\prepare_onnx_inputs.py `
  --model_dir models\koelectra-small-initial `
  --text "이재명 대통령은 3월 2일 서울에서 회의를 열었다." `
  --out_dir artifacts\input-initial `
  --max_length 64
```

### 2-5) C++ 빌드 (필요 시)

`ONNXRUNTIME_DIR`는 ONNX Runtime C++ 패키지 경로입니다.

```powershell
cd c:\Users\sungw\git_repo\NER\cpp
cmake -S . -B build -DONNXRUNTIME_DIR="C:/Users/sungw/git_repo/NER/third_party/onnxruntime-win-x64-1.24.3"
cmake --build build --config Release
```

### 2-6) C++ 추론 실행 (초기 모델)

```powershell
cd c:\Users\sungw\git_repo\NER\cpp

# A안: 현재 폴더의 실행 파일 사용
.\ner_infer.exe `
  ..\artifacts\onnx-initial\model.onnx `
  ..\artifacts\input-initial\input_ids.txt `
  ..\artifacts\input-initial\attention_mask.txt `
  ..\artifacts\input-initial\token_type_ids.txt `
  ..\artifacts\input-initial\tokens.txt `
  ..\artifacts\onnx-initial\bio_labels.txt

# B안: CMake 빌드 산출물 사용
.\build\Release\ner_onnx_infer.exe `
  ..\artifacts\onnx-initial\model.onnx `
  ..\artifacts\input-initial\input_ids.txt `
  ..\artifacts\input-initial\attention_mask.txt `
  ..\artifacts\input-initial\token_type_ids.txt `
  ..\artifacts\input-initial\tokens.txt `
  ..\artifacts\onnx-initial\bio_labels.txt
```

### 2-7) 파인튜닝

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

### 2-8) 파인튜닝 모델 ONNX + C++ 추론

```powershell
python scripts\export_onnx.py `
  --model_dir models\koelectra-small-finetuned `
  --output_dir artifacts\onnx-finetuned `
  --opset 17

python scripts\prepare_onnx_inputs.py `
  --model_dir models\koelectra-small-finetuned `
  --text "삼성전자는 5월 1일 서울 강남에서 신제품을 공개했다." `
  --out_dir artifacts\input-finetuned `
  --max_length 64

cd c:\Users\sungw\git_repo\NER\cpp
.\ner_infer.exe `
  ..\artifacts\onnx-finetuned\model.onnx `
  ..\artifacts\input-finetuned\input_ids.txt `
  ..\artifacts\input-finetuned\attention_mask.txt `
  ..\artifacts\input-finetuned\token_type_ids.txt `
  ..\artifacts\input-finetuned\tokens.txt `
  ..\artifacts\onnx-finetuned\bio_labels.txt
```

## 3) 참고 사항

- 초기 모델(`koelectra-small-initial`)은 token-classification head가 랜덤 초기화 상태이므로 정확도 자체는 의미가 작고, 파이프라인/런타임 검증 목적에 적합합니다.
- 라벨 alias는 학습 시 자동 적용됩니다.
  - `CulturalAsset -> CultureSite`
  - `DateDuration -> Duraion`
  - `QuantityTemperature -> QunatityTemperature`
  - `QuantityMoney -> QuantityPrice`
- 배포 환경 메모리 제약이 있는 경우 `max_length`를 48/64 수준으로 유지하고 batch=1 구성을 권장합니다.
