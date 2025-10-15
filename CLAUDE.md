# CLAUDE.md

**⚠️ 중요 사항:**
1. **이 파일의 목적**: 릴리즈 직전 코드 정리 및 문서화를 위한 개발자 참고 자료
2. **Git 커밋 시 주의**: 이 파일을 git push 할 때는 "Claude" 관련 내용을 제거하고 일반 개발 문서로 변환할 것
   - 파일명을 `DEVELOPMENT.md` 또는 `ARCHITECTURE.md`로 변경 권장
   - Claude Code 관련 언급은 모두 삭제

---

## 프로젝트 개요

LDM-EKI는 대기 확산 모델링과 앙상블 칼만 역산을 결합한 방사능 오염원 추정 시스템입니다:
- **CUDA/C++ 순방향 모델**: GPU 가속 입자 확산 시뮬레이션
- **Python 역산 모델**: 앙상블 칼만 기법을 이용한 오염원 최적화
- **IPC 통신**: POSIX 공유 메모리를 통한 고성능 프로세스 간 데이터 교환

## 빌드 및 실행

### 프로젝트 빌드

```bash
# 메인 EKI 실행파일 빌드 (기본 타겟)
make

# 모든 타겟 빌드 (ldm, ldm-eki, ldm-receptor-debug)
make all-targets

# 빌드 산출물 정리
make clean
```

**필수 요구사항:**
- CUDA 툴킷 (nvcc) 경로: `/usr/local/cuda/bin/nvcc`
- GPU 컴퓨팅 성능: SM 6.1 이상
- OpenMP 지원 (병렬 빌드용)

**빌드 최적화:**
- 자동 병렬 빌드 활성화 (CPU 코어 수만큼)
- 최적화 레벨: `-O2` (빠른 컴파일, 충분한 성능)
- 빌드 시간: ~30초-1분 (시스템에 따라 다름)

### 데이터 정리

```bash
# 이전 실행 데이터 정리 (수동 실행 시)
python3 util/cleanup.py

# 옵션:
# --dry-run         실제 삭제 없이 미리보기
# --no-confirm      확인 없이 즉시 삭제
# --logs-only       logs만 정리
# --output-only     output만 정리
# --shm-only        공유 메모리만 정리
```

**참고:** `./ldm-eki` 실행 시 자동으로 cleanup.py가 호출되어 이전 데이터를 정리합니다.

### 시뮬레이션 실행

```bash
# EKI 최적화 실행
./ldm-eki

# 실행 시 자동으로:
# 1. 이전 데이터 정리 (확인 프롬프트 표시)
# 2. 시뮬레이션 수행
# 3. 결과 시각화 자동 생성

# 출력 파일 위치:
# - logs/ldm_eki_simulation.log - 메인 시뮬레이션 로그
# - logs/python_eki_output.log - Python EKI 프로세스 로그
# - output/plot_vtk_prior/ - 사전 시뮬레이션 VTK 파일
# - output/plot_vtk_ens/ - 앙상블 시뮬레이션 VTK 파일
# - output/results/all_receptors_comparison.png - 시각화 결과 (자동 생성)
```

## 시스템 아키텍처

### 2-프로세스 설계

시스템은 **두 개의 협력 프로세스**로 동작합니다:

1. **LDM (C++/CUDA)**: `ldm-eki` 실행파일
   - 순방향 입자 확산 시뮬레이션 수행
   - GPU에서 CUDA 커널 실행
   - 관측값을 공유 메모리에 기록 (`/dev/shm/ldm_eki_*`)
   - 공유 메모리에서 앙상블 상태 읽기
   - 빠른 반복을 위한 기상 데이터 사전 로딩

2. **EKI (Python)**: `src/eki/RunEstimator.py`
   - LDM에 의해 백그라운드에서 자동 실행
   - 공유 메모리에서 관측값 읽기
   - 앙상블 칼만 역산 수행
   - 업데이트된 앙상블 상태를 공유 메모리에 기록
   - 수렴 또는 최대 반복 횟수까지 반복

### IPC 통신 흐름

```
[초기 실행]
LDM (C++) → 관측값 → 공유 메모리 (/dev/shm/ldm_eki_data)
                         ↓
Python EKI ← 관측값 ← 공유 메모리
Python EKI가 사전 앙상블 생성

[반복 루프 - N회 반복]
Python EKI → 앙상블 상태 → 공유 메모리 (/dev/shm/ldm_eki_ensemble_*)
                              ↓
LDM (C++) ← 앙상블 상태 ← 공유 메모리
LDM이 N개의 앙상블 시뮬레이션 실행 (각 멤버당 1개)
LDM (C++) → 앙상블 관측값 → 공유 메모리
                              ↓
Python EKI ← 앙상블 관측값 ← 공유 메모리
Python EKI가 칼만 이득을 사용해 앙상블 업데이트
[루프 계속...]
```

### 주요 IPC 모듈

**C++ 측:**
- `src/include/ldm_eki_ipc.cuh`: IPC writer/reader 클래스
  - `EKIWriter::writeObservations()` - 초기 관측값 기록
  - `EKIWriter::writeEnsembleObservations()` - 앙상블 관측값 기록
  - `EKIReader::waitForEnsembleData()` - Python 상태 대기
  - `EKIReader::readEnsembleStates()` - 앙상블 상태 읽기

**Python 측:**
- `src/eki/eki_ipc_reader.py`: C++로부터 관측값 읽기
  - `receive_gamma_dose_matrix_shm()` - 초기 관측값
  - `receive_ensemble_observations_shm()` - 앙상블 관측값
- `src/eki/eki_ipc_writer.py`: C++로 앙상블 상태 쓰기
  - `write_ensemble_to_shm()` - 앙상블 상태 전송

### 설정 시스템

**LDM 설정** (`data/input/setting.txt`):
- 시뮬레이션 파라미터: time_end, dt, 입자 수
- 물리 모델: 난류, 침적, 붕괴
- 파일 경로 및 그리드 차원

**EKI 설정** (`data/eki_settings.txt`):
- 수용체 위치 및 포착 반경
- 참값/사전 방출량 시계열
- EKI 알고리즘 파라미터 (앙상블 크기, 반복 횟수, adaptive/localized 옵션)
- GPU 설정

**공유 메모리 설정:**
- 두 프로세스 모두 `data/eki_settings.txt` 읽기
- C++가 전체 설정을 `/dev/shm/ldm_eki_full_config`에 기록 (128 바이트)
- Python이 `Model_Connection_np_Ensemble.py::load_config_from_shared_memory()`를 통해 설정 읽기

### 핵종 시스템

모델은 CRAM (Chebyshev Rational Approximation Method)을 사용한 방사성 붕괴 체인을 지원합니다:

- 핵종 정의: `data/input/nuclides_config_1.txt` (또는 60-핵종 체인용 `nuclides_config_60.txt`)
- CRAM 행렬: `cram/A60.csv`
- 붕괴 체인 처리: `src/include/ldm_nuclides.cuh`

## EKI 최적화 알고리즘

Python EKI 구현 (`src/eki/Optimizer_EKI_np.py`)은 다음 알고리즘들을 지원합니다:

- **EnKF**: 표준 앙상블 칼만 필터
- **Adaptive_EnKF**: 적응형 스텝 크기 조절
- **EnKF_with_Localizer**: 공분산 국소화 (거짓 상관관계 제거)
- **EnRML**: 앙상블 랜덤화 최대우도법
- **EnKF_MDA**: 다중 데이터 동화
- **REnKF**: 제약조건을 가진 정규화 EnKF

`data/eki_settings.txt`에서 제어:
```
EKI_ADAPTIVE=On/Off
EKI_LOCALIZED=On/Off
EKI_REGULARIZATION=On/Off
```

## 중요 구현 세부사항

### 기상 데이터 사전 로딩

앙상블 모드에서는 반복 전에 **모든 기상 데이터를 사전 로딩**합니다:
- 함수: `LDM::preloadAllEKIMeteorologicalData()` (`ldm.cuh`)
- 반복 중 파일 I/O를 피하기 위해 모든 타임스텝을 병렬로 로드
- `eki_meteo_cache` 멤버 변수에 저장
- 성능에 필수적: 빠른 앙상블 반복 가능

### 입자 초기화

**단일 모드** (초기 참값 시뮬레이션):
- `LDM::initializeParticlesEKI()`: 설정 파일의 `true_emissions` 사용

**앙상블 모드** (각 반복마다):
- `LDM::initializeParticlesEKI_AllEnsembles()`: 모든 앙상블 멤버용 입자 생성
- 각 앙상블은 고유한 `ensemble_id`를 가진 입자 세트 보유
- 입자 데이터 구조에 `ensemble_id`와 `timeidx` 필드 포함

### VTK 출력 제어

VTK 출력은 비용이 크므로 `ldm.enable_vtk_output`로 제어:
- 초기 참값 시뮬레이션(단일 모드)에서 활성화
- 중간 반복 중에는 **비활성화** (성능 최적화)
- 최종 반복에서만 **활성화**
- 선택된 앙상블 멤버(예: 앙상블 7)가 `output/plot_vtk_ens/`에 출력

### 관측 시스템

관측값은 **수용체 위치**에서 수집됩니다 (그리드 아님):
- 수용체는 `eki_settings.txt`에서 위도/경도로 정의
- 포착 반경: `RECEPTOR_CAPTURE_RADIUS` (도 단위)
- GPU 배열: `d_eki_receptor_observations`가 각 수용체의 타임스텝별 선량 저장
- 형태: `[num_ensemble][num_timesteps][num_receptors]`

### 데이터 재배열 규칙

**중요**: Python과 C++는 서로 다른 배열 레이아웃 사용:

**Python (NumPy)**: 상태에 대해 열-우선(Column-major)
- 앙상블 상태: `(num_states, num_ensemble)`
- 관측값: `(num_receptors, num_timesteps)`

**C++**: 행-우선(Row-major)
- 앙상블 상태: `[ensemble][state]`
- 관측값: `[ensemble][timestep][receptor]`

공유 메모리에 쓸 때는 **항상 행-우선 순서로 평탄화**해야 C++에서 올바르게 읽을 수 있습니다.

### Memory Doctor 모드

IPC 통신 문제 디버깅용:
```
MEMORY_DOCTOR_MODE=On
```
C++와 Python 간 모든 데이터 전송을 `/tmp/eki_debug/`에 로깅하여 비교 가능.

## 일반적인 개발 패턴

### 새로운 EKI 알고리즘 추가

1. `src/eki/Optimizer_EKI_np.py`의 `Inverse` 클래스에 메서드 추가
2. `data/eki_settings.txt`에 설정 옵션 추가
3. `Model_Connection_np_Ensemble.py`의 `load_config_from_shared_memory()` 업데이트
4. `Optimizer_EKI_np.py::Run()`에 새 메서드 호출 케이스 추가

### 관측 수집 방식 수정

1. C++ 측: `src/include/ldm_eki_ipc.cuh` 수정
2. Python 측: `src/eki/eki_ipc_reader.py` 수정
3. writer/reader 간 데이터 형식 일치 확인
4. 필요시 공유 메모리 버퍼 크기 업데이트

### 물리 모델 추가

1. `src/kernels/ldm_kernels.cuh`에 CUDA 커널 추가
2. `src/include/ldm_func.cuh`에서 커널 호출하도록 업데이트
3. `data/input/setting.txt`에 설정 추가
4. `src/include/ldm_init.cuh`에서 설정 파싱

## 파일 구조

```
src/
├── main_eki.cu              - EKI 실행파일 진입점
├── include/                 - CUDA 헤더 파일
│   ├── ldm.cuh             - 메인 LDM 클래스
│   ├── ldm_eki_ipc.cuh     - IPC 통신
│   ├── ldm_func.cuh        - 핵심 시뮬레이션 함수
│   ├── ldm_init.cuh        - 초기화
│   └── ldm_nuclides.cuh    - 붕괴 체인 처리
├── kernels/                 - CUDA 커널
│   └── ldm_kernels.cuh     - GPU 커널 (이류, 확산 등)
└── eki/                     - Python EKI 프레임워크
    ├── RunEstimator.py      - 메인 EKI 실행기
    ├── Optimizer_EKI_np.py  - 칼만 역산 알고리즘
    ├── Model_Connection_np_Ensemble.py - 순방향 모델 인터페이스
    ├── eki_ipc_reader.py    - C++로부터 읽기
    └── eki_ipc_writer.py    - C++로 쓰기

util/                        - 유틸리티 스크립트
├── cleanup.py               - 데이터 정리 스크립트
├── compare_all_receptors.py - 결과 시각화 (자동 실행)
├── compare_logs.py          - 로그 비교 도구
└── diagnose_convergence_issue.py - 수렴 진단 도구

data/
├── input/
│   ├── setting.txt          - LDM 시뮬레이션 설정
│   ├── nuclides_config_*.txt - 핵종 정의
│   └── gfsdata/             - 기상 데이터 (GFS 형식)
└── eki_settings.txt         - EKI 알고리즘 설정

output/
├── plot_vtk_prior/          - 참값 시뮬레이션 VTK 파일
├── plot_vtk_ens/            - 앙상블 실행 VTK 파일
└── results/                 - 그래프 및 분석 출력
```

## 디버깅 팁

**공유 메모리 문제:**
```bash
# 공유 메모리 파일 목록
ls -lh /dev/shm/ldm_eki*

# 필요시 수동 정리
rm -f /dev/shm/ldm_eki_*
```

**프로세스 통신 확인:**
```bash
# Python 프로세스 모니터링
ps aux | grep RunEstimator

# 로그 확인
tail -f logs/ldm_eki_simulation.log
tail -f logs/python_eki_output.log
```

**GPU 사용 확인:**
```bash
nvidia-smi
```

**Memory Doctor 진단:**
`MEMORY_DOCTOR_MODE=On` 활성화 후 `/tmp/eki_debug/`에서 상세한 데이터 전송 로그 확인.

## 최근 변경사항 (2025)

### 코드 정리 및 최적화
- **MPI 제거**: 단일 프로세스 모드로 단순화
  - `mpiRank`, `mpiSize` 변수 제거
  - `PROCESS_INDEX` 상수로 대체 (값: 0)
  - MPI 헤더 및 라이브러리 의존성 제거

- **빌드 최적화**:
  - `-O3` → `-O2`: 빠른 컴파일, 충분한 성능
  - 불필요한 플래그 제거: `-DCRAM_DEBUG`, `-fPIC`, `-lcublas`, `-lmpi`
  - 자동 병렬 빌드: `make -j$(nproc)` 자동 적용
  - 빌드 시간: ~2-3분 → ~30초-1분

- **유틸리티 스크립트 정리**:
  - 최상위 `.py` 파일들을 `util/` 폴더로 이동
  - 상대 경로 사용으로 변경
  - `cleanup.py`: 통합 데이터 정리 스크립트 추가

- **자동 정리 기능**:
  - `./ldm-eki` 실행 시 자동으로 `cleanup.py` 호출
  - 확인 프롬프트로 안전한 정리
  - `logs/`, `output/`, `/dev/shm/ldm_eki_*` 일괄 정리

### 시각화 개선
- `compare_all_receptors.py`: 출력 디렉토리 자동 생성
- 시뮬레이션 종료 후 자동 시각화 생성
- 다중 수용체 지원 (페이지네이션)

### 출력 스타일 가이드 작성 (2025-01-15)
- **OUTPUT_STYLE_GUIDE.md 생성**: 전문적이고 일관된 콘솔 출력을 위한 종합 가이드
  - 색상 시스템 정의 (태그, 강조, 구분선)
  - 태그 시스템 정리: 유지/삭제/통합 규칙
  - 출력 형식 표준화: 섹션 헤더, 진행률, 데이터 요약
  - Before/After 예제로 실용적 개선 방안 제시
  - 함수 문서화 규칙: @output 주석으로 터미널 출력 설명
  - 공백 사용 규칙: 논리적 그룹화로 가독성 향상
  - Quick Reference Card: 자주 쓰는 패턴 정리

### 코드 국제화 (2025-01-15)
- **전체 출력 메시지 영어 변환**: 릴리즈 준비를 위한 국제화 작업
  - 모든 한국어 출력 메시지를 영어로 번역
  - 대상 파일:
    - `src/main_eki.cu` - 메인 진입점
    - `src/include/ldm_eki_ipc.cuh` - IPC 통신
    - `src/include/ldm_func.cuh` - 시뮬레이션 함수
    - `src/include/ldm_init.cuh` - 입자 초기화
    - `src/include/ldm_mdata.cuh` - 기상자료 로딩
  - 로그 태그 정리: 불필요한 태그 제거 ([INIT], [EKI], [LOG] 등)
  - 필수 태그만 유지: [ERROR], [ENSEMBLE], [VTK], [VISUALIZATION], [DEBUG*]

- **색상 코딩 시스템 도입**:
  - `src/include/colors.h` 추가: ANSI 색상 코드 정의
  - 에러 메시지: 빨간색
  - 성공 메시지: 녹색
  - 경고 메시지: 노란색
  - 헤더: 청록색 + 굵게
  - 안전한 구현: ANSI 미지원 터미널에서도 정상 작동
