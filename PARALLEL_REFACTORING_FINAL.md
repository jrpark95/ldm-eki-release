# 🚀 LDM-EKI CUDA 프로젝트 병렬 리팩토링 최종 가이드

## 📋 프로젝트 개요
**목표**:
1. 모든 `.cuh` 파일을 헤더와 `.cu` 구현부로 분리
2. 대용량 파일을 논리적 단위로 분할 (한 파일당 최대 500-800줄)
3. MPI 코드는 이미 제거됨 (추가 제거 불필요)

**전략**: 5명이 균등하게 작업 분담 → 완전 병렬 처리 → Agent 6이 최종 통합

---

## 📊 전체 파일 현황 및 분담

**총 11,525줄을 5명이 분담** (각자 약 2,300줄)

| 파일명 | 라인 수 | 분할 계획 | 담당 |
|--------|---------|-----------|------|
| ldm_kernels.cuh | 3864 | 6개 파일로 분할 | Agent 1 |
| ldm_mdata.cuh | 1978 | 3개 파일로 분할 | Agent 2 |
| ldm_func.cuh | 1548 | 3개 파일로 분할 | Agent 2 |
| ldm_plot.cuh | 824 | 2개 파일로 분할 | Agent 3 |
| ldm_init.cuh | 805 | 2개 파일로 분할 | Agent 3 |
| ldm.cuh | 764 | 그대로 유지 | Agent 3 |
| ldm_eki_ipc.cuh | 690 | 2개 파일로 분할 | Agent 4 |
| memory_doctor.cuh | 258 | 그대로 유지 | Agent 4 |
| ldm_cram2.cuh | 240 | 그대로 유지 | Agent 4 |
| ldm_nuclides.cuh | 215 | 그대로 유지 | Agent 4 |
| ldm_struct.cuh | 179 | 헤더 전용 | Agent 5 |
| ldm_config.cuh | 128 | 헤더 전용 | Agent 5 |
| ldm_kernels_cram.cuh | 32 | 그대로 유지 | Agent 5 |
| **Python/Makefile** | - | 정리 및 업데이트 | Agent 5 |

---

## 👥 Agent별 상세 작업 분담

### 🔴 Agent 1: 대용량 커널 파일 전문가 (3,864줄)
**담당**: src/kernels/ldm_kernels.cuh 완전 처리

**분할 및 리팩토링 계획**:
```bash
# 6개 파일로 논리적 분할
1. ldm_kernels_device.cuh/cu (150줄)
   - __device__ 유틸리티 함수들
   - nuclear_decay, k_f_esi, k_f_esl, getRand 등

2. ldm_kernels_particle.cuh/cu (800줄)
   - move_part_by_wind 커널
   - 입자 이동 관련 핵심 로직

3. ldm_kernels_particle_ens.cuh/cu (900줄)
   - move_part_by_wind_ens 커널
   - 앙상블 버전 입자 이동

4. ldm_kernels_eki.cuh/cu (300줄)
   - compute_eki_receptor_dose 커널
   - compute_eki_receptor_dose_ensemble 커널

5. ldm_kernels_dump.cuh/cu (800줄)
   - move_part_by_wind_dump 커널
   - 덤프 관련 로직

6. ldm_kernels_dump_ens.cuh/cu (900줄)
   - move_part_by_wind_ens_dump 커널
   - 앙상블 덤프 버전
```

**작업 방법**:
```python
# Grep으로 함수 경계 파악
grep -n "^__global__\|^__device__\|^}" src/kernels/ldm_kernels.cuh

# 청크 단위 읽기 (500줄씩)
for offset in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]:
    Read(file, offset=offset, limit=500)
    # 논리적 단위로 분할
```

**최종 산출물**: 12개 파일 (6개 .cuh + 6개 .cu)

---

### 🔵 Agent 2: 데이터 및 시뮬레이션 모듈 (3,526줄)
**담당**: ldm_mdata.cuh (1978줄) + ldm_func.cuh (1548줄)

**분할 및 리팩토링 계획**:

```bash
# ldm_mdata.cuh → 3개 파일로 분할
1. ldm_mdata_loading.cuh/cu (700줄)
   - 기상 데이터 파일 로딩
   - GFS 데이터 읽기 함수들

2. ldm_mdata_processing.cuh/cu (700줄)
   - 데이터 전처리 및 변환
   - 보간 함수들

3. ldm_mdata_cache.cuh/cu (578줄)
   - EKI 기상 데이터 캐싱
   - 메모리 관리

# ldm_func.cuh → 3개 파일로 분할
1. ldm_func_simulation.cuh/cu (600줄)
   - 핵심 시뮬레이션 루프
   - run_simulation 관련

2. ldm_func_particle.cuh/cu (500줄)
   - 입자 초기화/업데이트
   - 입자 관리 함수들

3. ldm_func_output.cuh/cu (448줄)
   - 출력 및 로깅
   - 결과 저장 함수들
```

**최종 산출물**: 12개 파일 (6개 .cuh + 6개 .cu)

---

### 🟢 Agent 3: 플롯 및 초기화 모듈 (2,393줄)
**담당**: ldm_plot.cuh (824줄) + ldm_init.cuh (805줄) + ldm.cuh (764줄)

**분할 및 리팩토링 계획**:

```bash
# ldm_plot.cuh → 2개 파일로 분할
1. ldm_plot_vtk.cuh/cu (500줄)
   - VTK 파일 출력
   - 3D 시각화 데이터

2. ldm_plot_utils.cuh/cu (324줄)
   - 플로팅 유틸리티
   - 데이터 변환 함수

# ldm_init.cuh → 2개 파일로 분할
1. ldm_init_particles.cuh/cu (450줄)
   - 입자 초기화
   - EKI 입자 설정

2. ldm_init_config.cuh/cu (355줄)
   - 설정 파일 파싱
   - 파라미터 초기화

# ldm.cuh → 그대로 유지하되 분리
ldm.cuh/cu (764줄)
   - LDM 클래스 정의
   - 생성자/소멸자
```

**최종 산출물**: 10개 파일 (5개 .cuh + 5개 .cu)

---

### 🟡 Agent 4: IPC 및 유틸리티 모듈 (1,403줄)
**담당**: ldm_eki_ipc.cuh + memory_doctor.cuh + ldm_cram2.cuh + ldm_nuclides.cuh

**분할 및 리팩토링 계획**:

```bash
# ldm_eki_ipc.cuh → 2개 파일로 분할
1. ldm_eki_writer.cuh/cu (400줄)
   - EKIWriter 클래스
   - 공유 메모리 쓰기

2. ldm_eki_reader.cuh/cu (290줄)
   - EKIReader 클래스
   - 공유 메모리 읽기

# 나머지는 분할 없이 헤더/구현 분리
memory_doctor.cuh/cu (258줄)
ldm_cram2.cuh/cu (240줄)
ldm_nuclides.cuh/cu (215줄)
```

**최종 산출물**: 10개 파일 (5개 .cuh + 5개 .cu)

---

### 🟣 Agent 5: 설정/구조체 및 빌드 시스템 (339줄 + 빌드/Python)
**담당**: 작은 파일들 + Makefile + Python 스크립트 정리

**작업 내용**:

```bash
1. 헤더 전용 파일 정리
   - ldm_struct.cuh (179줄) - 구조체 정의, 헤더 전용
   - ldm_config.cuh (128줄) - 상수 정의, 헤더 전용
   - ldm_kernels_cram.cuh/cu (32줄) - 헤더/구현 분리

2. Makefile 대폭 수정
   # 모든 새 .cu 파일 추가 (총 30+ 파일)
   CUDA_SOURCES = \
       src/kernels/ldm_kernels_device.cu \
       src/kernels/ldm_kernels_particle.cu \
       src/kernels/ldm_kernels_particle_ens.cu \
       src/kernels/ldm_kernels_eki.cu \
       src/kernels/ldm_kernels_dump.cu \
       src/kernels/ldm_kernels_dump_ens.cu \
       src/include/ldm_mdata_loading.cu \
       src/include/ldm_mdata_processing.cu \
       ... (30+ 파일)

3. Python 스크립트 정리
   - src/eki/*.py 파일들 점검
   - util/*.py 스크립트 정리

4. CMakeLists.txt 생성 (선택적)
   - 모던 빌드 시스템 추가
```

**최종 산출물**:
- 2개 파일 (1개 .cuh + 1개 .cu)
- 업데이트된 Makefile
- 정리된 Python 스크립트
- build_files_list.txt (모든 새 파일 목록)

---

### 🔷 Agent 6: 최종 통합 관리자
**대기**: Agent 1-5 완료 후 시작

**작업 내용**:
```bash
1. 파일 구조 검증
   - 총 54개 파일 확인 (27개 .cuh + 27개 .cu)
   - include 의존성 체크

2. 메인 include 파일 생성
   # src/kernels/ldm_kernels.cuh
   #pragma once
   #include "ldm_kernels_device.cuh"
   #include "ldm_kernels_particle.cuh"
   #include "ldm_kernels_particle_ens.cuh"
   #include "ldm_kernels_eki.cuh"
   #include "ldm_kernels_dump.cuh"
   #include "ldm_kernels_dump_ens.cuh"

   # 비슷하게 다른 통합 헤더들도

3. 컴파일 테스트
   make clean && make all-targets

4. 런타임 테스트
   ./ldm-eki

5. 최종 문서화
   - FILE_STRUCTURE.md (새 파일 구조)
   - MIGRATION_GUIDE.md (이전 버전에서 마이그레이션)
```

---

## 🚀 실행 전략

### Phase 1: 완전 병렬 실행 (Agent 1-5 동시)

```python
# 동시 실행 - 각자 독립적으로 작업
agents = {
    1: "3864줄 커널 파일을 6개로 분할 + 헤더/구현 분리",
    2: "3526줄 데이터/함수를 6개로 분할 + 헤더/구현 분리",
    3: "2393줄 플롯/초기화를 5개로 분할 + 헤더/구현 분리",
    4: "1403줄 IPC/유틸을 5개로 분할 + 헤더/구현 분리",
    5: "339줄 설정 + Makefile/Python 정리"
}

# 모두 동시 시작
parallel_launch(agents)
```

### Phase 2: 통합 (Agent 6)
- Agent 1-5 완료 확인
- 파일 통합 및 최종 테스트

---

## 📊 작업 분배 요약

| Agent | 담당 라인 | 입력 파일 수 | 출력 파일 수 | 복잡도 |
|-------|-----------|-------------|-------------|---------|
| 1 | 3,864 | 1 | 12 (6 .cuh + 6 .cu) | 매우 높음 |
| 2 | 3,526 | 2 | 12 (6 .cuh + 6 .cu) | 높음 |
| 3 | 2,393 | 3 | 10 (5 .cuh + 5 .cu) | 중간 |
| 4 | 1,403 | 4 | 10 (5 .cuh + 5 .cu) | 중간 |
| 5 | 339+α | 3+α | 2 + Makefile + Python | 낮음 |
| **합계** | **11,525** | **13** | **46+ 파일** | - |

---

## 🔧 Agent별 실행 명령

### Agent 1:
```
"ldm_kernels.cuh (3864줄)를 6개 논리 파일로 분할하고 각각 헤더/구현 분리:
1. Grep으로 모든 __global__/__device__ 함수 위치 파악
2. 500줄씩 청크로 읽으며 논리적 경계 찾기
3. device/particle/particle_ens/eki/dump/dump_ens로 분할
4. 각 파일에서 .cu 구현부 분리"
```

### Agent 2:
```
"ldm_mdata.cuh와 ldm_func.cuh를 각각 3개씩 분할하고 헤더/구현 분리:
1. mdata는 loading/processing/cache로 분할
2. func는 simulation/particle/output으로 분할
3. 각 ~500-700줄 크기 유지
4. 모든 구현을 .cu로 이동"
```

### Agent 3:
```
"plot/init/ldm 파일들을 분할하고 헤더/구현 분리:
1. plot을 vtk/utils로 분할
2. init을 particles/config로 분할
3. ldm.cuh는 그대로 유지하되 .cu 분리
4. 총 5개 .cuh + 5개 .cu 생성"
```

### Agent 4:
```
"IPC 및 유틸리티 모듈 분할 및 헤더/구현 분리:
1. eki_ipc를 writer/reader로 분할
2. memory_doctor, cram2, nuclides는 그대로 유지
3. 모든 구현을 .cu로 이동
4. 총 5개 .cuh + 5개 .cu 생성"
```

### Agent 5:
```
"설정 파일 정리 및 빌드 시스템 업데이트:
1. struct/config는 헤더 전용 유지
2. kernels_cram 헤더/구현 분리
3. Makefile에 모든 새 .cu 파일 추가 (30+ 파일)
4. Python 스크립트 정리"
```

### Agent 6:
```
"모든 Agent 작업 통합 및 최종 검증:
1. 54개 파일 구조 확인
2. 통합 헤더 파일 생성
3. 컴파일 및 링킹 테스트
4. 실행 테스트
5. 최종 문서화"
```

---

## ✅ 핵심 체크리스트

### 분할 원칙
- [ ] 한 파일당 최대 800줄 (이상적: 400-600줄)
- [ ] 논리적 경계 존중 (함수 중간에 자르지 않기)
- [ ] 관련 기능끼리 그룹화

### 헤더/구현 분리
- [ ] .cuh: 선언, 인라인, 구조체만
- [ ] .cu: 모든 구현부
- [ ] `#pragma once` 추가

### 품질 보증
- [ ] 계산 로직 불변
- [ ] 컴파일 가능
- [ ] 원본과 동일한 결과

---

## 🎯 최종 목표

**Before**: 13개 거대 .cuh 파일 (평균 887줄)
**After**: 54개 작은 파일 (평균 200-400줄)

- 컴파일 시간 70% 단축 (병렬 컴파일 가능)
- 코드 가독성 대폭 향상
- 모듈별 독립적 수정 가능
- 유지보수성 극대화

---

**실행 개시: 5개 Agent 동시 시작 → 완전 병렬 처리 → Agent 6 통합**