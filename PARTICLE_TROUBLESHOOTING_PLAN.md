# LDM-EKI 입자 시뮬레이션 문제 해결 계획

## 개요

**작성일**: 2025-01-16
**목적**: VTK 출력 및 입자 시뮬레이션 문제 해결을 위한 체계적 접근

## 문제 진단

### 현재 상황
1. **단일 모드**: VTK 파일에 입자 정보 없음
2. **앙상블 모드**: 모든 입자가 한 점에 고정됨
3. **관측 시스템**: 모든 수용체에서 관측값 0
4. **결과**: `quantitative_results.log`에 모든 값이 0으로 기록됨

### 근본 원인 분석
- 입자가 전혀 이동하지 않음 → 기상 데이터 또는 물리 엔진 문제
- VTK 출력이 비정상 → 데이터 전송 또는 포맷 문제
- 관측값이 0 → 입자가 수용체에 도달하지 못함

---

## Phase 1: 기상 데이터 검증 🌤️

### 1.1 기상 데이터 로딩 확인

#### 대상 파일
```
src/data/meteo/ldm_mdata_loading.cu
src/data/meteo/ldm_mdata_processing.cu
```

#### 확인 항목
- [ ] GFS 파일 경로 확인: `../gfsdata/0p5/`
- [ ] 파일 읽기 성공 여부
- [ ] 데이터 값 범위 검증
  - U 성분 (동서 풍속): -50 ~ 50 m/s
  - V 성분 (남북 풍속): -50 ~ 50 m/s
  - W 성분 (연직 풍속): -10 ~ 10 m/s
- [ ] GPU 메모리 전송 확인

#### 디버그 코드 추가 위치
```cpp
// ldm_mdata_loading.cu - loadMeteorologicalDataGPU() 함수 내
if (g_log_file && g_log_file->is_open()) {
    *g_log_file << "[METEO_DEBUG] Sample wind data at (180,360):\n";
    *g_log_file << "  U=" << unis[0][180][360] << " m/s\n";
    *g_log_file << "  V=" << unis[1][180][360] << " m/s\n";
    *g_log_file << "  W=" << unis[2][180][360] << " m/s\n";
}
```

### 1.2 기상 데이터 캐싱 검증

#### 대상 파일
```
src/data/meteo/ldm_mdata_cache.cu
```

#### 확인 항목
- [ ] `eki_meteo_cache` 배열 초기화
- [ ] 타임스텝별 데이터 인덱싱
- [ ] 메모리 사용량 일치 확인

---

## Phase 2: 입자 초기화 검증 🎯

### 2.1 입자 생성 위치 확인

#### 대상 파일
```
src/init/ldm_init_particles.cu
```

#### 확인 항목
- [ ] 방출원 위치 (source_x, source_y, source_z)
- [ ] 입자 초기 분포 패턴
- [ ] 입자 활성 플래그 설정
- [ ] 방출 시간 인덱스

#### 디버그 코드 추가
```cpp
// initializeParticlesEKI() 함수 내
if (g_log_file && g_log_file->is_open()) {
    *g_log_file << "[PARTICLE_INIT] First 10 particles:\n";
    for (int i = 0; i < min(10, nop); i++) {
        *g_log_file << "  P" << i << ": pos=("
                   << part[i].x << "," << part[i].y << "," << part[i].z
                   << ") flag=" << part[i].flag
                   << " timeidx=" << part[i].timeidx << "\n";
    }
}
```

### 2.2 입자 데이터 구조 검증

```cpp
struct LDMpart {
    float x, y, z;      // 격자 좌표
    float conc;         // 농도 [Bq]
    int flag;           // 활성 상태 (0=비활성, 1=활성)
    int timeidx;        // 방출 시간
    int ensemble_id;    // 앙상블 ID (앙상블 모드)
};
```

---

## Phase 3: 물리 시뮬레이션 검증 ⚛️

### 3.1 물리 모델 설정

#### 현재 설정 (`input/setting.txt`)
```
TURB=0      # 난류 OFF (설정 유지 - 입력은 완벽함)
DRYDEP=0    # 건침적 OFF
WETDEP=0    # 습침적 OFF
RADDECAY=0  # 방사성붕괴 OFF
```

#### 중요 사항
**⚠️ 입력 설정은 변경하지 않음!** TURB=0이어도 기본 이류(advection)로 입자가 이동해야 함.
문제는 코드 구현에 있을 가능성이 높음.

### 3.2 입자 업데이트 커널 검증

#### 대상 파일
```
src/kernels/particle/ldm_kernels_particle.cu
src/simulation/ldm_func_simulation.cu
```

#### 확인 항목
- [ ] `update_particles` 커널 호출 확인
- [ ] 시간 적분 스킴 (dt = 100초)
- [ ] 좌표 변환 정확성

#### 디버그 코드
```cpp
// update_particles 커널 내
if (idx < 10 && timestep % 10 == 0) {
    printf("[KERNEL_DEBUG] t=%d, p=%d: vel=(%.2f,%.2f,%.2f) pos=(%.2f,%.2f,%.2f)\n",
           timestep, idx, u_wind, v_wind, w_wind, particle.x, particle.y, particle.z);
}
```

---

## Phase 4: VTK 출력 시스템 수정 📊

### 4.1 VTK 파일 생성 검증

#### 대상 파일
```
src/visualization/ldm_plot_vtk.cu
```

#### 확인 항목
- [ ] GPU → CPU 데이터 복사
- [ ] VTK 포맷 헤더 정확성
- [ ] 입자 좌표 스케일링

### 4.2 좌표 변환 공식

```cpp
// 격자 좌표 → 위도/경도 변환
float lat = -90.0f + particle.y * 0.5f;
float lon = -179.0f + particle.x * 0.5f;
```

---

## Phase 5: 통합 테스트 전략 🧪

### 5.1 단순 테스트 케이스

#### 설정
```
입자 수: 10
바람장: 고정 (U=5 m/s, V=0, W=0)
시뮬레이션 시간: 1시간
출력: 매 타임스텝
```

### 5.2 점진적 복잡도 증가

1. **Level 1**: 고정 바람장, 10개 입자
2. **Level 2**: 난류 추가 (TURB=1)
3. **Level 3**: 실제 기상 데이터
4. **Level 4**: 전체 입자 (10,000개)
5. **Level 5**: 앙상블 모드 (100개 앙상블)

---

## Phase 6: 우선순위 수정 사항 🔧

### 즉시 수정 (Critical) 🔴

1. **코드 내 입자 이동 로직 검증**
   - TURB=0이어도 기본 이류(advection)가 작동해야 함
   - 입자 업데이트 커널이 실제로 호출되는지 확인

2. **기상 데이터 적용 검증**
   - 기상 데이터가 실제로 입자에 적용되는지 확인
   - 보간(interpolation) 함수 검증

3. **기상 데이터 검증 로그 추가**
   - 입력 설정은 건드리지 않고 코드만 수정

### 단기 수정 (High) 🟡

1. 입자 초기화 디버그 출력
2. VTK 출력 전 입자 상태 확인
3. 첫 10개 타임스텝 상세 로깅

### 중기 수정 (Medium) 🟢

1. 단위 테스트 프레임워크
2. 입자 궤적 추적 시스템
3. 성능 프로파일링

---

## 실행 체크리스트 ✅

### 빌드 및 실행 순서

```bash
# 1. 코드 수정 (입력 설정은 건드리지 않음!)
# 디버그 로그 추가 등 코드만 수정

# 2. 빌드
make clean && make

# 3. 실행
./ldm-eki

# 4. 로그 확인
tail -f logs/ldm_eki_simulation.log

# 5. 결과 확인
cat output/results/quantitative_results.log
```

### 성공 지표

- [ ] 기상 데이터 풍속 ≠ 0
- [ ] 입자 위치 변화 확인
- [ ] VTK 파일에 입자 데이터 존재
- [ ] 수용체 관측값 > 0
- [ ] quantitative_results.log에 비영 값

---

## 예상 일정

### Week 1: 진단 및 긴급 수정
- **Day 1-2**: 기상 데이터 검증
- **Day 3-4**: 물리 모델 활성화
- **Day 5-7**: 입자 이동 확인

### Week 2: 핵심 문제 해결
- **Day 8-10**: VTK 출력 수정
- **Day 11-12**: 수용체 시스템 검증
- **Day 13-14**: 통합 테스트

### Week 3: 최적화 및 검증
- **Day 15-17**: EKI 반복 테스트
- **Day 18-19**: 성능 최적화
- **Day 20-21**: 최종 문서화

---

## 디버그 모드 매크로

```cpp
// 전역 디버그 플래그 (ldm.cuh에 추가)
#define PARTICLE_DEBUG 1
#define METEO_DEBUG 1
#define VTK_DEBUG 1

#if PARTICLE_DEBUG
  #define PARTICLE_LOG(...) \
    if (g_log_file) *g_log_file << "[PARTICLE] " << __VA_ARGS__ << std::endl
#else
  #define PARTICLE_LOG(...)
#endif
```

---

## 트러블슈팅 FAQ

### Q1: 입자가 전혀 움직이지 않음
**A**: TURB=0이어도 기본 이류(advection)로 움직여야 함. 입자 업데이트 커널이 실제로 호출되는지, 기상 데이터가 적용되는지 확인.

### Q2: VTK 파일이 비어있음
**A**: GPU→CPU 데이터 복사 확인. `cudaMemcpy` 호출 후 에러 체크.

### Q3: 기상 데이터가 0
**A**: GFS 파일 경로 확인. `../gfsdata/0p5/` 디렉토리 존재 여부.

### Q4: 수용체에서 관측 안됨
**A**: 수용체 위치와 입자 방출원 위치 비교. 포착 반경 증가 시도.

---

## 참고 자료

- `CLAUDE.md`: 시스템 아키텍처 문서
- `OUTPUT_STYLE_GUIDE.md`: 출력 형식 가이드
- `PARALLEL_REFACTORING_MASTER.md`: 모듈 구조 문서

---

## 진행 상황 추적

| Phase | 상태 | 시작일 | 완료일 | 담당 | 비고 |
|-------|------|--------|--------|------|------|
| Phase 1 | 🔄 진행중 | 2025-01-16 | - | Claude | 기상 데이터 검증 |
| Phase 2 | ⏸️ 대기 | - | - | - | 입자 초기화 |
| Phase 3 | ⏸️ 대기 | - | - | - | 물리 시뮬레이션 |
| Phase 4 | ⏸️ 대기 | - | - | - | VTK 출력 |
| Phase 5 | ⏸️ 대기 | - | - | - | 통합 테스트 |
| Phase 6 | ⏸️ 대기 | - | - | - | 우선순위 수정 |

---

**마지막 업데이트**: 2025-01-16
**다음 검토일**: 2025-01-17