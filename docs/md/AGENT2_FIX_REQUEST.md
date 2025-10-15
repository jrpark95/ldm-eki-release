# Agent 2 긴급 수정 요청

## 🔴 심각한 문제 발견

**문제**: `src/simulation/ldm_func_output.cuh` 파일이 잘못 작성됨

**오류 내용**:
```
error: member function "LDM::initializeEKIObservationSystem" may not be redeclared outside its class
```

## 📋 문제 분석

### 현재 코드 (잘못됨):
```cpp
// src/simulation/ldm_func_output.cuh
#pragma once

// Forward declarations
class LDM;

void LDM::startTimer();  // ❌ 잘못된 선언
void LDM::stopTimer();   // ❌ 잘못된 선언
void LDM::initializeEKIObservationSystem();  // ❌ 잘못된 선언
```

**문제점**:
- 클래스 외부에서 `void LDM::method()`는 **정의(definition)**처럼 보임
- 헤더에서는 이것이 허용되지 않음 (redeclaration 오류)
- 이 함수들은 이미 `src/core/ldm.cuh`의 LDM 클래스 내부에 선언되어 있음

### 해결방법 1: Forward Declaration만 사용 (권장)

```cpp
// src/simulation/ldm_func_output.cuh
#pragma once

#include "ldm_config.cuh"
#include "ldm_struct.cuh"
#include <string>
#include <vector>
#include <chrono>

// Forward declarations
class LDM;

// ✅ 클래스 밖에서 선언하지 않음
// 모든 메서드는 LDM 클래스에 이미 선언되어 있음
// 구현은 ldm_func_output.cu에 있음
```

### 해결방법 2: 클래스 include (대안)

```cpp
// src/simulation/ldm_func_output.cuh
#pragma once

#include "../core/ldm.cuh"  // LDM 클래스 전체 포함

// 이 경우 forward declaration 불필요
// 메서드 재선언 불필요 (이미 클래스에 있음)
```

## ✅ 수정 요청

**Agent 2는 다음을 수행하세요**:

1. **파일 확인**:
   - `src/simulation/ldm_func_output.cuh`
   - `src/simulation/ldm_func_simulation.cuh`
   - `src/simulation/ldm_func_particle.cuh`

2. **모든 `void LDM::method()` 형태 선언 제거**:
   - 라인 23-78의 모든 메서드 "선언" 삭제
   - Forward declaration만 유지: `class LDM;`

3. **구현 파일 (.cu)은 그대로 유지**:
   - `ldm_func_output.cu`의 구현은 문제없음
   - 구현 파일에서는 `void LDM::method() { ... }` 형태가 맞음

4. **검증**:
   - 헤더에는 `class LDM;` forward declaration만
   - 구현은 .cu 파일에만 있어야 함
   - 메서드 선언은 `src/core/ldm.cuh`의 LDM 클래스 안에만 있음

## 📝 수정 예시

### Before (잘못됨):
```cpp
// ldm_func_output.cuh
#pragma once
class LDM;

void LDM::startTimer();  // ❌
void LDM::stopTimer();   // ❌
// ... 10개 이상의 메서드 "선언"
```

### After (올바름):
```cpp
// ldm_func_output.cuh
#pragma once

#include "ldm_config.cuh"
#include "ldm_struct.cuh"
#include <string>
#include <vector>
#include <chrono>

// Forward declaration only
class LDM;

// ✅ 끝! 메서드 선언은 LDM 클래스에 이미 있음
```

## 🚨 긴급도: 최우선

이 문제로 인해 **모든 파일 컴파일이 실패**하고 있습니다.
93-94개의 컴파일 에러가 이 문제에서 발생합니다.

**즉시 수정 후 보고하세요!**