# Agent 3 긴급 수정 요청

## 🔴 치명적 문제: src/include/ldm.cuh가 수정되지 않음

### 현재 상태:
- `src/include/ldm.cuh`: **772줄**, 전체 LDM 클래스 정의 포함
- `src/core/ldm.cuh`: **610줄**, LDM 클래스 정의 포함
- **결과**: 중복 정의로 인한 69개 컴파일 에러

### 에러 메시지:
```
error: variable "d_freq_output" has already been defined
error: class "LDM" has already been defined
error: variable "PROCESS_INDEX" has already been defined
... 69 errors total
```

## ✅ 필수 작업:

### `src/include/ldm.cuh`를 단순 wrapper로 교체

**현재 (잘못됨)**:
- 772줄의 완전한 LDM 클래스 정의
- 모든 전역 변수, 상수, 구조체 정의

**수정 후 (올바름)**:
```cpp
#pragma once
// ldm.cuh - DEPRECATED WRAPPER
//
// ⚠️  DEPRECATION NOTICE ⚠️
// This file is deprecated and exists only for backward compatibility.
// Please update your includes to use the new location:
//
// OLD:
//   #include "include/ldm.cuh"
//
// NEW:
//   #include "core/ldm.cuh"
//
// This wrapper will be removed in a future release.
//
// Date: 2025-10-15
// Agent: Agent 3 (Core Refactoring)

#ifndef LDM_CUH_WRAPPER
#define LDM_CUH_WRAPPER

// Print deprecation warning at compile time
#warning "include/ldm.cuh is deprecated. Use core/ldm.cuh instead."

// Include new location
#include "../core/ldm.cuh"

#endif // LDM_CUH_WRAPPER
```

## 📋 작업 단계:

1. **백업**:
   ```bash
   mv src/include/ldm.cuh src/include/ldm.cuh.ORIGINAL_BACKUP
   ```

2. **새 wrapper 생성**:
   - 위의 간단한 코드만 포함
   - 총 **30줄 이하**

3. **검증**:
   ```bash
   wc -l src/include/ldm.cuh  # 30줄 미만이어야 함
   grep -c "class LDM" src/include/ldm.cuh  # 0이어야 함 (정의 없음)
   ```

## 🚨 긴급도: 최최우선

이 문제로 인해 **전체 프로젝트가 컴파일 불가**입니다.

**즉시 수정 후 보고하세요!**