# AI Chatbot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-ready AI chatbot with random sampling (Temperature, Top-K, Top-P), FastAPI server, and web UI for LLAISYS Project #3.

**Architecture:** Monolithic FastAPI server serving both API endpoints and static files. C++ sampling operator for performance. OpenAI-compatible API format with SSE streaming.

**Tech Stack:** C++ (sampling operator), Python (FastAPI, ctypes), HTML/CSS/JS (web UI), xmake (build system)

**Spec Document:** `docs/superpowers/specs/2026-03-13-chatbot-design.md`

---

## File Structure Overview

**New Files to Create:**
```
src/ops/sample/
  ├── cpu/sample_cpu.cpp       # C++ sampling implementation
  ├── cpu/sample_cpu.hpp       # C++ sampling header
  ├── op.cpp                   # Operator registration
  └── op.hpp                   # Operator interface

python/llaisys/server/
  ├── __init__.py              # Package init
  ├── main.py                  # FastAPI app entry point
  ├── routes.py                # API endpoints
  └── static/
      ├── index.html           # Chat UI
      ├── style.css            # Styling
      └── app.js               # Client logic

test/ops/sample.py             # Sampling operator tests
test/test_chatbot_api.py       # API tests
```

**Files to Modify:**
```
include/llaisys/models/qwen2.h              # Add sampling params to API
src/models/qwen2/qwen2.cpp                  # Call sampling operator
python/llaisys/libllaisys/models/qwen2.py   # Update ctypes bindings
python/llaisys/models/qwen2.py              # Update Python wrapper
xmake.lua                                   # Add sample operator build
```

---

## Chunk 1: C++ Sampling Operator Foundation

### Task 1.1: Create Sampling Operator Directory Structure

**Files:**
- Create: `src/ops/sample/cpu/sample_cpu.hpp`
- Create: `src/ops/sample/cpu/sample_cpu.cpp`
- Create: `src/ops/sample/op.hpp`
- Create: `src/ops/sample/op.cpp`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/ops/sample/cpu
```

- [ ] **Step 2: Create sample_cpu.hpp header**

Create `src/ops/sample/cpu/sample_cpu.hpp`:

```cpp
#ifndef LLAISYS_OPS_SAMPLE_CPU_HPP
#define LLAISYS_OPS_SAMPLE_CPU_HPP

#include "../../../tensor/tensor.hpp"

namespace llaisys {
namespace ops {
namespace sample {
namespace cpu {

void sample_cpu(
    tensor_t sampled_token,  // Output: [1] int64 tensor
    tensor_t logits,         // Input: [vocab_size] float tensor
    float temperature,       // Sampling temperature (0.1-2.0)
    int top_k,              // Top-K filtering (0 = disabled)
    float top_p             // Top-P (nucleus) filtering (0.0-1.0)
);

}  // namespace cpu
}  // namespace sample
}  // namespace ops
}  // namespace llaisys

#endif  // LLAISYS_OPS_SAMPLE_CPU_HPP
```

- [ ] **Step 3: Create op.hpp interface**

Create `src/ops/sample/op.hpp`:

```cpp
#ifndef LLAISYS_OPS_SAMPLE_OP_HPP
#define LLAISYS_OPS_SAMPLE_OP_HPP

#include "../../tensor/tensor.hpp"

namespace llaisys {
namespace ops {
namespace sample {

void sample(
    tensor_t sampled_token,
    tensor_t logits,
    float temperature,
    int top_k,
    float top_p
);

}  // namespace sample
}  // namespace ops
}  // namespace llaisys

#endif  // LLAISYS_OPS_SAMPLE_OP_HPP
```

- [ ] **Step 4: Commit directory structure**

```bash
git add src/ops/sample/
git commit -m "feat(ops): add sampling operator directory structure

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

### Task 1.2: Implement Temperature Scaling (TDD)

**Files:**
- Create: `test/ops/sample.py`
- Modify: `src/ops/sample/cpu/sample_cpu.cpp`

- [ ] **Step 1: Write failing test for temperature scaling**

Create `test/ops/sample.py`:

```python
import numpy as np
import torch
from test_utils import *
import llaisys

def test_temperature_scaling():
    """Test that temperature correctly scales logits"""
    # Create simple logits
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    temperature = 2.0

    # Expected: logits / temperature
    expected_scaled = logits / temperature

    # Create tensors
    logits_tensor = llaisys.tensor.from_numpy(logits, llaisys.DeviceType.CPU)
    output_tensor = llaisys.tensor.zeros([1], llaisys.DataType.I64, llaisys.DeviceType.CPU)

    # Call sample operator (will fail - not implemented yet)
    llaisys.ops.sample(output_tensor, logits_tensor, temperature, 0, 1.0)

    # For now, just check it doesn't crash
    # We'll add proper validation after implementation
    print("Temperature scaling test setup complete")

if __name__ == "__main__":
    test_temperature_scaling()
    print("✓ Temperature scaling test passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test/ops/sample.py`
Expected: ImportError or AttributeError (sample operator not exposed yet)

- [ ] **Step 3: Create minimal sample_cpu.cpp stub**

Create `src/ops/sample/cpu/sample_cpu.cpp`:

```cpp
#include "sample_cpu.hpp"
#include "../../../utils/type_cast.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

namespace llaisys {
namespace ops {
namespace sample {
namespace cpu {

void sample_cpu(
    tensor_t sampled_token,
    tensor_t logits,
    float temperature,
    int top_k,
    float top_p
) {
    // Get logits data
    auto dtype = logits->meta.dtype;
    size_t vocab_size = logits->meta.shape[0];

    // Convert logits to float32 for processing
    std::vector<float> logits_f32(vocab_size);

    // Type-aware loading
    if (dtype == DataType::F32) {
        float* data = static_cast<float*>(logits->data());
        for (size_t i = 0; i < vocab_size; i++) {
            logits_f32[i] = data[i];
        }
    } else if (dtype == DataType::F16) {
        // Use type_cast utility for F16
        auto* data = static_cast<uint16_t*>(logits->data());
        for (size_t i = 0; i < vocab_size; i++) {
            logits_f32[i] = fp16_to_fp32(data[i]);
        }
    } else if (dtype == DataType::BF16) {
        // Use type_cast utility for BF16
        auto* data = static_cast<uint16_t*>(logits->data());
        for (size_t i = 0; i < vocab_size; i++) {
            logits_f32[i] = bf16_to_fp32(data[i]);
        }
    }

    // Step 1: Apply temperature scaling
    for (size_t i = 0; i < vocab_size; i++) {
        logits_f32[i] /= temperature;
    }

    // TODO: Implement Top-K, Top-P, softmax, and sampling
    // For now, just return argmax
    size_t max_idx = 0;
    float max_val = logits_f32[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits_f32[i] > max_val) {
            max_val = logits_f32[i];
            max_idx = i;
        }
    }

    // Write result
    int64_t* output = static_cast<int64_t*>(sampled_token->data());
    *output = static_cast<int64_t>(max_idx);
}

}  // namespace cpu
}  // namespace sample
}  // namespace ops
}  // namespace llaisys
```

- [ ] **Step 4: Create op.cpp dispatcher**

Create `src/ops/sample/op.cpp`:

```cpp
#include "op.hpp"
#include "cpu/sample_cpu.hpp"

namespace llaisys {
namespace ops {
namespace sample {

void sample(
    tensor_t sampled_token,
    tensor_t logits,
    float temperature,
    int top_k,
    float top_p
) {
    // For now, only CPU implementation
    auto device = logits->device();
    if (device == DeviceType::CPU) {
        cpu::sample_cpu(sampled_token, logits, temperature, top_k, top_p);
    } else {
        throw std::runtime_error("Sample operator only supports CPU for now");
    }
}

}  // namespace sample
}  // namespace ops
}  // namespace llaisys
```

- [ ] **Step 5: Add C API exposure**

Check existing pattern in `src/llaisys/ops.cpp` and add:

```cpp
// In src/llaisys/ops.cpp (or create if doesn't exist)
#include "../ops/sample/op.hpp"

extern "C" {

__export void llaisysSample(
    llaisysTensor_t sampled_token,
    llaisysTensor_t logits,
    float temperature,
    int top_k,
    float top_p
) {
    llaisys::ops::sample::sample(
        sampled_token, logits, temperature, top_k, top_p
    );
}

}  // extern "C"
```

- [ ] **Step 6: Update xmake.lua to build sample operator**

Add to `xmake.lua` in the appropriate section (follow pattern from other operators):

```lua
-- Add sample operator source files
target("llaisys")
    add_files("src/ops/sample/op.cpp")
    add_files("src/ops/sample/cpu/sample_cpu.cpp")
```

- [ ] **Step 7: Build C++ code**

Run: `xmake && xmake install`
Expected: Successful compilation

- [ ] **Step 8: Add Python ctypes binding**

Add to `python/llaisys/libllaisys/ops.py`:

```python
# Add to imports
from .llaisys_types import llaisysTensor_t
from . import LIB_LLAISYS
import ctypes

# Add function signature
LIB_LLAISYS.llaisysSample.argtypes = [
    llaisysTensor_t,  # sampled_token
    llaisysTensor_t,  # logits
    ctypes.c_float,   # temperature
    ctypes.c_int,     # top_k
    ctypes.c_float,   # top_p
]
LIB_LLAISYS.llaisysSample.restype = None
```

- [ ] **Step 9: Add Python wrapper**

Add to `python/llaisys/ops.py`:

```python
def sample(sampled_token, logits, temperature=0.8, top_k=50, top_p=0.9):
    """
    Sample a token from logits using temperature, top-k, and top-p filtering.

    Args:
        sampled_token: Output tensor [1] of type int64
        logits: Input tensor [vocab_size] of type float
        temperature: Sampling temperature (0.1-2.0), default 0.8
        top_k: Top-K filtering (0 = disabled), default 50
        top_p: Top-P filtering (0.0-1.0), default 0.9
    """
    from .libllaisys import LIB_LLAISYS
    LIB_LLAISYS.llaisysSample(
        sampled_token._tensor,
        logits._tensor,
        float(temperature),
        int(top_k),
        float(top_p)
    )
```

- [ ] **Step 10: Install Python package**

Run: `pip install ./python/`
Expected: Successful installation

- [ ] **Step 11: Run test to verify it passes**

Run: `python test/ops/sample.py`
Expected: Test passes (basic functionality works)

- [ ] **Step 12: Commit temperature scaling implementation**

```bash
git add src/ops/sample/ src/llaisys/ops.cpp python/llaisys/ test/ops/sample.py xmake.lua
git commit -m "feat(ops): implement basic sampling operator with temperature scaling

- Add C++ sampling operator with temperature support
- Add Python bindings and wrapper
- Add basic test for temperature scaling
- Currently returns argmax, will add Top-K/Top-P next

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```


### Task 1.3: Implement Complete Sampling Algorithm

**Files:**
- Modify: `test/ops/sample.py`
- Modify: `src/ops/sample/cpu/sample_cpu.cpp`

- [ ] **Step 1: Add comprehensive tests**

Add to `test/ops/sample.py` tests for Top-K, Top-P, and multinomial sampling (see spec for details)

- [ ] **Step 2: Implement complete sampling algorithm**

Update `src/ops/sample/cpu/sample_cpu.cpp` with:
- Top-K filtering using partial_sort
- Top-P filtering with cumulative probability
- Softmax with numerical stability
- Multinomial sampling with thread-local RNG

- [ ] **Step 3: Build, test, and commit**

```bash
xmake && xmake install && pip install ./python/
python test/ops/sample.py
git add src/ops/sample/ test/ops/sample.py
git commit -m "feat(ops): complete sampling operator with Top-K, Top-P, and multinomial sampling"
```

---

## Chunk 2: Model API Integration

### Task 2.1: Update Model Header with Sampling Parameters

**Files:**
- Modify: `include/llaisys/models/qwen2.h:40`

- [ ] **Step 1: Update function signature**

```cpp
// Change from:
__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

// To:
__export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model * model,
    int64_t * token_ids,
    size_t ntoken,
    float temperature,
    int top_k,
    float top_p
);
```

- [ ] **Step 2: Commit header change**

```bash
git add include/llaisys/models/qwen2.h
git commit -m "feat(model): add sampling parameters to Qwen2 infer API"
```

### Task 2.2: Update Model Implementation

**Files:**
- Modify: `src/models/qwen2/qwen2.cpp` (or wherever model implementation is)

- [ ] **Step 1: Find model implementation file**

Run: `find src/ -name "*qwen2*.cpp" -o -name "*qwen2*.cc"`

- [ ] **Step 2: Update implementation to call sample operator**

Modify the inference function to:
1. Get logits from model forward pass
2. Call `llaisys::ops::sample::sample()` with parameters
3. Return sampled token

- [ ] **Step 3: Build and test**

```bash
xmake && xmake install
python test/test_infer.py --model /path/to/model --test
```

- [ ] **Step 4: Commit implementation**

```bash
git add src/models/qwen2/
git commit -m "feat(model): integrate sampling operator into Qwen2 inference"
```

### Task 2.3: Update Python Bindings

**Files:**
- Modify: `python/llaisys/libllaisys/models/qwen2.py`
- Modify: `python/llaisys/models/qwen2.py`

- [ ] **Step 1: Update ctypes binding**

In `python/llaisys/libllaisys/models/qwen2.py`, update function signature:

```python
LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
    ctypes.c_void_p,  # model
    ctypes.POINTER(ctypes.c_int64),  # token_ids
    ctypes.c_size_t,  # ntoken
    ctypes.c_float,   # temperature
    ctypes.c_int,     # top_k
    ctypes.c_float,   # top_p
]
```

- [ ] **Step 2: Update Python wrapper**

In `python/llaisys/models/qwen2.py`, update `generate()` method to pass sampling parameters to C++ backend.

- [ ] **Step 3: Install and test**

```bash
pip install ./python/
python test/test_infer.py --model /path/to/model --temperature 0.8 --top_k 50
```

- [ ] **Step 4: Commit Python changes**

```bash
git add python/llaisys/
git commit -m "feat(python): update Qwen2 bindings for sampling parameters"
```

---

## Chunk 3: FastAPI Server

### Task 3.1: Install Dependencies

- [ ] **Step 1: Install FastAPI and dependencies**

```bash
pip install fastapi uvicorn sse-starlette
```

- [ ] **Step 2: Create requirements file**

Create `python/llaisys/server/requirements.txt`:
```
fastapi>=0.104.0
uvicorn>=0.24.0
sse-starlette>=1.6.0
```

### Task 3.2: Create Server Structure

**Files:**
- Create: `python/llaisys/server/__init__.py`
- Create: `python/llaisys/server/main.py`
- Create: `python/llaisys/server/routes.py`

- [ ] **Step 1: Create directory and __init__.py**

```bash
mkdir -p python/llaisys/server
touch python/llaisys/server/__init__.py
```

- [ ] **Step 2: Create main.py with FastAPI app**

See spec for complete implementation. Key points:
- FastAPI app initialization
- Model loading at startup
- Static file serving
- CORS configuration

- [ ] **Step 3: Create routes.py with endpoints**

Implement:
- POST /v1/chat/completions (streaming and non-streaming)
- GET / (serve index.html)
- GET /health

- [ ] **Step 4: Test server startup**

```bash
python -m llaisys.server.main --model /path/to/model --port 8000
curl http://localhost:8000/health
```

- [ ] **Step 5: Commit server code**

```bash
git add python/llaisys/server/
git commit -m "feat(server): add FastAPI server with chat completion endpoint"
```

### Task 3.3: Implement Request Validation

**Files:**
- Modify: `python/llaisys/server/routes.py`

- [ ] **Step 1: Add Pydantic models for validation**

```python
from pydantic import BaseModel, Field, validator

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=128, gt=0)
    stream: bool = False
```

- [ ] **Step 2: Add error handling**

Implement proper error responses for validation failures, model errors, etc.

- [ ] **Step 3: Test validation**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [], "temperature": 5.0}'
# Should return 400 error
```

- [ ] **Step 4: Commit validation**

```bash
git add python/llaisys/server/routes.py
git commit -m "feat(server): add request validation and error handling"
```

### Task 3.4: Implement SSE Streaming

**Files:**
- Modify: `python/llaisys/server/routes.py`

- [ ] **Step 1: Implement streaming response generator**

```python
from sse_starlette.sse import EventSourceResponse

async def generate_stream(tokens, model, tokenizer, max_tokens, temperature, top_k, top_p):
    for i in range(max_tokens):
        next_token = model.generate(tokens, 1, temperature, top_k, top_p)[-1]
        tokens.append(next_token)

        text = tokenizer.decode([next_token], skip_special_tokens=True)
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": text}}]
        }
        yield {"data": json.dumps(chunk)}

        if next_token == model._end_token:
            break

    yield {"data": "[DONE]"}
```

- [ ] **Step 2: Test streaming**

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

- [ ] **Step 3: Commit streaming**

```bash
git add python/llaisys/server/routes.py
git commit -m "feat(server): implement SSE streaming for chat completions"
```

---

## Chunk 4: Web UI

### Task 4.1: Create HTML Structure

**Files:**
- Create: `python/llaisys/server/static/index.html`

- [ ] **Step 1: Create static directory**

```bash
mkdir -p python/llaisys/server/static
```

- [ ] **Step 2: Create index.html**

Implement chat interface with:
- Header with title and settings button
- Scrollable message container
- Input textarea and send button
- Settings panel (collapsible)

See spec for complete HTML structure.

- [ ] **Step 3: Test HTML loads**

Open `http://localhost:8000` in browser
Expected: UI displays correctly

- [ ] **Step 4: Commit HTML**

```bash
git add python/llaisys/server/static/index.html
git commit -m "feat(ui): add chat interface HTML structure"
```

### Task 4.2: Add CSS Styling

**Files:**
- Create: `python/llaisys/server/static/style.css`

- [ ] **Step 1: Create style.css**

Implement styling for:
- Layout and responsiveness
- Message bubbles (user/assistant)
- Input area
- Settings panel
- Loading indicators

See spec for color scheme and design principles.

- [ ] **Step 2: Test styling**

Refresh browser, verify UI looks good

- [ ] **Step 3: Commit CSS**

```bash
git add python/llaisys/server/static/style.css
git commit -m "feat(ui): add modern CSS styling for chat interface"
```

### Task 4.3: Implement Client-Side Logic

**Files:**
- Create: `python/llaisys/server/static/app.js`

- [ ] **Step 1: Create app.js**

Implement:
- `sendMessage()` - POST to /v1/chat/completions
- `handleStream()` - Process SSE events using fetch with ReadableStream
- `addMessage()` - Add message bubble to chat
- `updateSettings()` - Update sampling parameters
- `autoScroll()` - Scroll to latest message

Note: Use fetch() with ReadableStream for SSE, not EventSource (which only supports GET).

- [ ] **Step 2: Test end-to-end**

Send message in browser, verify:
- Message appears in chat
- Streaming response displays token-by-token
- Auto-scroll works
- Settings can be adjusted

- [ ] **Step 3: Commit JavaScript**

```bash
git add python/llaisys/server/static/app.js
git commit -m "feat(ui): implement client-side logic with SSE streaming"
```

---

## Chunk 5: Testing and Integration

### Task 5.1: Write API Tests

**Files:**
- Create: `test/test_chatbot_api.py`

- [ ] **Step 1: Create test file**

Implement tests for:
- Request validation (invalid parameters)
- Non-streaming response format
- Streaming response format
- Model integration
- Error handling

See spec for detailed test cases.

- [ ] **Step 2: Run tests**

```bash
python test/test_chatbot_api.py
```

- [ ] **Step 3: Commit tests**

```bash
git add test/test_chatbot_api.py
git commit -m "test(api): add comprehensive API tests"
```

### Task 5.2: Manual Testing

- [ ] **Step 1: Start server**

```bash
python -m llaisys.server.main --model /path/to/model --port 8000
```

- [ ] **Step 2: Run through manual testing checklist**

Follow checklist in spec:
- Basic functionality
- Settings panel
- Error handling
- Browser compatibility
- Visual quality

- [ ] **Step 3: Document any issues found**

Create issues or fix immediately if minor.

### Task 5.3: Final Integration Test

- [ ] **Step 1: Test complete workflow**

1. Start server
2. Open browser to http://localhost:8000
3. Send message "Hello, who are you?"
4. Verify streaming response
5. Adjust temperature to 0.1, send another message
6. Verify response is more deterministic
7. Adjust temperature to 2.0, send another message
8. Verify response is more creative/random

- [ ] **Step 2: Verify all requirements met**

Check against success criteria in spec:
- Random sampling works correctly
- FastAPI server runs stably
- Web UI provides smooth experience
- Streaming works without lag

- [ ] **Step 3: Create final commit**

```bash
git add .
git commit -m "feat(chatbot): complete Project #3 - AI chatbot implementation

- Implement random sampling (Temperature, Top-K, Top-P)
- Build FastAPI server with OpenAI-compatible API
- Create web UI with SSE streaming
- Add comprehensive tests
- All requirements met

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Summary

This plan implements LLAISYS Project #3 in 5 chunks:

1. **C++ Sampling Operator** - Core sampling algorithm with TDD
2. **Model API Integration** - Connect sampling to Qwen2 model
3. **FastAPI Server** - HTTP API with streaming support
4. **Web UI** - Chat interface with HTML/CSS/JS
5. **Testing & Integration** - Comprehensive testing and validation

Each task follows TDD principles with frequent commits. Total estimated time: 8-12 hours for experienced developer.

**Next Steps:**
- Use @superpowers:subagent-driven-development to execute this plan
- Each chunk will be reviewed before proceeding to next
- Manual testing checklist at the end ensures quality

