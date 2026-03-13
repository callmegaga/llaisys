# AI Chatbot Design Specification

**Project:** LLAISYS Project #3 - AI Chatbot
**Date:** 2026-03-13
**Status:** Approved

## Overview

Build an AI chatbot for LLAISYS that enables live conversations with a single user. The chatbot will implement random sampling (Temperature, Top-K, Top-P), provide a web-based chat interface, and follow OpenAI-compatible API conventions for future extensibility.

## Requirements

### Core Features
- Random sampling with Temperature, Top-K, and Top-P support
- FastAPI-based HTTP server with OpenAI-compatible endpoints
- Web UI with HTML/CSS/JS for chat interaction
- Streaming responses via Server-Sent Events (SSE)
- Single-user, single-conversation support (no session management)

### Quality Requirements
- Production-ready: proper error handling, input validation
- Clean, modern UI with good user experience
- Robust implementation following best practices
- Comprehensive testing strategy

### Out of Scope (Future Enhancements)
- Multi-user support
- Session management (multiple conversations)
- Edit/regenerate functionality
- KV-Cache prefix matching pool

## Architecture

### Approach: Monolithic FastAPI Server

We chose a monolithic architecture where a single FastAPI application serves both API endpoints and static files. This approach provides:
- Simple deployment (one server process)
- Direct integration with existing LLAISYS Python code
- No CORS complexity
- Easy development and debugging
- Perfect fit for single-user requirements

### System Components

1. **C++ Backend Layer** (New sampling operator)
   - `src/ops/sample/` - Random sampling operator with Temperature, Top-K, Top-P
   - Modified `llaisysQwen2ModelInfer` to accept sampling parameters
   - Returns sampled token based on logits and sampling strategy

2. **Python API Layer** (FastAPI server)
   - `python/llaisys/server/main.py` - FastAPI application entry point
   - `python/llaisys/server/routes.py` - API endpoints following OpenAI format
   - Model loaded once at startup, kept in memory
   - Handles streaming via Server-Sent Events (SSE)

3. **Web UI Layer** (Static files)
   - `python/llaisys/server/static/index.html` - Chat interface
   - `python/llaisys/server/static/style.css` - Styling
   - `python/llaisys/server/static/app.js` - Client-side logic for SSE streaming

### Request Flow

```
User Browser → FastAPI → Python Qwen2 Model → C++ Backend → Sample Operator → Token
                ↓                                                                ↓
            SSE Stream ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

### Key Design Decisions

- Single model instance (no concurrent requests for now, as per single-user requirement)
- Blocking endpoint during generation (simpler than async queue)
- Sampling happens in C++ for performance
- OpenAI-compatible API format for future extensibility

## Component Details

### 1. C++ Sampling Operator

**Location:** `src/ops/sample/`

**Purpose:** Implement Temperature, Top-K, and Top-P sampling on logits tensor.

**Interface:**
```cpp
void sample(tensor_t sampled_token,    // Output: [1] int64 tensor
            tensor_t logits,           // Input: [vocab_size] float tensor
            float temperature,         // Sampling temperature (0.1-2.0)
            int top_k,                 // Top-K filtering (0 = disabled)
            float top_p);              // Top-P (nucleus) filtering (0.0-1.0)
```

**Algorithm:**
1. Apply temperature scaling: `logits = logits / temperature`
2. Apply Top-K filtering: zero out all but top K logits
3. Apply Top-P filtering: zero out tokens outside cumulative probability threshold
4. Compute softmax to get probabilities
5. Sample from multinomial distribution using random number generator
6. Return sampled token index

**Modified Model API:**
```cpp
// Update signature in include/llaisys/models/qwen2.h
int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model* model,
    int64_t* token_ids,
    size_t ntoken,
    float temperature,  // NEW
    int top_k,         // NEW
    float top_p        // NEW
);
```

**Implementation Notes:**
- Support Float32, Float16, and BFloat16 data types
- Use efficient sorting for Top-K (partial sort or heap)
- Use cumulative sum for Top-P filtering
- Random number generation: use C++ `<random>` library with thread-local generator

### 2. FastAPI Server

**Location:** `python/llaisys/server/`

#### main.py - Application Setup

**Responsibilities:**
- Initialize FastAPI app
- Load Qwen2 model at startup (single instance)
- Load tokenizer from model directory
- Serve static files from `/static` directory
- Configure logging and error handlers

**Startup Sequence:**
```python
app = FastAPI(title="LLAISYS Chatbot")
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    model = llaisys.models.Qwen2(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
```

#### routes.py - API Endpoints

**POST /v1/chat/completions** - Main chat endpoint

Request format (OpenAI-compatible):
```json
{
  "messages": [
    {"role": "user", "content": "Hello, who are you?"}
  ],
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9,
  "max_tokens": 128,
  "stream": true
}
```

Response format (streaming):
```
data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hello"}}]}
data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "!"}}]}
data: [DONE]
```

Response format (non-streaming):
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! I'm an AI assistant."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
}
```

**GET /** - Serve index.html

**GET /health** - Health check endpoint
```json
{"status": "ok", "model_loaded": true}
```

### 3. Web UI

**Location:** `python/llaisys/server/static/`

#### index.html - Chat Interface Structure

**Layout:**
```
┌─────────────────────────────────────┐
│  LLAISYS Chatbot         [Settings] │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │ User: Hello                 │   │
│  └─────────────────────────────┘   │
│                                     │
│       ┌─────────────────────────┐  │
│       │ Assistant: Hi there!    │  │
│       └─────────────────────────┘  │
│                                     │
├─────────────────────────────────────┤
│ [Type your message...]      [Send] │
└─────────────────────────────────────┘
```

**Components:**
- Header with title and settings button
- Scrollable message container
- Message bubbles (user: right-aligned, assistant: left-aligned)
- Input textarea with send button
- Settings panel (collapsible):
  - Temperature slider (0.1 - 2.0)
  - Top-K slider (0 - 100)
  - Top-P slider (0.0 - 1.0)
  - Max tokens input

#### app.js - Client-Side Logic

**Key Functions:**
- `sendMessage()` - Send POST request to /v1/chat/completions
- `handleStream()` - Process SSE events and update UI
- `addMessage()` - Add message bubble to chat
- `updateSettings()` - Update sampling parameters
- `autoScroll()` - Scroll to latest message

**SSE Handling:**
```javascript
const eventSource = new EventSource('/v1/chat/completions');
eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  const data = JSON.parse(event.data);
  appendToMessage(data.choices[0].delta.content);
};
```

#### style.css - Styling

**Design Principles:**
- Clean, modern interface
- Good contrast and readability
- Responsive layout (works on mobile)
- Smooth animations for message appearance
- Loading indicators during generation

**Color Scheme:**
- User messages: Blue background (#007bff)
- Assistant messages: Gray background (#f1f3f5)
- Background: Light gray (#ffffff)
- Text: Dark gray (#212529)

## Data Flow

### Request Flow (Detailed)

**1. User sends message from browser:**
```javascript
// app.js sends POST request
fetch('/v1/chat/completions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    messages: [{role: 'user', content: 'Hello'}],
    temperature: 0.8,
    top_k: 50,
    top_p: 0.9,
    max_tokens: 128,
    stream: true
  })
})
```

**2. FastAPI receives request (routes.py):**
- Validate request parameters (temperature range, top_k > 0, etc.)
- Apply chat template to messages using tokenizer
- Encode text to token IDs
- Initialize Server-Sent Events response for streaming

**3. Generation loop (Python):**
```python
for i in range(max_tokens):
    # Call C++ backend with sampling params
    next_token = model.generate(
        tokens,
        max_new_tokens=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    tokens.append(next_token)

    # Decode and stream token
    text = tokenizer.decode([next_token])
    yield f"data: {json.dumps(chunk)}\n\n"

    # Check for end token
    if next_token == model.end_token:
        break

yield "data: [DONE]\n\n"
```

**4. C++ backend processes request:**
- Model forward pass generates logits
- Call sample operator with logits and parameters
- Return sampled token to Python

**5. Client receives and displays:**
- EventSource receives SSE events
- Parse JSON chunks
- Append text to assistant message bubble
- Auto-scroll to show new content

### State Management

**Server State:**
- Model instance (loaded at startup, persistent)
- Tokenizer instance (loaded at startup, persistent)
- KV cache (managed by C++ model, cleared between conversations)

**Client State:**
- Message history (array of {role, content} objects)
- Current settings (temperature, top_k, top_p, max_tokens)
- UI state (loading indicator, error messages)

## Error Handling

### Input Validation

**Parameter Constraints:**
- **Temperature:** Must be in range (0.1, 2.0), default 0.8
  - Too low (<0.1): deterministic, boring responses
  - Too high (>2.0): incoherent, random responses
- **Top-K:** Must be >= 0 (0 = disabled), default 50
  - 0 means no Top-K filtering
  - Typical range: 20-100
- **Top-P:** Must be in range (0.0, 1.0), default 0.9
  - 0.0 means only highest probability token
  - 1.0 means no filtering
- **Max tokens:** Must be > 0 and <= model.max_seq_len, default 128
- **Messages:** Must be non-empty array with valid role/content

**Validation Response:**
```json
{
  "error": {
    "message": "Temperature must be between 0.1 and 2.0",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_value"
  }
}
```
HTTP Status: 400 Bad Request

### Runtime Error Handling

**Model Loading Errors:**
- Catch exceptions during model initialization in startup event
- Log detailed error messages with stack trace
- Return HTTP 503 (Service Unavailable) for all requests if model fails to load
- Health endpoint returns `{"status": "error", "model_loaded": false}`

**Generation Errors:**
- Wrap generation loop in try-except block
- If C++ backend crashes or returns invalid token:
  - Stop generation gracefully
  - In streaming mode: send error chunk then [DONE]
  - In non-streaming mode: return HTTP 500 with error message
- Log error details for debugging

**Example Error Response (Streaming):**
```
data: {"error": {"message": "Generation failed", "type": "server_error"}}
data: [DONE]
```

**Example Error Response (Non-streaming):**
```json
{
  "error": {
    "message": "Internal server error during generation",
    "type": "server_error"
  }
}
```
HTTP Status: 500 Internal Server Error

### Client-Side Error Handling

**Network Errors:**
- Connection timeout: Show "Connection lost, please try again" message
- SSE connection drop: Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, max 30s)
- Failed to fetch: Display error in chat with retry button

**Invalid Response:**
- Parse error: Display "Invalid response from server" message
- Unexpected format: Log to console, show generic error to user

**User Experience:**
- Disable send button during generation
- Show loading indicator (typing animation)
- Allow cancellation of ongoing generation (close EventSource)
- Clear error messages on new request

### Resource Management

**Server Resources:**
- Model stays loaded in memory (single instance)
- KV cache managed by C++ backend (automatically cleared between conversations)
- Proper cleanup on server shutdown (SIGTERM handler)

**Memory Considerations:**
- Monitor memory usage during long conversations
- Consider adding conversation length limit (e.g., max 50 messages)
- Future enhancement: implement KV cache eviction strategy

## Testing Strategy

### Unit Tests

#### C++ Sampling Operator Tests
**Location:** `test/ops/sample.py`

**Test Cases:**
1. **Temperature Scaling**
   - Verify logits are divided by temperature
   - Test edge cases: temperature=1.0 (no scaling), temperature=0.5 (sharper), temperature=2.0 (flatter)

2. **Top-K Filtering**
   - Verify only top K tokens have non-zero probability
   - Test K=0 (disabled), K=1 (argmax), K=vocab_size (no filtering)
   - Verify correct token selection with different K values

3. **Top-P Filtering**
   - Verify cumulative probability threshold is respected
   - Test P=0.0 (only highest), P=1.0 (no filtering), P=0.9 (typical)
   - Verify tokens are sorted by probability before filtering

4. **Combined Filtering**
   - Test Temperature + Top-K
   - Test Temperature + Top-P
   - Test all three together

5. **Data Type Support**
   - Test with Float32, Float16, BFloat16 logits
   - Verify correct type casting and precision

6. **Comparison with PyTorch**
   - Generate samples with same seed
   - Compare distributions statistically
   - Verify sampling is correct (not just deterministic match)

**Test Execution:**
```bash
python test/ops/sample.py
```

#### Python API Tests
**Location:** `test/test_chatbot_api.py`

**Test Cases:**
1. **Request Validation**
   - Invalid temperature → 400 error
   - Invalid top_k → 400 error
   - Invalid top_p → 400 error
   - Empty messages → 400 error
   - Missing required fields → 400 error

2. **Non-Streaming Response**
   - Send request with stream=false
   - Verify response format matches OpenAI spec
   - Check all required fields present
   - Verify token counts are reasonable

3. **Streaming Response**
   - Send request with stream=true
   - Verify SSE format (data: prefix, [DONE] terminator)
   - Parse all chunks successfully
   - Reconstruct full message from chunks

4. **Model Integration**
   - End-to-end generation with real model
   - Verify generated text is coherent
   - Test with different sampling parameters
   - Verify end token stops generation

5. **Error Handling**
   - Model not loaded → 503 error
   - Generation failure → 500 error
   - Verify error response format

**Test Execution:**
```bash
python test/test_chatbot_api.py
```

### Integration Tests

#### End-to-End Chat Flow
**Location:** `test/test_chatbot_e2e.py`

**Test Scenario:**
1. Start server with test model
2. Send chat request via HTTP client
3. Verify streaming response arrives
4. Check response follows OpenAI format
5. Verify generated text is reasonable (not empty, not gibberish)
6. Send follow-up message
7. Verify conversation context is maintained (via KV cache)

**Test Execution:**
```bash
# Start server in background
python -m llaisys.server.main --model /path/to/model --port 8000 &
SERVER_PID=$!

# Run tests
python test/test_chatbot_e2e.py

# Cleanup
kill $SERVER_PID
```

#### Error Scenario Tests

**Test Cases:**
1. **Invalid Parameters**
   - Send request with temperature=5.0
   - Verify 400 error with helpful message

2. **Model Not Loaded**
   - Start server without model
   - Send request
   - Verify 503 error

3. **Network Interruption**
   - Start streaming request
   - Close connection mid-stream
   - Verify server handles gracefully (no crash)

### Manual Testing

#### Web UI Testing Checklist

**Basic Functionality:**
- [ ] Open browser to `http://localhost:8000`
- [ ] UI loads correctly (no console errors)
- [ ] Send message "Hello, who are you?"
- [ ] Verify streaming response appears token-by-token
- [ ] Verify message bubbles display correctly (user right, assistant left)
- [ ] Verify auto-scroll works

**Settings Panel:**
- [ ] Open settings panel
- [ ] Adjust temperature slider → verify value updates
- [ ] Adjust top_k slider → verify value updates
- [ ] Adjust top_p slider → verify value updates
- [ ] Send message with different settings
- [ ] Verify response quality changes (e.g., temperature=0.1 vs 2.0)

**Error Handling:**
- [ ] Disconnect network → verify error message
- [ ] Reconnect → verify can send messages again
- [ ] Send very long message → verify handles gracefully
- [ ] Rapid-fire multiple messages → verify queuing/blocking works

**Browser Compatibility:**
- [ ] Test on Chrome
- [ ] Test on Firefox
- [ ] Test on Safari (if available)
- [ ] Test on mobile browser

**Visual Quality:**
- [ ] Check responsive layout (resize window)
- [ ] Verify colors and contrast are good
- [ ] Check loading indicators appear
- [ ] Verify smooth animations

### Performance Testing (Optional)

#### Metrics to Measure

**Generation Speed:**
- Tokens per second (with different sampling parameters)
- Compare with PyTorch baseline
- Measure overhead of sampling operator

**Latency:**
- Time to first token (TTFT)
- Time between tokens
- End-to-end request latency

**Resource Usage:**
- Memory consumption (model + KV cache)
- CPU usage during generation
- Network bandwidth for streaming

**Test Execution:**
```bash
python test/benchmark_chatbot.py --model /path/to/model --num_requests 100
```

### Continuous Integration

**GitHub Actions Workflow:**
```yaml
name: Chatbot Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Build C++ backend
        run: xmake && xmake install
      - name: Run operator tests
        run: python test/ops/sample.py
      - name: Run API tests
        run: python test/test_chatbot_api.py
```

## Implementation Plan Overview

### Phase 1: C++ Sampling Operator
1. Create `src/ops/sample/` directory structure
2. Implement sampling algorithm in `cpu/sample_cpu.cpp`
3. Add operator registration and API exposure
4. Write unit tests in `test/ops/sample.py`
5. Update xmake.lua for compilation

### Phase 2: Model API Updates
1. Modify `include/llaisys/models/qwen2.h` to add sampling parameters
2. Update C++ model implementation to call sample operator
3. Update Python ctypes bindings in `python/llaisys/libllaisys/models/qwen2.py`
4. Update Python wrapper in `python/llaisys/models/qwen2.py`
5. Test with existing `test/test_infer.py`

### Phase 3: FastAPI Server
1. Create `python/llaisys/server/` directory
2. Implement `main.py` with FastAPI app and startup logic
3. Implement `routes.py` with chat completion endpoint
4. Add request validation and error handling
5. Implement SSE streaming response
6. Write API tests in `test/test_chatbot_api.py`

### Phase 4: Web UI
1. Create `python/llaisys/server/static/` directory
2. Implement `index.html` with chat interface structure
3. Implement `style.css` with modern, clean styling
4. Implement `app.js` with SSE handling and UI logic
5. Test manually in browser

### Phase 5: Integration & Testing
1. Run end-to-end tests
2. Fix bugs and edge cases
3. Performance testing and optimization
4. Documentation updates (README, usage guide)

## File Structure

```
llaisys/
├── src/
│   └── ops/
│       └── sample/
│           ├── cpu/
│           │   ├── sample_cpu.cpp
│           │   └── sample_cpu.hpp
│           ├── op.cpp
│           └── op.hpp
├── include/
│   └── llaisys/
│       └── models/
│           └── qwen2.h (modified)
├── python/
│   └── llaisys/
│       ├── server/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── routes.py
│       │   └── static/
│       │       ├── index.html
│       │       ├── style.css
│       │       └── app.js
│       ├── libllaisys/
│       │   └── models/
│       │       └── qwen2.py (modified)
│       └── models/
│           └── qwen2.py (modified)
└── test/
    ├── ops/
    │   └── sample.py
    ├── test_chatbot_api.py
    └── test_chatbot_e2e.py
```

## Dependencies

### Python Packages
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `sse-starlette` - Server-Sent Events support
- `transformers` - Tokenizer (already installed)
- `safetensors` - Model loading (already installed)

### Installation
```bash
pip install fastapi uvicorn sse-starlette
```

### C++ Libraries
- Standard library `<random>` for random number generation
- Existing LLAISYS tensor and operator infrastructure

## Usage

### Starting the Server
```bash
python -m llaisys.server.main --model /path/to/model --device cpu --port 8000
```

### Accessing the Chat UI
Open browser to: `http://localhost:8000`

### API Usage (curl example)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "max_tokens": 128,
    "stream": true
  }'
```

## Future Enhancements

### Short-term (Post-MVP)
- Add conversation history management (save/load)
- Implement stop button for generation
- Add copy button for assistant messages
- Support markdown rendering in messages
- Add dark mode toggle

### Medium-term
- Multi-user support with request queuing
- Session management (multiple conversations per user)
- Edit and regenerate functionality
- KV-Cache pool with prefix matching

### Long-term
- Continuous batching for multi-user inference
- Support for other models beyond Qwen2
- Plugin system for custom sampling strategies
- WebSocket support as alternative to SSE

## Success Criteria

The chatbot implementation will be considered successful when:

1. **Functional Requirements Met:**
   - Random sampling (Temperature, Top-K, Top-P) works correctly
   - FastAPI server runs stably and handles requests
   - Web UI provides smooth chat experience
   - Streaming responses work without lag or errors

2. **Quality Requirements Met:**
   - All unit tests pass
   - Integration tests pass
   - Manual testing checklist completed
   - Error handling works gracefully
   - Input validation prevents invalid requests

3. **Performance Requirements Met:**
   - Generation speed comparable to argmax baseline
   - Streaming latency < 100ms per token
   - Server handles requests without crashes

4. **User Experience:**
   - UI is intuitive and easy to use
   - Responses are coherent and relevant
   - Error messages are helpful
   - Settings panel works correctly

## Risks and Mitigations

### Risk 1: C++ Sampling Implementation Complexity
**Impact:** High - Core functionality
**Mitigation:** Start with simple implementation, compare with PyTorch, add comprehensive tests

### Risk 2: SSE Streaming Issues
**Impact:** Medium - User experience
**Mitigation:** Use proven library (sse-starlette), test with different browsers, implement reconnection logic

### Risk 3: Model Loading Time
**Impact:** Low - Startup delay
**Mitigation:** Load model at startup (not per-request), add health check endpoint, show loading state in UI

### Risk 4: Memory Usage with Long Conversations
**Impact:** Medium - Resource constraints
**Mitigation:** Monitor memory, implement conversation length limits, document memory requirements

## Conclusion

This design provides a solid foundation for building a production-ready AI chatbot with LLAISYS. The monolithic FastAPI architecture keeps implementation simple while maintaining extensibility for future enhancements. By implementing sampling in C++, we ensure optimal performance while following the project's existing patterns. The comprehensive testing strategy ensures reliability and correctness.

