# Rust Neural Network from Scratch - Progress Document

## Project Overview

Building a neural network library from scratch in Rust, with both CPU and GPU backends. The goal is to learn Rust, GPU programming (wgpu), and neural network fundamentals simultaneously.

## Current Status: Phase 2 - GPU Backend (In Progress)

### Completed Phases

#### Phase 1: Matrix Foundations (COMPLETE)

**File:** `src/matrix.rs`

Implemented:
- `Matrix` struct with row-major storage (`rows`, `cols`, `data: Vec<f64>`)
- Constructors: `zeros`, `ones`, `random`, `from_vec`
- Accessors: `get`, `set`
- `MatrixOperand` trait for generic operations (works with both `Matrix` and `f64`)
- Element-wise operations: `add`, `subtract`, `elem_multiply`, `elem_divide`
- Matrix multiplication: `matmul`
- `transpose`, `map`, `Display` trait
- 21 unit tests + 2 doc tests (23 total)
- Clean code with `#[must_use]` attributes

---

### Current Phase: GPU Backend with wgpu

**Goal:** Add GPU-accelerated matrix operations using WebGPU (wgpu crate)

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MatrixBackend trait                      │
│  - add(), subtract(), elem_multiply(), matmul(), map()      │
└─────────────────────────────────────────────────────────────┘
                    │                         │
                    ▼                         ▼
        ┌───────────────────┐     ┌───────────────────┐
        │    CpuBackend     │     │    GpuBackend     │
        │  Uses Matrix      │     │  Uses wgpu        │
        │  methods directly │     │  + WGSL shaders   │
        └───────────────────┘     └───────────────────┘
```

#### Dependencies Added to Cargo.toml

```toml
wgpu = "24.0"       # GPU abstraction (Vulkan/Metal/DX12)
pollster = "0.4"    # Minimal async runtime for blocking calls
bytemuck = "1.21"   # Safe byte casting for GPU buffers
```

---

## Step-by-Step GPU Implementation Guide

### Step 1: GpuBackend Struct (COMPLETE)

**File:** `src/gpu.rs`

```rust
use wgpu::{Device, Queue};

pub struct GpuBackend {
    pub device: Device,
    pub queue: Queue,
}
```

**Concepts:**
- `Device` - Logical connection to the GPU, used to create buffers/shaders
- `Queue` - Where commands are submitted for GPU execution

---

### Step 2: Create Instance (COMPLETE)

```rust
impl GpuBackend {
    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        // ...
    }
}
```

**Concepts:**
- `Instance` is the entry point to wgpu
- Discovers available backends (Vulkan on Linux, Metal on macOS, DX12 on Windows)
- `Backends::all()` tries all available backends

---

### Step 3: Request Adapter (CURRENT STEP)

**Status:** Not yet implemented

**Code to add after instance creation:**

```rust
let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })
    .await
    .expect("Failed to find a suitable GPU adapter");
```

**Concepts:**
- `Adapter` represents a physical GPU
- `HighPerformance` prefers discrete GPU over integrated
- `compatible_surface: None` because we do compute, not rendering
- Returns `Option<Adapter>`, hence the `.expect()`

---

### Step 4: Request Device and Queue (TODO)

**Code to add after adapter:**

```rust
let (device, queue) = adapter
    .request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Neural Network GPU"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )
    .await
    .expect("Failed to create GPU device");

Self { device, queue }
```

**Concepts:**
- `Device` is the logical GPU connection
- `Queue` is where we submit work
- `Features::empty()` - we don't need special GPU features
- `Limits::default()` - use sensible defaults for buffer sizes, etc.

---

### Step 5: Synchronous Wrapper (TODO)

```rust
pub fn new() -> Self {
    pollster::block_on(Self::new_async())
}
```

**Concepts:**
- GPU operations are async (they take time)
- `pollster::block_on()` runs async code and waits for completion
- Simpler API for users who don't need async

---

### Step 6: WGSL Compute Shader (TODO)

**Concept: What is a Shader?**

A shader is a program that runs on the GPU. For compute:
- CPU launches thousands of GPU threads
- Each thread runs the same shader code
- Each thread knows its unique ID (`global_invocation_id`)

**Shader for element-wise addition:**

```wgsl
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx < arrayLength(&a) {
        result[idx] = a[idx] + b[idx];
    }
}
```

**Concepts:**
- `@group(0) @binding(N)` - buffer bindings (like function parameters)
- `var<storage, read>` - read-only GPU buffer
- `var<storage, read_write>` - writable GPU buffer
- `@workgroup_size(64)` - 64 threads per workgroup
- `global_invocation_id` - unique thread ID across all workgroups
- `arrayLength(&a)` - bounds checking

**Important: f32 vs f64**
- GPUs are optimized for f32 (32-bit floats)
- Our Matrix uses f64, so we convert when sending to GPU
- Convert back to f64 when reading results

---

### Step 7: Implement GPU Add Method (TODO)

**Data flow:**

```
CPU                              GPU
 │                                │
 Matrix a (f64) ──convert──► Buffer a (f32)
 Matrix b (f64) ──convert──► Buffer b (f32)
                                  │
                            [Run Shader]
                                  │
 result (f64) ◄──convert─── Buffer result (f32)
```

**Buffer types needed:**
- `STORAGE` - shader can read/write
- `COPY_SRC` - can copy FROM this buffer
- `COPY_DST` - can copy TO this buffer
- `MAP_READ` - CPU can read this buffer

**Implementation outline:**

```rust
pub fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
    // 1. Convert f64 to f32
    let a_f32: Vec<f32> = a.data.iter().map(|&x| x as f32).collect();
    let b_f32: Vec<f32> = b.data.iter().map(|&x| x as f32).collect();

    // 2. Create GPU buffers
    let buffer_a = self.device.create_buffer_init(...);
    let buffer_b = self.device.create_buffer_init(...);
    let buffer_result = self.device.create_buffer(...);
    let staging_buffer = self.device.create_buffer(...);  // For reading back

    // 3. Compile shader and create pipeline
    let shader = self.device.create_shader_module(...);
    let pipeline = self.device.create_compute_pipeline(...);

    // 4. Create bind group (connect buffers to shader)
    let bind_group = self.device.create_bind_group(...);

    // 5. Encode and submit commands
    let mut encoder = self.device.create_command_encoder(...);
    {
        let mut pass = encoder.begin_compute_pass(...);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer_result, 0, &staging_buffer, 0, size);
    self.queue.submit(Some(encoder.finish()));

    // 6. Read results back
    let slice = staging_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    self.device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let result_f32: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

    // 7. Convert back to f64
    let result_f64: Vec<f64> = result_f32.iter().map(|&x| x as f64).collect();
    Matrix::from_vec(a.rows, a.cols, result_f64)
}
```

---

### Step 8: Backend Trait and Tests (TODO)

**File:** `src/backend.rs` (already written, just needs uncommenting in lib.rs)

The backend module provides:
- `MatrixBackend` trait with `add`, `subtract`, `elem_multiply`, `matmul`, `map`
- `CpuBackend` - uses Matrix methods directly
- `GpuBackend` - wraps gpu.rs, falls back to CPU for unimplemented ops

**To enable:** Uncomment `pub mod backend;` in `src/lib.rs`

---

## Current File States

### src/gpu.rs (In Progress)

```rust
//! GPU compute backend using wgpu.

use wgpu::{Adapter, Device, Queue};

pub struct GpuBackend {
    pub device: Device,
    pub queue: Queue,
}

impl GpuBackend {
    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // TODO: Step 3 - Request adapter
        // TODO: Step 4 - Request device and queue

        todo!()
    }

    // TODO: Step 5 - Add pub fn new() -> Self

    // TODO: Step 7 - Add pub fn add(&self, a: &Matrix, b: &Matrix) -> Matrix
}
```

### src/lib.rs

```rust
// pub mod backend;  // Temporarily disabled - depends on gpu
pub mod gpu;
pub mod matrix;
```

### src/backend.rs (Complete, waiting for gpu.rs)

Contains `MatrixBackend` trait with `CpuBackend` and `GpuBackend` implementations.
Note: References `GpuContext` which should be renamed to `GpuBackend` or the backend.rs updated.

---

## Future Phases

### Phase 3: Neural Network Building Blocks
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, Cross-Entropy)
- Forward propagation

### Phase 4: Backpropagation
- Gradient computation
- Chain rule implementation
- Weight updates

### Phase 5: Training Loop
- Batch processing
- Epochs
- Learning rate scheduling

### Phase 6: Optimizations
- More GPU shaders (subtract, multiply, matmul)
- Benchmark CPU vs GPU
- Memory optimizations

---

## Key Concepts Learned

### wgpu Architecture
```
Instance → Adapter → Device + Queue
   │          │           │
   │          │           └── Submit commands here
   │          └── Physical GPU
   └── Entry point, discovers backends
```

### GPU Memory Model
- GPU has separate memory (VRAM)
- Must explicitly copy data CPU ↔ GPU
- Buffer types control access patterns

### Compute Shaders
- Thousands of threads run same code
- Each thread has unique ID
- Workgroups: threads grouped for efficiency (typically 64 or 256)

### Async in Rust
- `async fn` - function that can pause/resume
- `.await` - pause here, let other work happen
- `pollster::block_on()` - run async code synchronously

---

## Commands Reference

```bash
# Build the project
cargo build

# Run all tests
cargo test

# Run specific test
cargo test test_gpu

# Check for errors without building
cargo check

# Format code
cargo fmt

# Run with release optimizations
cargo build --release
```

---

## Next Action

Continue from **Step 3**: Add the adapter request code to `src/gpu.rs`:

```rust
let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })
    .await
    .expect("Failed to find a suitable GPU adapter");
```

Then proceed to Steps 4-8 as documented above.
