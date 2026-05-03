# PyTorch Cheatsheet

Personal learning reference — built up section by section from *Deep Learning with PyTorch (2nd Edition)*.

---

## 1. The Core Module Map

The skeleton every PyTorch project hangs on.

| Module | What it provides | Stage of project |
|--------|------------------|------------------|
| `torch` | Tensors + math operations (the core) | Everywhere |
| `torch.nn` | Neural net layers, activations, loss functions | Building the model |
| `torch.utils.data` | `Dataset` & `DataLoader` (feeding data to the model) | Data pipeline |
| `torch.optim` | Optimizers — SGD, Adam, RMSProp, etc. | Training loop |
| `torch.autograd` | Automatic differentiation engine (gradients) | Training loop (under the hood) |
| `torch.distributed` | Multi-GPU / multi-machine training | Scaling up |
| `torch.compile` | JIT compile model for speed | Production / performance |

---

## 2. The 4-Step Training Loop → PyTorch Modules

Every training iteration is the same four steps. Each step has a home in PyTorch.

| # | Step | What happens | PyTorch module |
|---|------|--------------|----------------|
| 1 | Forward pass | Input → model → prediction | `torch.nn` |
| 2 | Loss | Compare prediction to target | `torch.nn` |
| 3 | Backward pass | Compute gradients (how much each weight contributed to the loss) | `torch.autograd` |
| 4 | Optimizer step | Update weights to reduce loss | `torch.optim` |

Feeding data into step 1: `torch.utils.data` (Dataset + DataLoader).

In code these surface as:

```python
pred = model(x)            # 1. forward        (torch.nn)
loss = loss_fn(pred, y)    # 2. loss           (torch.nn)
loss.backward()            # 3. backward       (torch.autograd)
optimizer.step()           # 4. update weights (torch.optim)
```

---

## 3. Key Concepts (from Ch. 1)

### Eager vs. Graph Mode

| Mode | Debug? | Speed? | Who uses it |
|------|--------|--------|-------------|
| Eager | Easy | Slower | PyTorch (default), TF 2.0+ |
| Graph | Hard | Faster | Old TF 1.x, compiled PyTorch |

In graph mode, variables are *descriptions* of computation, not *results*.

### `torch.compile`

Write code in **eager** mode, run it at the speed of **graph** mode. It traces eager code into a graph behind the scenes and fuses/optimizes operations.

### Autograd

- **Computes gradients.** That is its only job.
- It does NOT update weights — that is the optimizer's job.
- Works by recording every operation on a tensor into a history (computation graph), then walking backward through that history applying the chain rule.
- No history = no gradients = no learning.

### Python vs. C++/CUDA in PyTorch

- **Python layer** → usability, ecosystem integration.
- **C++/CUDA core** → speed, GPU parallelism.
- PyTorch is a thin Python API over a fast compiled core.

### Why `DataLoader` uses multiple worker processes

- GPU work is fast (ms). Disk I/O is slow (orders of magnitude slower).
- If done sequentially, the GPU sits idle waiting for data.
- Workers load the *next* batches in the background while the GPU trains the *current* one → **pipelining**, hides I/O latency.
- Uses processes (not threads) to bypass Python's GIL.

---

## 4. Ecosystem Quick Reference

| Tool | One-line pitch |
|------|----------------|
| PyTorch | Eager-mode DL framework, research default, industry standard |
| TensorFlow | Graph-first origin, now eager-by-default in 2.0+ |
| JAX | NumPy + GPU + autodiff + JIT — near-zero learning curve for NumPy users |
| Hugging Face | High-level wrapper: fast to ship, low on fine-grained control |
| ONNX | Vendor-neutral model exchange format |

---

## 5. Key Concepts (from Ch. 2)

### Anatomy of a Neural Network

| Part | What it is | Created by |
|------|-----------|------------|
| **Architecture** | Structural design — layers, connections (Python code) | Human designer (cheap) |
| **Weights** | Billions of numbers inside the layers | Training on data (expensive) |

A *pretrained model* = architecture + already-trained weights. The valuable part is the weights — they took massive GPU time to produce.

### Training vs. Inference

| Mode | What's happening | Gradients? | Weights change? |
|------|------------------|-----------|-----------------|
| **Training** | Model learns from data | Yes | Yes |
| **Inference** | Model makes predictions on new data | No | No |

### Neural Network as a Learned Function

Every DL task is a learned function from an input type to an output type.

| Task | Input | Output | Eval difficulty |
|------|-------|--------|-----------------|
| Image classification | Image | Label (discrete) | Easy — count right/wrong |
| Image-to-image (e.g. horse→zebra) | Image | Image | Hard — open-ended |
| Text completion | Text | Text | Hard — open-ended |
| Image captioning | Image | Text | Hard — open-ended |

**Rule:** Discrete outputs → easy to evaluate. Open-ended outputs → need human judgment or statistical proxies.

---

## 6. ImageNet Classifier (from Ch. 2)

### Dataset

- **Full ImageNet:** ~14M images, ~20k categories.
- **ILSVRC subset:** 1.2M training images across exactly **1000 classes**. This is what "pretrained on ImageNet" usually means.
- "Label" and "class" are used interchangeably.

### Data splits (universal ML practice)

| Split | Purpose |
|-------|---------|
| Train | Model learns from this |
| Validation | Tune hyperparameters, check generalization during dev |
| Test | Untouched until final evaluation |

### Pipeline

```
Image file → preprocessing → (3, 224, 224) tensor → model → (1000,) scores → argmax → class name
```

### Input / Output Shapes

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Input | `(3, 224, 224)` | 3 color channels (RGB) × height × width |
| Output | `(1000,)` | One score per class |

Predicted class = **index of the largest score** (argmax), mapped via a lookup table to the class name.

### Preprocessing Pipeline (standard for ImageNet models)

| Step | What | Why |
|------|------|-----|
| 1. Resize | Shortest side → 256 px | Uniform size |
| 2. Center crop | 224×224 | Model trained on 224×224 |
| 3. To tensor | 0–255 ints → 0.0–1.0 floats; shape `(3, H, W)` | Model expects float tensor |
| 4. Normalize | subtract mean `[0.485, 0.456, 0.406]`, divide by std `[0.229, 0.224, 0.225]` per channel | Match training distribution |

**Rule:** Input distribution at inference **must** match input distribution at training, or the model silently outputs garbage (no error raised).

### Top-1 vs Top-5 Accuracy

- **Top-1:** Model's #1 guess must be exactly right.
- **Top-5:** Correct class must appear in model's top 5 guesses.
- Top-5 is fairer when classes are visually similar (husky vs. malamute, 120 dog breeds, etc.).

---

## 7. Architectures So Far

### CNN vs Transformer

| | CNN (e.g. AlexNet) | Transformer (e.g. ViT) |
|---|---|---|
| Core op | Convolution | Self-attention |
| What it sees | Local neighborhoods | Global — any patch can attend to any other |
| Good at | Local pattern matching (edges, textures) | Long-range relationships |
| Data hunger | Modest | Massive |

### Universal Classifier Skeleton

Every image classifier has the same 3-stage skeleton, regardless of architecture:

```
raw image → feature representation → class scores
```

| Role | AlexNet | ViT |
|------|---------|-----|
| Turn image into features | `features` (conv stack) | `conv_proj` (patch embedding) |
| Process / refine | (implicit + `avgpool`) | `encoder` (attention layers) |
| Classify | `classifier` (FC layers) | `heads` (Linear) |

### AlexNet Shape Journey

```
(3, 224, 224) → conv stack → (256, 6, 6) → flatten → (9216,) → FC → (4096,) → FC → (1000,)
```

- Spatial dims **shrink** as you go deeper.
- Channel dim **grows** — each channel is a learned "feature map."
- Early layers learn edges/textures; middle layers learn parts; late layers learn whole objects.

### ViT Naming Decomposition

```
vit_b_16
    │  │
    │  └── patch size (pixels)
    └───── size class (b=Base, l=Large, h=Huge)
```

- A 224×224 image with patch size 16 → `224/16 = 14` × `14` = **196 patches**.
- Each patch → 768-dim embedding → attention across all 196 patches.

### Drivers of DL Progress (2012→today, 15% → 3% top-5 error)

All four are needed simultaneously:
- **Architecture** (AlexNet → ResNet → Transformers)
- **Data** (more, cleaner, better augmentation)
- **Training tricks** (Adam, BatchNorm, Dropout, schedulers)
- **Compute** (bigger GPUs, distributed training)

---

## 8. Modules and Preprocessing (from Ch. 2.3)

### What's a "Module" in PyTorch?

**Not a Python module.** A PyTorch **module** is a **named unit of computation in the network tree**.

- Primitive modules: `Linear`, `Conv2d`, `ReLU`, `LayerNorm`, `Dropout`, `MultiheadAttention`
- Container modules: `Sequential`, `Encoder`, `EncoderBlock`
- Every module: has a `forward` method, may hold learnable parameters, can contain sub-modules.

Modules **nest** — a model is a tree of modules. The indented printout of a model shows this tree.

### `transforms.Compose([...])`

Takes a list of transforms and returns a **single callable** that applies them in order. Pure function composition:

```python
preprocess = transforms.Compose([t1, t2, t3])
preprocess(img)   # ≡ t3(t2(t1(img)))
```

Does nothing to the image — just bundles steps.

### Standard ImageNet Preprocessing Pipeline

```python
preprocess = transforms.Compose([
    transforms.Resize(256),                    # shortest side → 256 px (ensures min size)
    transforms.CenterCrop(224),                # 224×224 square from center (model's input size)
    transforms.ToTensor(),                     # PIL→Tensor, 0-255→0-1, HWC→CHW
    transforms.Normalize(                       # match training distribution
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

**Resize before Crop** — otherwise crop might fail on images smaller than 224 in one dimension.

### `ToTensor()` — the three silent jobs

| # | Job | Before → After |
|---|-----|----------------|
| 1 | Type | PIL Image → `torch.Tensor` |
| 2 | Scale | int 0–255 → float 0.0–1.0 |
| 3 | Axes | `(H, W, C)` → `(C, H, W)` |

`ToTensor()` does **not** resize. Resizing is `Resize`'s job.

### Normalize mean/std — where the numbers come from

Not magic — **measured per-channel mean and std of the ImageNet training set** (~1.2M images, scaled to 0–1).

```
normalized = (pixel - mean) / std
```

After normalization, each channel has mean ≈ 0, std ≈ 1 — matching the distribution the model was trained on.

### The Batch Dimension Rule

PyTorch models **always expect** a 4D input: `(N, C, H, W)`. Even for a single image.

```python
img_t = preprocess(img)              # shape (3, 224, 224)
batch_t = torch.unsqueeze(img_t, 0)  # shape (1, 3, 224, 224)  ← ready for the model
```

`torch.unsqueeze(tensor, dim)` inserts a size-1 dimension at position `dim`. Axis 0 = batch dim.

**Why:** GPU parallelizes across the batch dim. Training uses mini-batches. The model's math assumes the first axis is "how many examples." This is consistent between training and inference.

### Universal Inference-Prep Mental Model

```
File on disk
  ↓ PIL.Image.open(path)
PIL Image (H, W, C) int 0-255
  ↓ preprocess(img)
Tensor (C, H, W) float, normalized
  ↓ torch.unsqueeze(img_t, 0)
Tensor (1, C, H, W) — ready for model(batch_t)
```

---

## 9. Running Inference and Decoding Outputs (from Ch. 2.3 "Run!")

### The Full Inference Pipeline

```
Image on disk
  ↓ PIL.Image.open(path)
PIL Image
  ↓ preprocess(img)              # Compose pipeline
Tensor (C, H, W)
  ↓ torch.unsqueeze(img_t, 0)
Tensor (1, C, H, W)               # batch-ready
  ↓ model.eval()                  # turn off dropout/BN train behavior
  ↓ out = model(batch_t)          # forward pass
Tensor (1, 1000)                  # logits
  ↓ _, index = torch.max(out, 1)  # winner index (as tensor)
  ↓ index.item()                  # Python int
  ↓ labels[index.item()]          # class name
"golden retriever"

And for confidence:
  ↓ torch.nn.functional.softmax(out, dim=1)[0] * 100
percentage (1000,) in 0-100 range → percentage[index.item()].item()
```

### `eval()` vs `train()` mode

| Mode | Dropout | BatchNorm | Call before |
|------|---------|-----------|-------------|
| `model.eval()` | All neurons active | Uses running stats | Every inference |
| `model.train()` | Random dropout active | Uses batch stats | Every training step |

Forgetting = silent bug. Predictions become non-deterministic; running inference twice gives different answers.

### `torch.max(tensor, dim)`

Returns a tuple `(values, indices)`, both reduced along `dim`.

```python
_, index = torch.max(out, 1)   # index = tensor of class IDs per image
```

The `_` is Python convention for "discard this." We only want the winning class index.

### `.item()`

Converts a **single-element tensor** to a Python scalar.

```python
torch.tensor([207]).item()  # → 207 (int)
```

Needed because Python lists want int indices, not tensors.

### Logits vs Probabilities

- **Logits** = raw model scores. Any real number. Order is meaningful; magnitude isn't directly interpretable.
- **Softmax** applied along the class dim → probabilities in [0, 1] summing to 1.
- **argmax** gives the same winner on either (softmax preserves order).
- Use softmax only when you need **confidence values**, not when you just need the winner.

### Top-K with `torch.sort`

```python
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
```

Top-k diagnoses **what else the model considered**. Useful for spotting spurious correlations (e.g. "tennis ball" in top-5 for a dog photo → dog-photos-contain-tennis-balls bias).

---

## 10. Tensor Dimensions — The Core Mental Model

### Reading Shapes

**The shape tuple reads left to right, and the position of each number is its axis index.**

| Shape | axis 0 | axis 1 | axis 2 | axis 3 |
|-------|--------|--------|--------|--------|
| `(4,)` | 4 | | | |
| `(2, 3)` | 2 rows | 3 cols | | |
| `(3, 224, 224)` | 3 channels | 224 H | 224 W | |
| `(1, 3, 224, 224)` | 1 batch | 3 channels | 224 H | 224 W |
| `(1, 1000)` | 1 batch | 1000 classes | | |

### `dim=N` — the "disappearing axis" rule

> **The axis you name in `dim=` is the one that gets reduced away.** What's left is everything else.

| Input shape | `dim=0` | `dim=1` | `dim=2` |
|-------------|---------|---------|---------|
| `(1, 1000)` | `(1000,)` | `(1,)` | error |
| `(32, 768)` | `(768,)` | `(32,)` | error |
| `(3, 4, 5)` | `(4, 5)` | `(3, 5)` | `(3, 4)` |

Applies to `torch.max`, `torch.sum`, `torch.mean`, `torch.argmax`, `softmax` (with `dim=`), etc.

### Indexing vs Slicing — what happens to axes

| Operation on `x` of shape `(1, 1000)` | Result shape | Axis 0 behavior |
|---------------------------------------|--------------|-----------------|
| `x` | `(1, 1000)` | unchanged |
| `x[0]` | `(1000,)` | **removed** (int indexing) |
| `x[0:1]` | `(1, 1000)` | kept (slicing) |

**Rule:** Integer indexing **removes** the axis. Slicing **keeps** it (with size 1).

### Batch Dimension Conventions

- PyTorch models **always** expect a batch dim at axis 0.
- Images: `(N, C, H, W)` — batch, channel, height, width.
- Sequences / text: `(N, L, D)` or `(L, N, D)` depending on the module (check docs!).
- To add a batch dim for a single input: `torch.unsqueeze(x, 0)`.

---

## 11. Insights & Gotchas

### Networks learn correlations, not concepts

Neural networks don't "understand" the world. They find statistical correlations between pixels and labels in the training set.

- A dog classifier can learn to weight "tennis ball present" as evidence for "dog" — because dog training photos often contain tennis balls.
- Surveying top-k predictions is the cheapest diagnostic for spurious correlations.
- This is why data quality and augmentation matter as much as architecture.

### Predict → Run → Verify

Always predict the output shape of a tensor operation before running it. Then run and compare. Mismatches = bug. Matches = you understood.

### Training vs Inference Checklist (set these before forward pass)

- [ ] Model in correct mode: `model.eval()` for inference, `model.train()` for training
- [ ] Input has batch dim: shape starts with `N`
- [ ] Input is normalized using the same stats the model was trained on
- [ ] Input tensor is on the same device as the model (`.cuda()` / `.to(device)` — covered later)

---

## 12. Diffusion Models and Inpainting (from Ch. 2.2.1)

### Core Idea: Learn to Reverse Noise

Diffusion models learn to **undo** a gradual noising process.

- **Training:** take any clean image → add Gaussian noise → train model to predict the noise (or the clean version). Repeat at many noise levels.
- **Key win:** training pairs are generated **automatically** from any dataset of clean images. No labeling needed.
- **Inference:** start from pure noise (or a noised input image) → iteratively apply the learned denoiser, guided by a prompt → coherent image emerges.

### Gaussian Noise

Random values drawn from a bell curve (normal distribution), mean 0.

- **σ (std dev)** controls intensity: small σ = grain, large σ = TV static.
- Mathematically convenient — sums of Gaussians are Gaussian, enabling closed-form noise schedules.
- Natural — matches real-world sensor noise.

### The Three Generation Modes

Same diffusion model, different inputs and starting points:

| Mode | Inputs | Starts from | What happens |
|------|--------|-------------|--------------|
| **Text-to-image** | prompt only | pure noise | Brand-new image that matches prompt |
| **Image-to-image** | prompt + input image | (lightly noised) input | Whole image re-rendered, composition preserved |
| **Inpainting** | prompt + input image + mask | (noised) input | Only **masked** region changes; rest preserved pixel-perfect |

### Mask Conventions (Binary)

- **Black pixels = preserved.** Copied back from the original at every denoising step.
- **White pixels = editable.** The model is free to repaint.
- Binary gives a **clean yes/no** decision per pixel — no blending ambiguity.

### Inpainting's Three Inputs

| Input | Role |
|-------|------|
| **Prompt** (text) | *What* we want |
| **Input image** | *Where* we begin |
| **Mask** (binary image) | *Where* edits may happen |

### Separation of Concerns: Model vs. Loop

**The model is creative. The inference loop is disciplined.**

- The diffusion model itself doesn't "understand" masks. It just denoises.
- The inference loop **enforces** the mask by overwriting unmasked pixels back to the original at every step.
- Without this re-insertion, the stochastic model would drift outside the mask.

**Rule:** the model doesn't obey constraints — the loop does.

---

## 13. Running Stable Diffusion Inpainting (from Ch. 2.2.2)

### The `diffusers` Library

Hugging Face's library for diffusion models. Sits on top of PyTorch. Provides pre-assembled **pipelines** that bundle all components needed for generation.

```python
from diffusers import StableDiffusionInpaintPipeline
```

### Pipeline Components (What's Inside)

A `StableDiffusionInpaintPipeline` bundles **five core components**:

| Component | What it does |
|-----------|--------------|
| `tokenizer` | Text prompt → token IDs |
| `text_encoder` (CLIP) | Token IDs → embedding vectors |
| `unet` | The denoiser — the core diffusion network |
| `vae` | Image ↔ latent space (compression & reconstruction) |
| `scheduler` | Controls noise removal schedule per step |

Plus housekeeping parts: `feature_extractor`, optional `safety_checker`.

### Loading a Pretrained Pipeline

```python
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",   # Hugging Face repo ID
    torch_dtype=torch.float32                         # float32 on MPS, float16 on CUDA
).to(device)
```

**`from_pretrained("...")` vs `.to(device)`:**

| Step | What it does |
|------|--------------|
| `from_pretrained(repo_id)` | **Downloads weights** from Hugging Face Hub → loads into CPU RAM |
| `.to(device)` | **Transfers** weights from CPU RAM → GPU/MPS memory. In-memory, not disk. |

**Rule:** Model and its inputs must live on the **same device**. Mismatch = error.

### Device Selection (Cross-platform)

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

| Hardware | Device | Notes |
|----------|--------|-------|
| NVIDIA GPU | `"cuda"` | Best performance, `float16` works fine |
| Apple Silicon (M1/M2/M3/M4) | `"mps"` | Metal backend; use `float32` — MPS has issues with some fp16 ops |
| Intel Mac / no GPU | `"cpu"` | Very slow for diffusion |

### Inpainting Call Signature

```python
out = pipe(
    prompt=prompt,
    image=img,
    mask_image=mask_img,
    negative_prompt=negative,
    guidance_scale=7.5,
    strength=0.8,
    generator=torch.Generator(device).manual_seed(42),
)
result_image = out.images[0]
```

| Argument | Controls | Typical |
|----------|----------|---------|
| `prompt` | What you want | "a zebra, same pose, background unchanged" |
| `image` | Starting image (PIL) | `img` |
| `mask_image` | Binary mask (PIL) | `mask_img` |
| `negative_prompt` | What to **avoid** (repels the model) | "blurry, watermark, distorted" |
| `guidance_scale` | How literally to follow prompt. Higher = stricter, lower = more creative. **7.5 = standard**. | 7.5 |
| `strength` | How much to change in masked region. **0 = no change, 1 = ignore input**. 0.8 = mostly regenerate. | 0.8 |
| `generator` | Seeded RNG → reproducibility. Same seed → same output. | `manual_seed(42)` |

### Prompt-Writing Rules

- Describe both **what to add** AND **what to preserve**. "A zebra" is weak. "A zebra in the same pose, same lighting, background unchanged" is strong.
- `negative_prompt` is a **separate repelling signal** — don't just say "not blurry" in the positive prompt; put it in the negative.

### Reproducibility

Diffusion is stochastic. Without a seed, every run gives a different output. With `torch.Generator(device).manual_seed(N)`, results are deterministic — essential for comparing prompt variants without confounding randomness.

### The Big Insight

> **The network is a scaffold — the juice is in the weights.**

No architecture code is "zebra-specific" or "horse-specific." All domain knowledge lives in the learned parameters. Same architecture + different weights = entirely different model behavior.

---

## 14. Tensor Internals — Storage, Offset, Stride (from Ch. 3.9.1)

### The Core Mental Model

A `Tensor` is **not** a self-contained block of numbers. It's a thin metadata wrapper over a flat 1D `Storage`.

```
Tensor = (storage, storage_offset, size, stride)
```

| Field | What it is |
|-------|-----------|
| `storage` | The actual flat 1D buffer of numbers in RAM |
| `storage_offset` | Storage index where this tensor's first element lives |
| `size` | Tuple of per-dimension lengths (a.k.a. `shape`) |
| `stride` | Tuple of per-dimension **jump sizes in storage** |

A 2D (or N-D) tensor *looks* multi-dimensional to the user, but in memory it's still a flat 1D array. The metadata defines how to **navigate** that flat array as if it were N-D.

```
Tensor (view)              Storage (flat 1D buffer)
┌───────────────┐          ┌──────────────────────┐
│ storage_offset│ ────────►│ [4.0, 1.0, 5.0, 3.0, │
│ size          │          │  2.0, 1.0]           │
│ stride        │          └──────────────────────┘
└───────────────┘
```

### The Indexing Formula (the most important equation in PyTorch)

```
storage_index = storage_offset + Σ stride[d] * index[d]
                                  d
```

Every n-dimensional index op reduces to this single line. Burn it in.

**Example.** `points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])`

```
storage:  [ 4.0 , 1.0 , 5.0 , 3.0 , 2.0 , 1.0 ]
index:      0     1     2     3     4     5
stride:    (2, 1)   storage_offset: 0   size: (3, 2)

points[2, 1] → storage_offset + stride[0]*2 + stride[1]*1
             = 0 + 2*2 + 1*1
             = 5  →  storage[5] = 1.0  ✓
```

### What Stride Actually Means

> `stride[d]` = **how many storage positions to skip when moving one step along dimension d.**

Stride is **not** a coordinate. It's a jump size.

```
points (3 × 2), stride (2, 1):

  step a row    → jump 2 in storage   (each row = 2 elements)
  step a column → jump 1 in storage   (columns are adjacent)

storage:  [ 4.0 , 1.0 , 5.0 , 3.0 , 2.0 , 1.0 ]
            ▲           ▲           ▲
            row 0       row 1       row 2
            (jump 2 to advance one row)

            ▲     ▲
            col 0 col 1
            (jump 1 to advance one column)
```

### Slicing = New Metadata, Same Storage (O(1))

When you slice with **integer indexing**, that dimension is **locked** — its stride/size are dropped, and `storage_offset` advances by `stride[d] * k`.

```
new_storage_offset = old_storage_offset + stride[d] * (the integer you picked)
new_size           = old_size with axis d removed
new_stride         = old_stride with axis d removed
```

**Example.** `second_point = points[1]`

```
                  storage_offset    size      stride
points    (2D):         0          (3, 2)     (2, 1)
points[1] (1D):         2          (2,)       (1,)
                        │                       │
       row 1 starts at  ┘   col-stride survives ┘
       storage[2]            (rows can't move anymore)
```

No data is copied. Slicing is `O(1)` regardless of tensor size — only three small numbers change.

### Slice with `:` (keep dim) vs integer (kill dim)

```python
points[1]      # int       → axis 0 dies         → 1D, size (2,)
points[1:2]   # slice     → axis 0 kept (size 1) → 2D, size (1, 2)
points[:, 1]  # all rows, col 1 → axis 1 dies   → 1D, size (3,)
```

For `points[:, 1]` (all rows, column 1):
- Locked dim = column (`stride[1]=1, k=1`) → `storage_offset = 0 + 1*1 = 1`
- Surviving dim = rows → `stride = (2,)`, `size = (3,)`

```
storage:  [ 4.0 , 1.0 , 5.0 , 3.0 , 2.0 , 1.0 ]
                ▲           ▲           ▲
                start       jump 2      jump 2
                (offset=1)  →           →
              last_col[0] last_col[1] last_col[2]
                = 1.0       = 3.0       = 1.0
```

### Views Share Storage — Mutations Propagate

Slices are **views**. Writing through a view writes to the shared storage. The parent tensor sees it.

```python
second_point = points[1]
second_point[0] = 10.0      # writes through to shared storage
points
# tensor([[ 4.,  1.],
#         [10.,  3.],   ← changed
#         [ 2.,  1.]])
```

Two tensors, one buffer:
```
points ─┐
        ├──► Storage S   (any write through either side mutates both)
sliced ─┘
```

### `.clone()` Breaks the Link

`.clone()` allocates a **brand-new storage**, copies the data in, and returns a tensor pointing at the new buffer. The two tensors are now fully independent.

```
Before clone:                 After .clone():

points ─┐                     points ──► Storage A
        ├──► Storage S
sliced ─┘                     cloned ──► Storage B   (independent)
```

```python
second_point = points[1].clone()
second_point[0] = 10.0
points          # unchanged
```

### When to Reach for `.clone()`

- You need an **independent copy** for in-place modification.
- You're returning a slice from a function and don't want the caller mutating the parent.
- You see an `inplace operation modified` autograd error and you actually need a fresh buffer.

### Two Worlds — Don't Mix Them Up

```
TENSOR WORLD                         STORAGE WORLD
multi-dim, what you write            flat 1D buffer, what RAM holds
─────────────────────────            ──────────────────────────
points[2, 1]   (tensor index)        storage[5]   (storage index)
second_point[1]                      storage[3]
```

A **tensor index** is a tuple of per-axis coords. A **storage index** is a single integer 0..N-1. The formula above is the bridge between them.

---

## 15. Transpose & Contiguity (from Ch. 3.9.2)

### Transpose = Stride Swap (no copy)

`tensor.t()` (2D shorthand for `transpose(0, 1)`) returns a new view that **shares storage** with the original. Only `size` and `stride` are swapped; the buffer is untouched.

```
                  before .t()              after .t()
                  ───────────              ──────────
   storage        SHARED ◄──────────────── SAME (no copy)
   storage_offset 0                        0
   size           (3, 2)                   (2, 3)        swapped
   stride         (2, 1)                   (1, 2)        swapped
```

Generalizes: `transpose(d1, d2)` for any tensor swaps `size[d1]↔size[d2]` and `stride[d1]↔stride[d2]`. `.t()` is the 2D shortcut.

### Why Stride-Swap = Transpose

`stride[d]` is the jump-per-step along dim `d`. After swapping strides, "moving along axis 0" now jumps the way columns used to, and "moving along axis 1" jumps the way rows used to. That **is** a transpose.

```
        same storage:  [ 4.0 , 1.0 , 5.0 , 3.0 , 2.0 , 1.0 ]
                         ┬────┬────┬────┬────┬────┬────

points (3×2) stride (2,1):     points_t (2×3) stride (1,2):
walk row-major                 walk column-major over the original
row0: 0,1                      row0: 0,2,4   ← [4, 5, 2]
row1: 2,3                      row1: 1,3,5   ← [1, 3, 1]
row2: 4,5
```

### Double Transpose Returns the Original Metadata

```
points         shape (3, 2)   stride (2, 1)
points.t()     shape (2, 3)   stride (1, 2)    ← swap
points.t().t() shape (3, 2)   stride (2, 1)    ← swap again → matches points exactly
```

Two swaps cancel — same offset, size, stride, storage as the original.

### Contiguous — The Real Rule

A tensor is **contiguous** when its elements are packed in storage in natural reading order (last axis varies fastest, no gaps).

> **Rule (general):** `stride[d] = size[d+1] * size[d+2] * ... * size[N-1]`, with `stride[N-1] = 1`.
>
> **Rule (2D shape `(R, C)`):** contiguous stride = `(C, 1)`. **Row-stride equals the number of columns**, not "column-stride + 1."

```
shape (3, 2) contiguous → stride (2, 1)   ← jump 2 to skip a row of 2 cols
shape (3, 5) contiguous → stride (5, 1)   ← jump 5 to skip a row of 5 cols
shape (R, C) contiguous → stride (C, 1)
```

### Common False Heuristic (Don't Use)

> ✗ "Row-stride is 1 more than column-stride."

Coincidence on `(3, 2)`. Fails on `(3, 5)` (`5-1=4`), `(3, 100)` (`100-1=99`), etc. Stick to **row-stride = #columns**.

### Why Transpose Is Non-Contiguous

```
points   (3, 2) stride (2, 1)   contiguous? need (2, 1).  ✓ YES
points_t (2, 3) stride (1, 2)   contiguous? need (3, 1).  ✗ NO  (row-stride 1 ≠ #cols 3)
```

Transposing a contiguous tensor **always** produces a non-contiguous view (unless one of the axes has size 1).

### The Hidden Cost of Free Transpose

Free transpose isn't truly free — the price is layout. Many ops require contiguous memory:

```python
points_t.view(6)              # ❌ RuntimeError: view size is not compatible
                              #    with input tensor's size and stride
points_t.contiguous().view(6) # ✓  .contiguous() copies into a packed buffer
```

| Op | Requires contiguous? |
|----|----------------------|
| `.view()` | **Yes** |
| `.reshape()` | No (will copy if needed) |
| Most arithmetic (`+`, `*`, etc.) | No |
| Many low-level / fused kernels | Often yes |

`.contiguous()` is the escape hatch — it allocates a fresh row-packed Storage and copies elements into reading order. After it, `stride` matches the contiguous formula.

### Mental Model: Two Ways to Walk One Buffer

The transpose isn't a new array. It's a **different walking rule** over the same array. Storage is the road; stride tells you which direction you're driving.

---

## 16. Higher-Dim Transpose & Contiguous Stride Formula (from Ch. 3.9.3)

### `transpose(d1, d2)` for any rank

`tensor.transpose(d1, d2)` swaps axes `d1` and `d2`, leaving all other axes alone. It's a **position-wise swap** on `size` and `stride`:

```
size[d1]  ↔ size[d2]
stride[d1] ↔ stride[d2]
all other entries: unchanged
```

`tensor.t()` is the 2D shorthand for `transpose(0, 1)`. For higher ranks, use `transpose(d1, d2)` or `permute(...)` (general re-ordering).

### Contiguous Stride Formula — General Form

> **`stride[d]` = product of all sizes to the right of `d`.**
>
> `stride[N-1] = 1`, then build leftward by multiplying.

In plain English: to skip one step along axis `d`, you must jump over **everything that lies inside** that axis (all the dims to the right of it).

```
                   d            d+1   d+2   ...   N-1
stride[d] =   1   ×   size[d+1] × size[d+2] × ... × size[N-1]
```

### Worked Examples

```
shape (3, 2)        → stride (2, 1)
                     stride[1] = 1
                     stride[0] = size[1] = 2

shape (3, 4, 5)     → stride (20, 5, 1)
                     stride[2] = 1
                     stride[1] = size[2] = 5
                     stride[0] = size[1]*size[2] = 4*5 = 20

shape (2, 3, 4, 5)  → stride (60, 20, 5, 1)
                     stride[3] = 1
                     stride[2] = size[3] = 5
                     stride[1] = size[2]*size[3] = 4*5 = 20
                     stride[0] = size[1]*size[2]*size[3] = 3*4*5 = 60
```

### The Slab Picture (3D)

A `(3, 4, 5)` tensor = **3 slabs**, each slab = **4 rows × 5 columns**, all packed in storage:

```
[ ─── slab 0 (20 cells) ─── ][ ─── slab 1 (20 cells) ─── ][ ─── slab 2 (20 cells) ─── ]

within each slab: 4 rows of 5 cells each, packed
within each row:  5 cells, adjacent

  step 1 along axis 2 (column): jump 1   →  stride[2] = 1
  step 1 along axis 1 (row):    jump 5   →  stride[1] = 5  (= one row's width)
  step 1 along axis 0 (slab):   jump 20  →  stride[0] = 20 (= one slab's size)
```

### Higher-Dim Transpose Example

```
some_t      = torch.ones(3, 4, 5)
              shape = (3, 4, 5),    stride = (20, 5, 1)

some_t.transpose(0, 2):
              swap size[0]↔size[2]:    shape  = (5, 4, 3)
              swap stride[0]↔stride[2]: stride = ( 1, 5, 20)

  Note: axis 1 (size 4, stride 5) is unchanged — it wasn't named in the swap.
```

### Contiguity Check (any rank)

Compute the contiguous stride for the *current* shape using the formula, then compare to the actual stride:

```
some_t.transpose(0, 2):
  current shape         = (5, 4, 3)
  contiguous would be   = (12, 3, 1)    ← (4*3, 3, 1)
  actual                = ( 1, 5, 20)
  mismatch  →  NON-CONTIGUOUS
```

### Definition of Contiguous (book's wording)

> A tensor is **contiguous** if its values are laid out in storage starting from the **rightmost dimension** and moving outward.

Practically: last axis varies fastest, fully packed, no gaps. CPUs and GPUs read sequential memory faster than scattered memory — that's the **performance** benefit of contiguity (data locality).

### Mental Shortcut

Transposing a freshly contiguous tensor **always** yields non-contiguous output (unless one of the swapped axes has size 1). Two transposes that cancel (e.g. `t().t()`) restore the original metadata exactly.

---

## 17. Saving Tensors to Disk (from Ch. 3.13.1 — HDF5 with h5py)

### Two Persistence Options in PyTorch

| Option | Format | When to use |
|---|---|---|
| `torch.save(t, path)` / `torch.load(path)` | PyTorch-native pickle | PyTorch-only workflow. Fast round-trip. |
| **HDF5 via `h5py`** | Open scientific format | (a) Other tools/languages need to read it. (b) **Dataset is bigger than RAM.** |

### The Killer Feature: Lazy Disk Access

HDF5 lets you **open a file without loading the data**. You only read what you ask for, when you ask for it. Critical for files larger than your RAM.

```
file on disk                   handle in Python                 result in RAM
────────────                   ────────────────                 ─────────────
big array       ◄─ open ─►     dset = f['coords']              0 bytes
of numbers                     (knows shape, type,
                                where data lives)

                ◄─ slice ─►    dset[-2:]                        only those rows
                               (h5py seeks + reads
                                just those bytes)
```

### The Write Flow

```python
import h5py
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

f = h5py.File('ourpoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())   # 'coords' is the key
f.close()
```

- `points.numpy()` is **zero-copy** — same buffer, new metadata wrapper.
- HDF5 stores the bytes under the key `'coords'` (you can have many keys, even nested).

### The Read Flow

```python
f = h5py.File('ourpoints.hdf5', 'r')   # 1. Open. Nothing read yet.
dset = f['coords']                      # 2. Get handle to the 'coords' array.
                                        #    Still nothing in RAM.
last_points = torch.from_numpy(dset[-2:])  # 3. Read just last 2 rows from disk → RAM,
                                           #    then COPY into PyTorch storage.
f.close()                               # 4. Close. dset becomes invalid.
```

After step 4: `last_points` survives (it has its own copy). `dset[0]` raises an exception (its file is closed).

### `.numpy()` ↔ `from_numpy()` — Zero-Copy Bridge

A `numpy.ndarray` is also metadata over a flat 1D buffer — same model as a PyTorch tensor.

```
PyTorch tensor                NumPy array
   │                              │
   │   metadata                   │   metadata
   │                              │
   └──── same flat buffer ────────┘
         [4.0, 1.0, 5.0, 3.0, ...]
```

- **`tensor.numpy()`** — zero-copy, shares buffer. Mutations propagate.
- **`torch.from_numpy(arr)`** — usually zero-copy, shares buffer. Mutations propagate.

### When `from_numpy(...)` Actually COPIES (the HDF5 case)

The book says `torch.from_numpy(dset[-2:])` *does* copy. Two reasons:

1. **Disk-to-RAM crossing.** `dset[-2:]` is the moment data is first read from disk into a fresh NumPy buffer. That step itself is a copy — disk bytes → RAM bytes. Unavoidable.
2. **Buffer lifetime is tied to the file.** The buffer h5py returns belongs to h5py and dies when `f.close()` is called. If PyTorch shared it (zero-copy), your tensor would point at freed memory after close. So PyTorch copies into its own storage so the tensor outlives the file.

```
disk file                    h5py-owned buffer            tensor storage
─────────       read         (dies on f.close)            (independent)
big array  ──────────────►   [5.0, 3.0, 2.0, 1.0]  ──►   [5.0, 3.0, 2.0, 1.0]
                              (temporary)                  (survives close)
```

### Handle vs. Copy — The Mental Distinction

The whole section turns on this distinction:

| | What it is | Survives `f.close()`? |
|---|---|---|
| `dset` (h5py dataset) | A **handle** that re-reads the file on every access | **No** — dies with the file |
| `last_points` (PyTorch tensor) | A **copy** of the data in RAM | **Yes** — independent |

**Rule:** Anything you want to use after closing the file must be a copy in RAM, not a handle.

---

## 18. Precision Rules (Lessons from Mistakes)

- "Faster" is vague — faster to *compile* or faster to *run*? Be specific.
- Autodiff ≠ training. Autodiff is one of four steps.
- Batching ≠ parallel loading. `DataLoader` workers solve I/O latency, not memory.
- Level of abstraction (raw PyTorch vs. Hugging Face) should match the goal — shipping a product vs. building understanding.
- A "pretrained model" is architecture **+** weights, not a third thing. The two **parts** of a network are architecture and weights.
- Skipping normalization doesn't "break training" (we're not training during inference) — it causes activations to explode and produces silent garbage.
- Neural networks output **numbers**, not labels. The label comes from argmax + a class lookup.
- PyTorch "module" ≠ Python module. It's a node in the network tree.
- `transforms.Compose` doesn't modify the image — it **chains** transforms into one callable.
- `ToTensor()` doesn't resize; it only reorders axes (HWC→CHW), converts type (PIL→Tensor), and rescales (0-255→0-1).
- Normalize stats are **measured**, not random. Per-channel mean/std of ImageNet training set.
- **Always** unsqueeze before inference. Even one image needs a batch dim. Shape `(1, C, H, W)` is the contract.
- **Predict → run → verify** — build this habit for every tensor op.
- `torch.max(t, dim=N)` returns `(values, indices)` — a tuple. Use `_` to discard what you don't need.
- `.item()` only works on **single-element** tensors. It unwraps to a Python scalar.
- **Axis you name in `dim=` is the axis that dies.** Everything else stays.
- `x[0]` removes axis 0. `x[0:1]` keeps it (size 1). Mix these up and shapes go wrong silently.
- Skipping `model.eval()` → non-deterministic predictions. Silent bug.
- Softmax is only needed when you want **confidence**. Argmax doesn't need it.
- Diffusion training data is "free" — any image + random Gaussian noise = a training pair.
- Inpainting mask is **binary** for a hard yes/no decision per pixel; no blending ambiguity.
- The model doesn't respect masks — the **inference loop** enforces them via re-insertion.
- Three generation modes: text-to-image, image-to-image, inpainting. Keep them distinct.
- Pipeline **components** ≠ pipeline call **arguments**. Components are the model internals (tokenizer, unet, vae...); arguments are the values you pass at inference (prompt, image, mask_image...).
- `from_pretrained("...")` **downloads** weights to CPU RAM. `.to(device)` **transfers** them to GPU/MPS memory. Two separate steps.
- On Apple Silicon MPS, prefer `torch.float32` over `torch.float16` — some fp16 ops are buggy on MPS.
- Model and inputs must be on the **same device** — forgetting this is the most common PyTorch runtime error.
- `strength=0` preserves the original; `strength=1` ignores it. Middle values blend.
- Seed the generator when iterating on prompts — otherwise, you can't tell if a change came from your prompt or from random variation.
- A Tensor is metadata over a Storage, not a block of numbers: `(storage, storage_offset, size, stride)`.
- **Stride is a jump size**, not a coordinate. `stride[d]` = positions in storage to skip per step in dim d.
- **Tensor index ≠ storage index.** Tensor index is a tuple of per-axis coords; storage index is a single int. The formula `storage_offset + Σ stride[d]*index[d]` is the bridge.
- Integer indexing (`x[1]`) **kills** the axis; slicing (`x[1:2]`) **keeps** it with size 1. Predictable shape change.
- A slice/view shares the parent's Storage. Mutations propagate. `.clone()` allocates a fresh Storage and breaks the link.
- Slicing is `O(1)` regardless of tensor size — only metadata changes, no data is copied.
- `storage_offset` advances by `stride[d] * k` when you pick integer `k` along dim `d`. So `points[:, 1]` has offset `1`, not `0`.
- "Spooky" mutation bug: `batch = big_tensor[0]; batch += 1.0` mutates `big_tensor`. Fix: `.clone()`.
- **Transpose is metadata-only** — `.t()` swaps `size` and `stride`, leaves storage untouched. `id(a.storage()) == id(a.t().storage())`.
- `.t()` is shorthand for `.transpose(0, 1)` and **only** works on 2D. Use `.transpose(d1, d2)` (or `.permute(...)`) for higher-rank tensors.
- **Stride-swap = transpose** because stride is "jumps in storage per step." Swapping the jump-rules swaps the axes. No data moves.
- Double transpose `t().t()` returns the original metadata exactly — same storage, offset, size, stride.
- **Contiguous rule (2D, shape `(R, C)`):** stride = `(C, 1)`. Row-stride **equals number of columns**.
- **Don't use the "off-by-1" heuristic** for contiguity — it's a coincidence on `(3, 2)` and breaks for any other shape.
- Transposing a contiguous tensor produces a **non-contiguous** view (unless an axis has size 1).
- Non-contiguous tensors work for arithmetic but break ops that require packed memory (notably `.view()`). Use `.contiguous()` to copy into a row-packed buffer.
- `.reshape()` is "smart" — copies if needed. `.view()` is "strict" — errors on non-contiguous. Pick based on whether you want a guaranteed view or a guaranteed result.
- **`transpose(d1, d2)` only swaps the two named positions** in `size` and `stride`. Other axes are untouched. It is **not** a full reversal of the tuple.
- For full axis re-ordering use `permute(...)`. `.t()` is 2D-only shorthand for `transpose(0, 1)`.
- **Contiguous stride formula:** `stride[d]` = product of all sizes to the right of `d`; `stride[N-1] = 1`. Build leftward by multiplying.
- Trick to remember the formula: "to skip one step along axis `d`, you must jump over everything that lies *inside* that axis."
- For shape `(R, C)` the formula gives `(C, 1)`. For `(D₀, D₁, D₂)` it gives `(D₁·D₂, D₂, 1)`. Generalizes to any rank.
- Definition of contiguous (book): values laid out from the **rightmost dim outward**, packed tight. Last axis varies fastest in storage.
- Why contiguity matters for performance: sequential memory reads are dramatically faster than scattered reads on real CPUs/GPUs (data locality).
- Transposing a contiguous tensor produces a non-contiguous view **except** when one of the swapped axes has size 1 (size-1 axis is "trivial" — its stride doesn't affect layout).
- **Tensors and NumPy arrays use the same model:** metadata over a flat 1D buffer. `.numpy()` and `from_numpy()` are zero-copy bridges across the two libraries.
- **Pick persistence by audience.** `torch.save/load` for PyTorch-only. **HDF5** when you need other tools/languages or your dataset doesn't fit in RAM.
- HDF5's killer feature is **lazy disk access**: open a file without loading it; index into it to pull only the bytes you need. Essential for datasets larger than RAM.
- An h5py `dset` is a **handle**, not a copy. It depends on the open file. After `f.close()`, accessing it raises an exception.
- **Copy vs handle survival:** anything you want to use after `f.close()` must be a copy in RAM (e.g., `torch.from_numpy(dset[...])`), not a raw handle.
- `torch.from_numpy(dset[-2:])` copies (unlike a normal `from_numpy`) because (a) disk→RAM is a copy by definition, and (b) the source buffer's life is tied to the open file.
