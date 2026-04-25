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

## 14. Precision Rules (Lessons from Mistakes)

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
