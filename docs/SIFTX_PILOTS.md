# SiftX: what the CPU pilots taught us

All pilots: MQAR, 128 tokens, 16 key-value pairs, d_model 64, 4 layers, batch 32, lr 3e-3, CPU.
Chance accuracy ≈ 2.5%. Logs are in `experiments/logs/`.

| Run | What changed | Result |
|---|---|---|
| `pilot6_mqar_hybrid` | Sift + one **full** attention layer, no gate | **100%** by step 900 (sudden jump at ~800) |
| `pilot6_mqar_siftx` | learned memory + learned gate, budget 30% from step 0 | 2% (chance); gate ~14% open |
| `diag_gate_only` | **full** attention + learned gate | 2%: stalled *even while every gate was still open* |
| `diag_memory_only` | learned memory, gate always open | slowly learning (loss 3.69 → 3.56 by step 700) |
| `pilot7_mqar_siftx` | + gate no longer sends gradient into the hidden state, budget warm-up | 4% at step 800, gate drifting with the budget |
| `pilot8_mqar_siftx` | + **curriculum**: gate forced open for 1000 steps, then budget tightens | follows the hybrid's curve ~300 steps behind (loss 3.25 at step 1000); **stalls once the budget starts tightening** before the jump; 10% at step 2000, gate 48% open |

## Diagnosis, in plain words

1. **The gate was poisoning the model.** Its budget penalty flowed back into the shared hidden
   state. With full attention and every gate open, the model still couldn't learn. Fix: the gate
   reads the hidden state through `detach()`, so it controls retrieval without reshaping
   what the model has learned.
2. **Two things can't be learned at once.** The memory has to learn *what to store*, and the gate
   has to learn *when to read*. Each is useless until the other works, so learning both from step 0
   sits at chance. Fix: a curriculum. First keep the gate open so the memory can learn, then
   tighten the budget.
3. **The order matters a lot.** If the budget tightens before the model has made its "aha" jump on
   the task, learning stalls (pilot 8). The GPU suite therefore keeps the gate open for the
   first 60% of training, tightens over the next 25%, and holds at budget for the final 15%
   (`research/make_configs.py: gate_curriculum`).
4. **Learned top-k memory learns more slowly than full attention** (about 300 steps behind in
   pilot 8), but it does learn. That is the price of attending to M + C tokens instead of
   all L tokens.

## What this means for the claims

- A learned "when to retrieve" gate is **not free**: it needs a curriculum, and without one it
  fails completely. This is a real, reportable negative result.
- The GPU suite includes `siftx_always` (memory, no gate) and `hybrid` (full attention, no gate)
  so the cost of the memory and the cost of the gate can be measured separately.

## Counterfactual gate pilot (`pilot9_mqar_siftx_cf`, same schedule as pilot 8)

| Measure | pilot 8 (learned gate) | pilot 9 (counterfactual gate) |
|---|---|---|
| query accuracy at step 2000 | 10% | 9% |
| loss after the gate takes over (step 1000 → 2000) | fell to 3.13 by step 1400, then **crept back up** to 3.16 | **kept falling** (3.59 → 3.45) |
| gate open on questions / on other tokens | **91% / 41%** | 4% / 5% |

- The learned (budget + straight-through) gate **did learn *when* to look**: it opens on questions far
  more than elsewhere, even though the model doesn't solve the task yet.
- The counterfactual gate did not hurt learning, but its labels ("did retrieval lower the loss
  here?") carry little signal until retrieval actually helps. On CPU that point never arrives, so
  the gate ends up mostly closed. The GPU runs, which train long enough to solve the task, decide this.
