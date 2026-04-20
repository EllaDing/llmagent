# F24 Lecture 1 — Foundation of LLMs and Reasoning
**Speaker:** Denny Zhou (Google DeepMind)

- Humans learn from a few examples because they can reason, not because they learn from statistics
- Few-shot prompting with reasoning traces can match traditional ML models trained on large labeled datasets

---

## Do Intermediate Steps Help? (Yes — and here's the history)

CoT is not a concept invented in 2022. It crystallized from independent lines of work all converging on the same insight:

| Year | Paper | Domain | Mechanism |
|------|-------|--------|-----------|
| 2017 | [Rationale Generation, ACL](https://aclanthology.org/P17-1015.pdf) | NLP / math word problems | Training with natural language rationales as supervision |
| 2021 | [Scratchpads, DeepMind](https://arxiv.org/abs/2112.00114) | Program synthesis (code) | Model writes intermediate computation steps to a scratchpad |
| 2021 | [GSM8K, OpenAI](https://arxiv.org/abs/2110.14168) | Math benchmarking | Fine-tuning GPT-3 on step-by-step solutions + verifier ranking |
| 2022 | [Chain-of-Thought, Google Brain](https://arxiv.org/abs/2201.11903) | General reasoning | Few-shot prompting with reasoning traces — no fine-tuning needed |

**Unified takeaway:** regardless of the mechanism (training, fine-tuning, or prompting), when provided with examples that include intermediate steps, LLMs generate responses that also include intermediate steps.

---

## Does Including a Reasoning Strategy Help Further? (Yes)

CoT shows how to think through a problem, but it fails when test problems are harder than the demonstrations. Humans don't just follow steps — they apply strategies.

> **Intellectual root — *How to Solve It* (George Pólya, 1945)**
> A classic book on mathematical problem-solving heuristics. Denny Zhou cited two of its core principles as direct inspiration:
> 1. **Decompose and recombine** → inspired Least-to-Most Prompting
> 2. **Find a related, already-solved problem** → inspired Analogical Prompting

**[Least-to-Most Prompting, ICLR 2023](https://arxiv.org/abs/2205.10625)** *(Pólya: decompose & recombine)*
- Decompose problem into simpler subproblems, solve sequentially feeding answers forward
- With only 0.1% of training examples, achieves perfect generalization
- GPT-3: 16% → 99%+ on SCAN benchmark
- **Compositional Generalization:** training examples much simpler than test examples, yet model handles hard cases

---

## Why Are Intermediate Steps Helpful?

**[CoT Empowers Transformers to Solve Inherently Serial Problems, ICLR 2024](https://arxiv.org/abs/2402.12875)**

Transformers are parallel machines; many problems are inherently serial (each step depends on the previous). CoT bridges this gap by externalizing serial computation into the output stream.

- **AC⁰:** class of problems solvable by constant-depth, polynomial-size circuits — what a fixed-depth transformer without CoT is bounded to
- **Boolean circuit of size T:** CoT with T steps ≈ T gates of reasoning power

| Claim | Proven result |
|-------|--------------|
| With CoT, any serial problem solvable if depth > constant | Constant-depth transformer + T CoT steps solves any problem of boolean circuit size T |
| Without CoT, serial problems require huge depth or impossible | Without CoT, bounded to AC⁰ — excludes most serial problems |

Depth and CoT steps are interchangeable. CoT is not a prompting trick — it fundamentally expands what transformers can compute.

---

## Can We Trigger Intermediate Steps Without Demonstrations?

**[Zero-Shot Reasoners, NeurIPS 2022](https://arxiv.org/abs/2205.11916):** "Let's think step by step" — no examples needed. MultiArith: 17.7% → 78.7%; GSM8K: 10.4% → 40.7%. Reasoning is latent; the prompt unlocks it. **Limitation:** generally worse than few-shot CoT.

---

## Beyond Zero-Shot: LLM Self-Generates Its Own Examples

> **Stefan Banach:** *"A mathematician is a person who can find analogies between theorems; a better mathematician is one who can see analogies between proofs and the best mathematician can notice analogies between theories. One can imagine that the ultimate mathematician is one who can see analogies between analogies."*
>
> | Banach's level | LLM reasoning analog |
> |----------------|----------------------|
> | Analogies between theorems | Few-shot CoT — surface similarity |
> | Analogies between proofs | Least-to-Most — same strategy applies |
> | Analogies between theories | Analogical Prompting — cross-domain structure |
> | Analogies between analogies | Frontier — frameworks analogous to frameworks |

**[LLM as Analogical Reasoners, ICLR 2024](https://arxiv.org/abs/2310.01714)** *(Pólya: find a related resolved problem)*
- Before solving, prompt LLM to recall and generate relevant examples and knowledge for that specific problem
- Outperforms both zero-shot and manual few-shot CoT on GSM8K, MATH, Codeforces, BIG-Bench
- **Takeaway:** adaptively generate problem-specific examples rather than using a fixed set

---

## CoT Without Any Prompting

**[CoT Reasoning Without Prompting, NeurIPS 2024](https://arxiv.org/abs/2402.10200)** *(Xuezhi Wang & Denny Zhou)*

Every prior technique still requires something from the human. This paper asks: is reasoning so embedded that we don't need to trigger it at all?

**CoT-Decoding:** instead of greedy decoding, examine top-k alternative tokens — reasoning paths appear naturally in those alternatives.
1. Branch into top-k tokens at first decoding step → k candidate sequences
2. Score each path by answer confidence (probability gap between top-1 and top-2 tokens at answer positions)
3. Pick the most confident path — 88% of top-confidence paths on GSM8K contain reasoning chains

**Key insight:** reasoning is a property of the model that decoding strategy either surfaces or conceals. Greedy decoding was suppressing reasoning that was always there.

---

## The True Objective: Marginalizing Over Reasoning Paths

```
P(final answer | problem) = Σ_r P(r, final answer | problem)
                          = Σ_r P(final answer | r, problem) × P(r | problem)
```

Greedy finds `arg max P(r | problem)` — the wrong objective. We want the most probable *answer*, not path.

**Self-Consistency ([Wang et al., 2022](https://arxiv.org/abs/2203.11171)):** approximate the sum via sampling — sample N paths, take majority vote. More consistent = more likely correct: correct answers accumulate mass from many paths; wrong answers only from paths that make that specific error.

**Caveat:** breaks down with systematic model bias.

| Technique | Relationship to the sum |
|-----------|------------------------|
| Greedy decoding | Ignores the sum — wrong objective |
| CoT-Decoding | Approximates with k samples, selects by confidence |
| Self-Consistency | Monte Carlo approximation via majority vote |

**[Universal Self-Consistency, 2023](https://arxiv.org/abs/2311.17311)** *(Xinyun Chen)*: extends self-consistency to free-form answers — instead of majority voting, prompt the LLM itself to select the most consistent response among N candidates.

---

## Limitations of LLM Reasoning That Affect Intermediate-Step Performance

Three bundled limitations — all showing LLMs reason along surface prompt structure, not abstract logic:

1. **Irrelevant context** — adding distracting sentences degrades performance; the model cannot ignore noise
2. **Cannot self-correct** — [LLMs Cannot Self-Correct Reasoning Yet, ICLR 2024](https://arxiv.org/abs/2310.01798): intrinsic self-correction degrades performance; only extrinsic signal (code execution, search, human feedback) genuinely helps. *Oracle feedback*: triggers correction only on wrong answers using ground truth — unavailable at test time, serves as research upper bound. Supporting: [Teaching LLMs to Self-Debug, 2023](https://arxiv.org/abs/2304.05128) — rubber duck debugging (explain code in natural language before fixing) surfaces errors that reading alone misses
3. **Premise order matters** — [Premise Order Matters, ICML 2024](https://arxiv.org/abs/2402.08939): permuting logically equivalent premise order causes >30% performance drop; models need premises in the sequence matching the reasoning steps

**Common thread:** LLMs are not abstract logical reasoners — they are sensitive to surface presentation, ordering, and context content. In long CoT chains, these effects compound.

---

## What's Next — First Principles Thinking

> *"If I were given one hour to save the planet, I would spend 59 minutes defining the problem and one minute resolving it."* — Einstein

Takeaway from Denny Zhou:
1. Define the right problem to work on
2. Solve it from first principles
