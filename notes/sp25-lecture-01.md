# SP25 Lecture 1 — Inference-Time Techniques for LLM Reasoning
**Jan 27, 2025** | **Xinyun Chen** (Google DeepMind)
[Video](https://www.youtube.com/live/g0Dwtf3BH-0) · [Slides](https://llmagents-learning.org/slides/inference_time_techniques_lecture_sp25.pdf) (74)

> Overlaps heavily with [F24 Lecture 1](f24-lecture-01.md) (Denny Zhou), cross-linked inline below.
> The difference in angle: Denny Zhou asks *why intermediate steps work*. Xinyun Chen asks *how to spend an inference budget*.

---

## Framing

**Why agentic frameworks:** real tasks involve trial and error · tools and external knowledge extend what's in the weights · agentic workflows handle complexity.

**"2025 is the year of agents"** — code generation, computer use, personal assistants, robotics. The second driver is reasoning models: o1, Gemini 2.0 Flash Thinking, DeepSeek-R1, Kimi k1.5. (o3 hit 87.5% on ARC-AGI at >$1k of test-time compute per task.)

**Core idea:** get the LLM to generate a long chain of thought. Three ways to trigger it — few-shot CoT, instruction prompting, instruction tuning.

**The lecture's three parts are three ways to spend compute:**

| Part | Move | Axis |
|---|---|---|
| 1. Basic prompting | More tokens on one solution | Depth |
| 2. Search and selection | Many candidates, then pick | **Width** |
| 3. Iterative self-improvement | Revise what you have | Depth again, sequentially |

---

## Part 1 — Basic Prompting

**Read this part as history.** The results predate post-training. Base models had no learned habit of reasoning step by step, so the structure had to come from the prompt. That makes it a clean experiment — no CoT-shaped SFT data muddying the result — and it shows the capability was already **latent in pretraining**, waiting to be elicited.

**CoT gains grow with model size** (ties to *Emergent Abilities*, Wei et al. 2022). → [F24 L1](f24-lecture-01.md)

**Zero-shot CoT** ("let's think step by step") beats plain zero-shot but is **still worse than few-shot CoT**. That gap motivates everything that follows: how do we get few-shot quality without hand-labeling exemplars?

### Analogical prompting
Have the LLM generate its own exemplars — and also its own **high-level knowledge** about the problem. Inspired by human analogical reasoning (Pólya, *How to Solve It*, 1945). → [F24 L1](f24-lecture-01.md)

**Stronger models benefit more.** Weak models gain little. Strong models beat CoT using hand-written *or retrieved* exemplars, because the self-generated CoT is **tailored to that model**.

> **Exemplar vs. example.** An *example* is training data — learned from via gradients. An *exemplar* is a demonstration in the prompt — costs tokens, changes no weights. Keeping them distinct is what makes "14 exemplars vs. 15,000 training examples" an honest comparison.

### Automating prompt design
Models are very sensitive to prompt wording, and there's no principled way to write the best prompt. So automate it.

- **[APE](https://arxiv.org/abs/2211.01910)** (ICLR 2023) — LLM proposes candidate instructions; each gets scored.
- **[OPRO](https://arxiv.org/abs/2309.03409)** (ICLR 2024, Chen coauthor) — feed the LLM its own past attempts as sorted (solution, score) pairs and ask for better ones. On GSM8K: 60.8% → beats "let's think step by step" by ~8%, matching few-shot CoT at 80.7%. Improves with more steps, then plateaus.

### What CoT actually gives you
1. **Variable computation** — harder problems get more steps
2. **Reasoning strategies** — decomposition, planning, and so on
3. So you can **name the strategy you want** in the prompt

### Least-to-most prompting
Break the problem into subproblems and solve them in order, **feeding each answer into the next call's context**. That accumulation across *separate calls* is the difference from CoT, which is one continuous trace in a single pass. Least-to-most is a small pipeline, not a prompt. → [F24 L1](f24-lecture-01.md)

**SCAN result:** 14 exemplars → **≥99%** on any split including the length split, versus **16%** for CoT, versus specialized models *trained* on 15,000+ examples. The point isn't efficient training. It's **no training at all**.

**Dynamic least-to-most** (Drozdov, Schärli, … Chen, Zhou, ICLR 2023) generates the decomposition per problem instead of fixing it. Evaluated on CFQ.

### SELF-DISCOVER
Once per **task**: SELECT from 39 atomic reasoning modules → ADAPT them → IMPLEMENT as a plan. Then every instance just follows that plan.

| | Least-to-most | SELF-DISCOVER |
|---|---|---|
| Structure built | Per instance | Per task — cost amortizes |
| Strategy options | One: decompose | 39 modules, composed |
| Human effort per task | Write decomposition demos | None |
| Portability | Tied to the task | Transfers across model families |

Up to **+32%** over CoT on BBH/T4D/MATH, and **>20%** over CoT-Self-Consistency at **10–40× less** compute.

The argument: least-to-most only knows one move. That's great for compositional problems and wrong when the task needs perspective-taking or working backward. The tell is that SELF-DISCOVER can pick a **verification** module ("double-check for errors") — least-to-most structurally cannot, since its only question is "what smaller problem comes first?"

---

## Part 2 — Search and Selection

Don't limit the model to one solution. Multiple branches let it recover from a mistake in any single generation. You can branch at two granularities: **whole solutions**, or **next reasoning steps**.

**The hard part:** picking the best candidate **at inference time with no oracle.** Generating candidates is cheap. Knowing which one is right is not. Everything below is a different answer to that.

### Self-consistency
Sample several reasoning paths, keep the **most common final answer**. The paths don't need to match — only the answers are compared.

It's a Monte Carlo approximation of marginalizing over reasoning paths. Correct answers pick up mass from many different derivations; a wrong answer only gets mass from paths making that same specific error. → [F24 L1](f24-lecture-01.md), which also notes it **breaks under systematic model bias**.

- **Beats Sample-and-Rank** (picking the highest log-probability answer), and scales much better with more samples — *"unless the model is trained to be a good verifier."*
- **Diversity is the whole game.** Beam search and greedy prompt-variant ensembles are the baselines that *lose*: high-probability paths are near-duplicates that agree for the same reason. Use high temperature or nucleus sampling.
- **Consistency tracks accuracy**, so it doubles as a **confidence signal**. That's the more durable use — send low-consistency cases to more compute, a better model, a tool, or a human.

### Execution consistency (AlphaCode)
Self-consistency needs answers you can compare for equality. Programs break that: correct solutions look different, similar-looking programs behave differently. So compare **behavior** instead.

1. Drop programs that fail the **given** test cases — a real but weak oracle
2. **Generate new test inputs** — inputs only, no expected outputs
3. Run every surviving program on every input
4. **Cluster** programs by identical behavior
5. Pick from the biggest cluster

The clever bit is step 2: the input generator doesn't need to know any correct answers. Agreement among the programs supplies the expected behavior. Clustering beats filtering alone, but **there's still a real gap from oracle selection**.

> This is the asymmetry running under the whole lecture. Where a free semi-oracle exists — run the code, check the math, verify the proof — consensus becomes something much stronger. Where none exists, you're left with the model grading itself. Which is exactly where Part 3 falls apart.

### Universal Self-Consistency
Plain self-consistency needs an answer-extraction step, so it doesn't work on free-form output. **[USC](https://arxiv.org/abs/2311.17311)** (Chen et al. 2023) asks the LLM to pick the most consistent response instead. Helps on summarization and open-ended QA. → also in [F24 L1](f24-lecture-01.md)

### Trained verifiers: ORM vs. PRM
- **ORM** — score the whole solution (Cobbe et al. 2021)
- **PRM** — score each step (*Let's Verify Step by Step*, 2023)

Strong verifiers beat consistency-based selection, and **PRM scales better with more samples**. Her caveats: results depend heavily on **verifier quality**, and a verifier **may not generalize to other tasks**.

### Tree of Thoughts
Waiting for complete responses wastes a step-wise scorer. ToT explores promising **partial** solutions instead. Each step does thought generation (propose next steps) plus thought evaluation (score the state) — evaluation can be voting-based, with the LLM voting repeatedly.

On Game of 24, ToT with BFS scales better than CoT per token spent, and extends to MCTS. It **depends on good self-evaluation** — precisely the assumption Part 3 knocks down.

---

## Part 3 — Iterative Self-Improvement

Go **deeper** instead of wider: revise rather than resample. Sampling many solutions reduces one-shot mistakes but gives you **no feedback loop**.

**Reflexion / Self-Refine:** the LLM critiques its own output, using external evaluation where available, then revises.

**These work when the external evaluation is good.** Reflexion helps where solid evaluation heuristics exist (ALFWorld). On HotPotQA, the external evaluation hands over answer correctness at each step — that's an oracle.

**Self-debugging** ([Chen, Lin, Schärli, Zhou, ICLR 2024](https://arxiv.org/abs/2304.05128)) is the natural fit, because running code is free feedback — humans debug better in an IDE for the same reason. Better feedback helps more: a generic "this is wrong" < unit test results < line-by-line explanation.

### The negative result — the heart of the lecture
Earlier work showing self-correction gains **used an oracle verifier**. Most real use cases don't have one.

**[LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)** (Huang, Chen, Mishra, Zheng, Yu, Song, Zhou — ICLR 2024):

- Without an oracle, the model has to judge its own correctness — and **judges it wrong**, so performance gets **worse** after self-correction
- Rewording the feedback prompt changes how often it keeps its first answer, but none of the variants beat the original performance
- **Multi-agent debate doesn't beat self-consistency.** Without a good evaluator, it's just a worse way to spend the same tokens

> The through-line of the whole lecture: **verification is the bottleneck, not generation.** Everything that works here either has an external oracle or uses consensus as a stand-in. Nothing works when the model grades itself.

### Spending the budget
- **[Snell et al. 2024](https://arxiv.org/abs/2408.03314)** — how to split budget between parallel sampling and sequential revision is **model- and task-specific**, depending on how good the model's self-reflection is
- **[Wu et al. 2024](https://arxiv.org/abs/2408.00724)** — at a fixed FLOPs budget you can sample more from a smaller model; the compute-optimal model size **changes with the budget**

### Closing: the Bitter Lesson
She ends on Sutton as the guide for designing reasoning techniques, at inference time and training time alike:

> *"The great power of general purpose methods, of methods that continue to scale with increased computation."*
> *"We want AI agents that can discover like we can, not which contain what we have discovered."*

---

## ⚠️ Staleness Pass (Aug 2026)

Recorded a week after DeepSeek-R1 — right at the hinge between hand-built inference-time structure and RL-trained reasoning. **She closes by citing the principle that dates her own Part 1.**

### What held up

**The negative result held up completely — the most durable claim in the lecture.** 2026 work confirms self-correction without external feedback is still severely limited, and has sharpened *why*: the bottleneck is **error detection, not error correction**. Models can fix an error once you point at it, but can't find it. Gains still come from outside — an execution signal, a tool-based critic, or correction trained in.

**The verification/generation asymmetry** became the organizing constraint of the field. RLVR is this lecture's insight moved into training: it works where a free oracle exists, which is why code and math got most of the gains.

**Consistency as a confidence signal** outlived consistency as a selector — it's now used for routing and abstention.

**Her PRM caveats were prescient, and understated.** 2026 adversarial work finds state-of-the-art PRMs behave as **fluency detectors rather than reasoning verifiers**: policies reach >0.9 PRM reward while true accuracy sits below 4%, with ~43% of the reward gain coming from stylistic shortcuts.

### What changed

| In the lecture | Now |
|---|---|
| Part 1 prompting techniques as practice | **Absorbed by post-training.** Reasoning models decompose, plan, and backtrack unprompted. Don't hand-write CoT scaffolds. |
| Base models with no post-training | That experimental setting is gone; everything is heavily post-trained. |
| APE / OPRO prompt optimization | Continued as DSPy optimizers — MIPROv2, now **GEPA** (~13% over MIPROv2, ~20% over GRPO with 35× fewer rollouts). OPRO is the prior work Khattab builds on in F24 Lecture 5. |
| Self-consistency as a decoding trick | Survived and **got repurposed as a training signal** — TTRL uses consensus as a pseudo-label for RL without ground truth. |
| ToT / MCTS at inference time | **Relocated.** Long-CoT models backtrack natively; tree search now mostly generates *training traces*. Explicit inference-time tree search is niche. |
| PRMs as the promising direction | Became infrastructure, but **reward hacking is the live problem**. |
| o1 / R1 / Kimi k1.5 as frontier | Several generations back. |

### Reading the lecture through its own closing slide

> *"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. … We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done."*
> — Rich Sutton, [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019)

Two things worth separating in that passage:

- **The scaling claim** — the two methods that scale arbitrarily with computation are **search** and **learning**.
- **The epistemological claim** — building in a discovery obscures *the discovering process itself*. This is a statement about research method, not only about which system wins a benchmark.

Sorting the lecture along the first axis:

| Technique | Category | Where it ended up |
|---|---|---|
| Least-to-most, SELF-DISCOVER, analogical prompting | Built-in structure | Absorbed by post-training |
| Self-consistency, ToT | Search | Self-consistency survived; ToT relocated into training-trace generation |
| RLVR (came after the lecture) | Learning | Became the dominant approach |

Two things complicate a clean sort:

- ToT was search, but the *scaffold* was still hand-built — models learned to backtrack without it.
- RLVR looks like pure learning, but the human input moved into the reward function rather than disappearing.

On the second, epistemological axis: least-to-most encodes Pólya's heuristics, which are themselves human discoveries *about how problem-solving works*. SELF-DISCOVER moves up a level — supplying a menu of 39 modules rather than one fixed method — but the inventory is still human-curated. Neither is what Sutton means by an agent that discovers.

**One distinction this suggests for evaluating any technique:** does it encode an answer, or define what counts as a good answer? A prompt scaffold does the former. A verifier or eval does the latter — it specifies success without prescribing method. That maps onto the lecture's own finding that verification, not generation, is the binding constraint.

### Verdict
**Part 1 is history. Part 2 is half-absorbed, half-relocated into training. Part 3 aged best** — because it's the negative result, and the constraint it found (models can't verify themselves) is still binding in 2026.

**Worth watching:** the three-families framing · consistency and calibration · execution consistency in AlphaCode · **all of Part 3**.
**Skim:** the Part 1 walkthrough, unless you want the history.
**Patch with:** RLVR, and the 2026 PRM reward-hacking literature.
