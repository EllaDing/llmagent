# F24 Lecture 2 — Planning and Tool Utilization
**Speaker:** Shunyu Yao (OpenAI)

---

## Defining "Agent"

An **agent** = an "intelligent" system that interacts with some "environment"
- Definition is fluid — depends on how you define "intelligent" and "environment"
- Both concepts change over time

**How do you define "intelligent"?**
- **Behaviorist:** outputs indistinguishable from a human (Turing Test) — defines by external behavior, not internal mechanism
- **Goal-directed:** reliably achieves goals across a wide range of environments — emphasizes adaptability over narrow performance
- **Learning-based:** improves from experience — distinguishes from hard-coded systems
- **Resource-bounded:** does well given limited time, compute, and information — reasoning well under constraints matters more than perfect reasoning with infinite resources
- **Key insight:** intelligence is a moving target — what counts shifts as the bar rises (chess AI → search algorithm; LLMs → next frontier). Intelligence is defined by what humans can't yet do routinely

---

## Three Layers of LLM Agents

| Layer | Mechanism | Reasoning? | Example |
|-------|-----------|-----------|---------|
| 1 — Text Agent | Rules or learned policy over text | No | ELIZA, LSTM-DQN |
| 2 — LLM Agent | LLM selects actions | Implicit | SayCan, Language Planner |
| 3 — Reasoning Agent | LLM reasons explicitly before acting | Yes — explicit | ReAct, AutoGPT |

**Layer 1 — Text Agent:** uses text as both action and observation
- **ELIZA (1966):** one of the first chatbots, by Joseph Weizenbaum at MIT. Pattern matching and keyword substitution to simulate a psychotherapist. No real understanding — pure rule-based text transformation. Famous for the "ELIZA effect": users attributed understanding and emotion to it despite its simplicity.
- **LSTM-DQN:** deep RL agent for text-based games (Zork-style). Uses an LSTM to encode game state (room description, inventory, last feedback), then DQN selects actions (verb + object). Pre-LLM example of a learned text agent.

**Layer 2 — LLM Agent:** uses LLM for high-level reasoning and planning
- **[SayCan, Google 2022](https://arxiv.org/abs/2204.01691):** LLM plans what to do; low-level robot skills execute how. Skill value functions "ground" the LLM by scoring which actions are physically feasible. PaLM-SayCan: 84% planning success, 74% execution success.
- **[Language Planner, 2022](https://wenlong.page/language-planner/):** LLMs decompose high-level tasks into low-level steps with zero additional training — purely from pretraining knowledge.

**Layer 3 — Reasoning Agent:** LLM reasons explicitly before acting. **Key focus of the field.**
- **ReAct:** interleaves reasoning and acting in a single loop
- **AutoGPT:** autonomous agent for long-horizon goals with minimal human intervention

---

## Brief History of LLM Agents

Two parallel research lines developed independently, then converged at ReAct:

```
Reasoning (Lecture 1)          Acting (grounding / tool use)
CoT, scratchpads,         ←→   SayCan, Language Planner,
least-to-most, ...              text games, ...
         \                           /
          \                         /
           ——————— ReAct ———————————
                (reasoning agent)
```

**What ReAct unlocked:**
- New applications: web browsing, software engineering, scientific discovery
- New methods: memory, learning, planning, multi-agent systems

---

## Concrete Example: QA as a History of Agent Capabilities

Each type of question exposed a new failure mode, driving a new technique:

| Question type | Requirement | Technique |
|--------------|-------------|-----------|
| Simple factual | None | Direct LLM answer |
| Chain of logic | Reasoning | CoT, intermediate steps |
| News / current events | External knowledge | RAG — retriever + corpora |
| No available corpora | Live knowledge access | Tool use — special tokens trigger web search |
| Complex math | Precise computation | Code augmentation — LLM writes and executes code |

**RAG:** adds external corpora + retriever to LLM pipeline
- **BM25:** classical sparse retrieval — keyword overlap with TF-IDF weighting; fast, no neural network, strong baseline
- **DPR (Dense Passage Retrieval):** two BERT encoders embed query and passages into dense vectors; nearest-neighbor search captures semantic similarity beyond keyword match

**Tool use:** introduce special tokens (e.g. `[SEARCH]`, `[CALL]`) — model emits them when it needs external information; result injected back into context. Generalizes RAG to any tool.

**Code augmentation:** LLM generates and executes code rather than reasoning numerically — more reliable than CoT for computation-heavy problems.

---

## The Piecewise Problem → The Need for Abstraction

Looking back at QA: many benchmarks, each with a bespoke solution. The field felt fragmented.

**The abstraction:** every solution reduces to one or both of two primitives:
- **Reasoning** — thinks; generates intermediate steps, plans, deductions
- **Acting** — does; calls tools, retrieves information, executes code, interacts with environment

**ReAct** unifies both — one architecture handles all QA types.

---

## ReAct: A New Paradigm of Agents that Reason and Act

**[ReAct: Synergizing Reasoning and Acting in Language Models, 2022](https://arxiv.org/abs/2210.03629)**

**The thought-action-observation loop:**
```
Thought     → reason about what to do next
Action      → call a tool / query an environment
Observation → result returned and added to context
(repeat until done)
```

**Formal contrast — traditional agent vs. reasoning agent:**

*Traditional agent:*
- Fixed action space **A** defined by environment
- Context: `c_t = (o_1, a_1, o_2, a_2, ..., o_t)`
- Policy: `π(a | c_t)` — maps context to distribution over **A**
- Every action affects the world

*Reasoning agent:*
- Action space extended with **reasoning** — internal actions that don't affect the world
- Reasoning only updates the agent's own context/memory, which then shapes external actions
- Key distinction: **reasoning is an internal action**

```
Traditional:  c_t → π(a) → world changes → o_{t+1}
Reasoning:    c_t → think → c_t' (updated) → π(a) → world changes → o_{t+1}
```

**The synergy — bidirectional:**
- **Acting supports reasoning:** grounds the model in facts, prevents hallucination, provides new information
- **Reasoning guides acting:** decides what to act on, when to call a tool, how to interpret results

**Results:**

| Domain | Benchmark | Result |
|--------|-----------|--------|
| Question answering | HotpotQA | Reduces hallucination via Wikipedia lookups |
| Fact verification | Fever | Reduces hallucination via Wikipedia lookups |
| Interactive decision making | ALFWorld | +34% over baselines |
| Interactive decision making | WebShop | +10% over baselines |

---

## Memory: Short-Term vs. Long-Term

Rooted in human psychology and neuroscience — CoALA borrows this taxonomy as a principled mapping, not just metaphor.

**Short-term memory = context window.** Properties:
1. **Append-only** — can only add, never edit or remove
2. **Limited context** — fixed maximum length; old information dropped
3. **Limited attention** — attention degrades over long contexts; middle content underweighted
4. **Does not persist** — lost when session ends

Human analog: working memory (~7±2 chunks, Miller 1956) — active, conscious, volatile.

**Long-term memory** — persists beyond the context window, across sessions.

Human analog — three subtypes:
- *Episodic* — personal experiences and events
- *Semantic* — world knowledge and facts
- *Procedural* — skills and how-to knowledge (implicit)

**The defining distinction:** short-term = currently active and accessible; long-term = persists when attention moves away. Boundary: duration + capacity + whether active maintenance is required.

**Agent mapping:**

| Human memory | Agent analog |
|-------------|-------------|
| Short-term / working memory | Context window |
| Episodic long-term | Reflexion's memory buffer, Generative Agents' memory stream |
| Semantic long-term | LLM weights, RAG knowledge base |
| Procedural long-term | Voyager's skill library |

---

## Long-Term Memory: Examples

**[Reflexion, 2023](https://arxiv.org/abs/2303.11366)** — verbal reinforcement learning

Replaces RL's scalar reward → weight update with natural language feedback stored as text:
```
Standard RL:  reward → weight update → better policy
Reflexion:    feedback → verbal reflection → stored in memory → better decisions next attempt
```
1. Agent attempts task (ReAct-style)
2. Receives feedback on success/failure
3. Generates verbal reflection: *"I failed because I didn't verify X"*
4. Stores in episodic memory buffer (long-term)
5. Reads prior reflections on next attempt → better decisions

Result: 91% pass@1 on HumanEval, surpassing GPT-4's 80%.

---

**[Voyager, 2023](https://arxiv.org/abs/2305.16291)** — code-based skill library

Reflexion stores verbal reflections. Voyager stores *executable code*.

Setting: Minecraft — continuous open-ended exploration without human intervention.

Three components:
1. **Automatic curriculum** — agent generates its own exploration goals
2. **Skill library** — growing repository of executable, compositional code snippets
3. **Iterative prompting** — writes code, observes errors, revises until verified, then stores

Why code > text as memory: executable (directly reusable), compositional (complex skills call simpler ones), prevents catastrophic forgetting.

---

**[Generative Agents, 2023](https://arxiv.org/abs/2304.03442)** — full experiential memory

Setting: Sims-like sandbox — agents live, work, socialize autonomously.

Three components:
1. **Memory stream** — append-only record of all experiences in natural language
2. **Reflection** — periodically synthesizes memories into higher-level beliefs
3. **Dynamic retrieval** — selects most relevant memories (by recency, importance, relevance) to fit in context window

Emergent behavior: from one suggestion about a Valentine's Day party, agents autonomously spread invitations, formed relationships, coordinated attendance over two simulated days.

What's distinct: stores full experiential texture → forms a persistent *identity* (beliefs, relationships, preferences).

---

**The LLM itself as long-term memory:** model weights encode world knowledge from pretraining — implicit, static, cannot store agent-specific experience.

**Progression:**

| System | What is stored | Format | Reusable? |
|--------|---------------|--------|-----------|
| Reflexion | Verbal reflections on failures | Text | As context only |
| Voyager | Learned behaviors | Executable code | Directly callable |
| Generative Agents | Full experiential history + reflections | Natural language | Via retrieval |
| LLM weights | World knowledge | Parameters | Always available (static) |

---

## Unified Agent Architecture

**[CoALA: Cognitive Architectures for Language Agents, 2023](https://arxiv.org/abs/2309.02427)**

Any agent can be expressed through three dimensions:
1. **Memory** — working (context), episodic, semantic, procedural
2. **Action space** — internal memory operations + external environment interactions
3. **Decision-making procedure** — planning, reasoning, retrieval, learning

```
External world
      ↕ (observations / actions)
Code-based controller
      ↕ retrieves relevant knowledge
Long-term memory  ←  learns from external knowledge
      ↕
Short-term memory (context window)
      ↕ informs decision
Action
```

**Internal memory vs. external environment:**

| | Internal Memory | External Environment |
|--|----------------|---------------------|
| Definition | Controlled by and persisting *within* the agent | Exists *independently* of the agent |
| Types | Working memory, episodic, semantic, procedural | Physical, dialogue (humans/agents), digital (APIs, games, websites) |
| Control | Agent owns and manages | Agent can only interface with it |

**The boundary:** grounding procedures convert external feedback into internal text representations. The line: *does the agent own it, or does it interface with it?*

---

## History of Agent Environments

| Era | Environment | Practical? | Scalable? |
|-----|------------|-----------|----------|
| 1 | **Physical world** | Yes — real impact | No — slow, costly, unsafe at scale |
| 2 | **Simulations / games** | No — sim-to-real transfer is hard | Yes — unlimited cheap interactions |
| 3 | **Digital world** | Yes — real impact | Yes — software is cheap to interact with |

**Digital world is the sweet spot** — practical and scalable. Why web browsing, software engineering, and tool-use agents dominate today.

---

## 5 Open Dimensions for Future Agent Research

**1. Training — [FireAct: Toward Language Agent Fine-Tuning, 2023](https://arxiv.org/abs/2310.05915)**
- Problem: LLMs fine-tuned for general language, not agent use — LLM builders and agent builders are different people
- FireAct: fine-tune backbone LM on agent trajectories (reasoning + action sequences)
- Llama2-7B fine-tuned on 500 GPT-4 trajectories → 77% HotpotQA improvement
- Broader paradigm — **model-agent synergy:** use agents to generate training data, then fine-tune models on it
  - Labels capture outcomes ("a good blog post"); agents can generate the process ("how to write a good blog post") — reasoning trajectories become training signal
  - Goal: improve planning, self-evaluation, calibration

**2. Interface — [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering, 2024](https://arxiv.org/abs/2405.15793)**
- Insight: if you can't optimize the agent, optimize the environment
- Analogy: a software engineer with a better IDE performs better — same for LLM agents
- **Agent-Computer Interface (ACI):** purpose-built for LM agents, analogous to HCI for humans
- SWE-agent's ACI improves code editing, repo navigation, test execution
- Results: 12.5% pass@1 on SWE-bench, 87.7% on HumanEvalFix (SOTA at time)

**3. Robustness**
- How do we make agents work reliably in real-world conditions?
- Agents that pass benchmarks often fail in deployment: edge cases, distribution shift, cascading errors

**4. Human**
- Agents must work *with* humans — handle dynamic, ambiguous instructions, follow domain-specific policies, remain reliable across multi-turn interactions

**5. Benchmark — [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains, 2024](https://arxiv.org/abs/2406.12045)**

Three components simulate realistic customer-facing scenarios:
1. **Realistic database + tool APIs** — actual business tools backed by realistic data
2. **Domain-specific policy document** — agent must follow rules, not just answer correctly
3. **LLM-simulated user** — dynamic, realistic, sometimes ambiguous requests

**Critical gap — benchmarks vs. what humans need:**

| | Benchmark mindset (e.g. AlphaCode) | Human-in-the-loop (e.g. customer service) |
|--|-----------------------------------|--------------------------------------------|
| Metric | pass@k — succeed *at least once* in k tries | Robustness — never fail across N tries |
| Sampling | More samples → higher score | Every failure matters |
| Analog | Coder who eventually finds a solution | Customer service agent who must always be reliable |

**pass^k vs pass@k:**

| Metric | Question | Higher k means... |
|--------|----------|-------------------|
| pass@k | Succeed at least once in k tries? | Easier to pass |
| pass^k | Succeed every time across k tries? | Harder to pass |

An agent failing 1 in 8 times scores pass^8 = 0 — the right metric for deployment. GPT-4o: <50% per task, pass^8 <25% in retail. Current agents are far from deployment-ready.
