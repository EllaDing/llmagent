# Chapter 4: Dynamic Programming

DP assumes you have a **complete model**: the transition dynamics $p(s', r \mid s, a)$. The goal is to find the optimal value function $v_\*$ or policy $\pi_\*$.

---

## 4.1 — Policy Evaluation (Prediction)

**Problem:** Given a fixed policy $\pi$, compute $v_\pi$.

The Bellman equation for $v_\pi$ is:

$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_\pi(s')\bigr]$$

**Iterative solution:** Start with $V_0$ (arbitrary), then repeatedly apply the update:

$$V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma V_k(s')\bigr]$$

This is called an **expected update** (or full backup) — it uses the full distribution over next states. It converges to $v_\pi$ as $k \to \infty$.

**Stopping criterion:** Halt when $\max_s |V_{k+1}(s) - V_k(s)| < \theta$ for some small threshold $\theta$.

**Key insight:** Each update uses the *old* values of neighboring states to revise the current state's estimate. This is **bootstrapping**.

---

## 4.2 — Policy Improvement

**Problem:** Given $v_\pi$, can we find a better policy?

**Policy improvement theorem:** Define a new greedy policy:

$$\pi'(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_\pi(s')\bigr]$$

Then $v_{\pi'}(s) \geq v_\pi(s)$ for all $s$ — the greedy policy is at least as good.

If $v_{\pi'} = v_\pi$, you have found the **optimal** policy $\pi_*$.

**Intuition:** $v_\pi(s)$ captures the long-term value of *following $\pi$* from state $s$. Being greedy w.r.t. $v_\pi$ means: take the best first action, then follow $\pi$ afterward — which is provably at least as good as just following $\pi$ from the start.

### Why is the new policy guaranteed to be better, not worse?

**Step 1 — The core inequality.** Since $\pi'$ picks the action that maximises $q_\pi$, it satisfies by definition of $\max$:

$$q_\pi(s, \pi'(s)) = \max_a\, q_\pi(s, a) \geq q_\pi(s, \pi(s)) = v_\pi(s)$$

The only way this could fail is if $\max_a q_\pi(s,a) < q_\pi(s,\pi(s))$, which is a contradiction since $\pi(s)$ is one of the actions in the $\max$.

**Step 2 — Unroll the guarantee recursively.** That single-step advantage chains into a full trajectory bound:

$$v_\pi(s) \leq q_\pi(s, \pi'(s)) = \mathbb{E}\bigl[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s,\, A_t=\pi'(s)\bigr]$$

Apply the same inequality at $S_{t+1}$ (taking $\pi'$ there is also $\geq v_\pi$):

$$\leq \mathbb{E}\bigl[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t=s,\, \text{following } \pi'\bigr]$$

Keep unrolling to infinity:

$$\leq \mathbb{E}\bigl[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t=s,\, \text{following } \pi'\bigr] = v_{\pi'}(s)$$

**Key subtlety:** We measure "goodness" using $v_\pi$ itself — the current policy's own value function — not some external ground truth. Being greedy w.r.t. $v_\pi$ means the chosen action is the best *as judged by $v_\pi$*. Taking that action and then following $\pi$ for the rest costs at least $v_\pi(s)$. Each step's single-step guarantee chains into the next, so following $\pi'$ everywhere is at least as good as following $\pi$ everywhere.

---

## 4.3 — Policy Iteration

Alternate between evaluation and improvement until convergence:

$$\pi_0 \xrightarrow{\text{eval}} v_{\pi_0} \xrightarrow{\text{improve}} \pi_1 \xrightarrow{\text{eval}} v_{\pi_1} \xrightarrow{\text{improve}} \pi_2 \;\to\; \cdots \;\to\; \pi_*$$

Since there are finitely many deterministic policies in a finite MDP, this **must converge** in a finite number of steps.

**Cost:** Each "eval" step requires many sweeps until $v_\pi$ converges. This can be expensive for large state spaces.

### The Two Nested Loops

A very common point of confusion: policy iteration has **two nested loops**.

**Inner loop — Policy Evaluation.** Given a fixed policy $\pi$, iterate $V_0 \to V_1 \to V_2 \to \cdots$ until convergence to $v_\pi$. The policy does not change here — only our estimate of its value function.

**Outer loop — Policy Iteration.** Alternates between evaluation and improvement:

$$\pi_1 \xrightarrow{\text{eval}} v_{\pi_1} \xrightarrow{\text{improve}} \pi_2 \xrightarrow{\text{eval}} v_{\pi_2} \xrightarrow{\text{improve}} \pi_3 \xrightarrow{\text{eval}} v_{\pi_3} \xrightarrow{\text{improve}} \pi_4 \;\to\; \cdots$$

### How Does Each Improvement Step Work?

Policy improvement is "for each state $s$, independently find the action that maximizes $q_\pi(s,a)$":

$$\pi_{k+1}(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_{\pi_k}(s')\bigr]$$

One action per state. This produces a new deterministic policy.

### How Do We Go From $\pi_2$ to $\pi_3$?

The crucial step people miss: **you must re-run policy evaluation on $\pi_2$ before the next improvement**.

1. **Evaluate $\pi_2$**: Run the inner loop *again* (fresh iteration $V_0 \to V_1 \to \cdots$) to compute $v_{\pi_2}$. This is a new value function — different from $v_{\pi_1}$ in general, because $\pi_2$ behaves differently, so long-term returns from each state differ.

2. **Improve using $v_{\pi_2}$**:

$$\pi_3(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_{\pi_2}(s')\bigr]$$

Note the subscript: we use $v_{\pi_2}$, **not** $v_{\pi_1}$. If you reused the old $v_{\pi_1}$, you'd just get $\pi_2$ back (since $\pi_2$ was already greedy w.r.t. $v_{\pi_1}$).

### Why the Value Function Changes Between Iterations

$v_{\pi_1}(s)$ answers "what is the expected return if I start at $s$ and follow $\pi_1$ forever?" When the policy switches to $\pi_2$, the expected return from $s$ changes because the future actions are different. So $v_{\pi_2}(s) \neq v_{\pi_1}(s)$ in general (in fact $v_{\pi_2} \geq v_{\pi_1}$ by the improvement theorem).

Because the value function has changed, the greedy argmax can select new actions — producing $\pi_3$.

### Termination

The loop stops when $\pi_{k+1} = \pi_k$ — policy improvement produces no change. At that point:

$$\pi_k(s) = \arg\max_a q_{\pi_k}(s, a) \quad \text{for all } s$$

This is exactly the Bellman optimality equation, so $\pi_k = \pi_*$.

### Concrete Mini-Example

Imagine a 2-state MDP:

- $\pi_1$: always go left. Evaluation gives $v_{\pi_1}(s_1) = 3,\; v_{\pi_1}(s_2) = 1$.
- Improve: using $v_{\pi_1}$, argmax says "go right from $s_1$". So $\pi_2$: right from $s_1$, left from $s_2$.
- **Re-evaluate $\pi_2$**: maybe $v_{\pi_2}(s_1) = 5,\; v_{\pi_2}(s_2) = 4$. Different numbers because the policy changed.
- Improve using $v_{\pi_2}$: maybe now argmax says "go right from $s_2$" too. $\pi_3$: right everywhere.
- Re-evaluate $\pi_3$, improve, ... until argmax stops producing new actions.

---

## 4.4 — Value Iteration

**Key idea:** Don't wait for $v_\pi$ to fully converge — do *one* sweep, then immediately improve.

This collapses policy evaluation + improvement into a single update using the **Bellman optimality equation**:

$$V_{k+1}(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma V_k(s')\bigr]$$

This converges to $v_*$. Once converged, extract the greedy policy:

$$\pi_*(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_*(s')\bigr]$$

**Why this is a shortcut:** Value iteration never explicitly maintains a policy during the process. Instead of the expensive nested-loop structure of policy iteration (full evaluation → improvement → full evaluation → ...), you just bake the $\max$ directly into the value update. When $V$ converges to $v_*$, extract $\pi_*$ once at the end.

**Comparison with policy iteration:**

| | Policy Iteration | Value Iteration |
|---|---|---|
| Eval per cycle | Many sweeps (full convergence) | One sweep |
| Policy extraction | Each cycle | Once at the end |
| Character | Fewer, heavier iterations | More, lighter iterations |

---

## 4.6 — Efficiency of DP

- DP is **polynomial** in $|S|$ and $|A|$ — far better than searching all policies (there are $|A|^{|S|}$ deterministic policies).
- **Curse of dimensionality:** $|S|$ grows *exponentially* in the number of state variables. DP is tractable in theory but breaks down in practice for large state spaces — motivating the rest of the book (function approximation, sampling, etc.).
- **Asynchronous DP:** States can be updated in any order, even one at a time, and convergence still holds. Useful for prioritizing states that matter most (e.g., states the agent actually visits).

---

## 4.7 — Summary

Three core operations that recur throughout RL:

| Operation | What it does |
|---|---|
| **Policy evaluation** | Compute $v_\pi$ given $\pi$ |
| **Policy improvement** | Compute greedy $\pi'$ given $v_\pi$ |
| **Policy iteration** | Alternate the two until reaching $\pi_*$ |

**Value iteration** is the efficient special case that merges evaluation and improvement into one step.

All DP methods share two key properties:

1. **Bootstrapping** — updates use estimates of neighboring states, not just raw rewards
2. **Full backups** — every update considers *all* possible next states weighted by their probability (requires a complete model)

---

## Key Equations at a Glance

| Name | Equation |
|---|---|
| Bellman expectation (v) | $v_\pi(s) = \sum_a \pi(a\|s) \sum_{s',r} p(s',r\|s,a)[r + \gamma v_\pi(s')]$ |
| Iterative policy eval update | $V_{k+1}(s) \leftarrow \sum_a \pi(a\|s) \sum_{s',r} p(s',r\|s,a)[r + \gamma V_k(s')]$ |
| Greedy policy improvement | $\pi'(s) = \arg\max_a \sum_{s',r} p(s',r\|s,a)[r + \gamma v_\pi(s')]$ |
| Value iteration update | $V_{k+1}(s) \leftarrow \max_a \sum_{s',r} p(s',r\|s,a)[r + \gamma V_k(s')]$ |

---

## What to Watch For When Reading

- The **gridworld example** (pp. 76–77) is canonical — trace through it manually to build intuition.
- Bellman equations come in two flavors: **expectation form** (evaluating a fixed $\pi$) and **optimality/max form** (finding $\pi_*$).
- Section 4.5 (Asynchronous DP, between 4.4 and 4.6) is short but bridges theory and practice.

---

## Common Conceptual Questions

### Q1: If $\pi$ is greedy w.r.t. its **own** value function $v_\pi$, is it optimal?

**TRUE.** When $\pi$ is greedy w.r.t. $v_\pi$:

$$v_\pi(s) = \max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_\pi(s')\bigr]$$

This is the Bellman optimality equation. Its unique solution is $v_*$, so $v_\pi = v_*$ and $\pi = \pi_*$.

### Q2: If $\pi$ is greedy w.r.t. the **equiprobable random policy's** value function $v_{\pi_{\text{rand}}}$, is it optimal?

**FALSE.** This is only **one step** of policy improvement from a random baseline. The policy improvement theorem guarantees:

$$v_\pi \;\geq\; v_{\pi_{\text{rand}}}$$

That is improvement, not optimality. Generally you'd need many rounds of policy iteration (or value iteration) to reach $\pi_*$.

### Side-by-Side

| Setup | Conclusion |
|---|---|
| Greedy w.r.t. $v_\pi$ (some other policy's value function) | $v_{\pi'} \geq v_\pi$ — *improvement* |
| Greedy w.r.t. $v_*$ | Optimal |
| Greedy w.r.t. its **own** $v_\pi$ | Optimal (fixed-point condition implies $v_\pi = v_*$) |

### The General Rule

> Greedy w.r.t. **somebody else's** value function $\Rightarrow$ improvement over that policy.
> Greedy w.r.t. **its own** value function $\Rightarrow$ optimality (fixed-point condition).
> Greedy w.r.t. $v_*$ $\Rightarrow$ optimality (by definition).

### Why the Distinction Matters

In policy iteration, each greedy improvement step uses the **current** policy's value function — so a single step gives improvement, not optimality. Optimality only appears at termination, when the policy stops changing — at which point the policy is greedy w.r.t. its own $v_\pi$ (Q1's condition).

### Counter-Example Intuition for Q2

In a gridworld with walls or stochastic dynamics, $v_{\pi_{\text{rand}}}$ is a distorted picture of long-run value (it averages over uniform random behavior). Greedy w.r.t. it picks "the best one-step direction assuming I'll act randomly afterward" — which is generally not the same as the optimal action.

---

# Appendix: Warren Powell — Approximate DP for Fleet Management

A real-world application that bridges textbook DP with industrial-scale RL. This is a working summary of Powell's tech talk on truckload trucking ADP.

## A. Single Driver — The "Nomadic Trucker"

This is the warm-up. State = where the driver is. $V(\text{city})$ is a small table.

The driver looks at available loads, picks the one that maximizes:

$$\text{revenue} \;+\; V(\text{destination})$$

Initially all $V$'s are zero, so he goes by raw revenue. Each time he visits a city, he updates that city's value with TD smoothing:

$$V(\text{city}) \;\leftarrow\; (1-\alpha)\, V(\text{city}) \;+\; \alpha\, \hat{V}(\text{city})$$

Powell's example: was in Texas with $V(\text{Texas}) = 450$. Later returns to Texas and observes $\hat{V} = 800$ (an $800$ load). With $\alpha = 0.1$:

$$V(\text{Texas}) \leftarrow 0.9 \cdot 450 + 0.1 \cdot 800 = 485$$

This is **exactly TD(0)** on the location state space.

## B. Why Multi-Driver Is Hard

If you naively define the system state as "where is every driver", you get a combinatorial explosion. Number of joint states:

$$\binom{\text{trucks} + \text{locations} - 1}{\text{locations} - 1}$$

| Trucks | Locations | States |
|---|---|---|
| 1 | 1,000 | 1,000 |
| 5 | 100 | 91 million |
| 5 | 1,000 | 8 trillion |
| 50 | many | intractable |

Real fleets have 500–20,000 trucks. And drivers aren't just defined by location — they have an **attribute vector** (location, equipment type, home domicile, hours of service used, etc.) with up to $10^{20}$ possible values.

## C. The Key Object: $\bar{V}(a)$ Over the Driver Attribute Space

Maintain a value function over the attribute vector $a$ of a *single driver*:

$$\bar{V}(a) = \text{"how valuable is one driver with attribute } a \text{?"}$$

This is the multi-driver analog of $V(\text{city})$ — but $a$ is now a 10–15 dimensional attribute vector, not a city ID.

## D. Step 1 — The Decision Is a Linear Program

At each time $t$, decide who hauls which load. Each (driver $d$, load $j$) pair has value:

$$\underbrace{c_{dj}}_{\text{revenue}} \;+\; \underbrace{\bar{V}(a_d^{\text{after}})}_{\text{downstream value of driver's resulting attribute}}$$

Maximize total value across all assignments, subject to: each driver does one thing, each load is taken at most once. This is a linear program — handled by Gurobi or CPLEX, scales to thousands of drivers.

## E. Step 2 — Marginal Value $\hat{v}_d$ via Leave-One-Out

The marginal value of driver $d$ is defined literally:

$$\hat{v}_d \;=\; (\text{LP objective with all drivers}) \;-\; (\text{LP objective without driver } d)$$

Naively this requires $N$ LP re-solves. **Computational shortcut:** the LP solver gives these as **dual variables for free** — one LP solve produces $\hat{v}_d$ for every driver simultaneously.

## F. Step 3 — TD Smoothing on Driver Attributes

For each driver $d$, take its observed marginal value $\hat{v}_d$ and blend it into $\bar{V}$ at that driver's attribute:

$$\bar{V}(a_d) \;\leftarrow\; (1-\alpha)\, \bar{V}(a_d) \;+\; \alpha\, \hat{v}_d$$

Powell's key sentence:

> *"It's just like what I was doing with the driver in Texas but instead of the value of the driver in Texas, it'll be the marginal value."*

Each driver provides one fresh observation about the value of *its own* attribute vector, and you smooth it in. Same TD update as the nomadic trucker, just lifted from the location space to the attribute space.

## G. Step 4 — Simulate Forward (Monte Carlo)

Sample exogenous noise $W$ (new loads, drivers entering/leaving, travel-time variability), advance to $t+1$, repeat the loop. Across many iterations, $\bar{V}$ converges.

## H. Hierarchical Aggregation — Key Trick for $10^{20}$ States

With an attribute space that large, you'll never visit most attribute vectors. Powell's fix: maintain $\bar{V}$ at **multiple levels of aggregation simultaneously**.

Example aggregation levels:

| Level | Attributes Used | Bucket Count |
|---|---|---|
| 1 | Location only | ~50 |
| 2 | Location + equipment | ~4,000 |
| 3 | Location + equipment + hours-of-service | ~50 million |
| 4 | Full attribute vector | $10^{20}$ |

When you observe $\hat{v}_d$ for driver $d$, **smooth it into $\bar{V}^{(g)}$ at every level $g$** (the driver belongs to one bucket per level).

When the LP needs $\bar{V}(a)$, compute a **weighted sum** across levels:

$$\bar{V}(a) \;=\; \sum_{g} w_g(a)\, \bar{V}^{(g)}(a)$$

Weights $w_g(a)$ are inversely proportional to $\text{bias}^2 + \text{variance}$:

- **Aggregate levels:** lots of data → low variance, but coarse → high bias.
- **Disaggregate levels:** few visits → high variance, but precise → low bias.
- **Early in training:** weight aggregate levels (coarse but reliable).
- **Late in training:** weight disaggregate levels (fine and now reliable).

## I. Hierarchical Aggregation Also Solves Exploration

Recall the nomadic trucker's dilemma: never visited Minnesota → $V(\text{Minnesota}) = 0$ → never visit it.

With hierarchical aggregation, even an unvisited attribute gets a non-zero estimate from the aggregate level (e.g., "drivers in the upper Midwest are worth ~$X$"). So the LP naturally chooses to dispatch into Minnesota when it's competitive — no explicit exploration bonus needed.

In a simulation Powell ran:
- **Pure greedy on disaggregate $\bar{V}$:** driver visited only ~7 cities, kept revisiting them.
- **Hierarchical-aggregate $\bar{V}$:** driver visited everywhere.

## J. Connecting Back to Sutton & Barto Ch. 4

| S&B Concept | Powell's Fleet ADP |
|---|---|
| State $s$ | Driver attribute $a$ |
| Value $v(s)$ | Marginal value $\bar{V}(a)$ |
| TD update | Same: $\bar{V}(a) \leftarrow (1-\alpha)\bar{V}(a) + \alpha \hat{v}$ |
| Bellman backup | Solving the LP with $\bar{V}$ in the objective |
| Curse of dimensionality | $10^{20}$ attribute vectors; solved by hierarchical aggregation |
| Bootstrapping | $\bar{V}(a^{\text{after}})$ in the LP objective |

This is why Ch. 4's "exact DP" gives way to function approximation in real applications — the joint fleet state is intractable, but a clever decomposition (LP duality + per-driver attribute value function + hierarchical aggregation) makes it solvable at scale.
