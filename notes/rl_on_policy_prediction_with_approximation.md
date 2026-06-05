# Reinforcement Learning — On-Policy Prediction with Approximation

> **Context:** Personal learning notes on Chapter 9 of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), pp. 197–209. Section numbers (§9.1–9.12) refer to that textbook. Companion to the [Dynamic Programming](rl_dynamic_programming.md) and [TD learning](rl_td_learning.md) notes.
>
> **Theme:** estimating $v_\pi$ for a *fixed* policy $\pi$ when the value function is a *parameterized function* rather than a table.

---

## 0. The Big Picture — Dropping the Table

The central shift of Part II: the value function is no longer a lookup table but a **parameterized functional form**:

$$\hat{v}(s, w) \approx v_\pi(s), \qquad w \in \mathbb{R}^d, \qquad d \ll \lvert\mathcal{S}\rvert$$

- $w$ can be linear feature weights, neural-net weights, decision-tree splits, etc.
- **Generalization** is the new property: updating one state's estimate moves many others. Powerful (learn about rare states) but harder to manage (can't make all states correct independently).
- **Partial observability comes free:** if $\hat{v}$ can't depend on some aspect of state, that aspect is effectively unobservable. (Can't add *memory* of past observations, though.)
- **Still prediction, still on-policy:** $\pi$ fixed (no max); data + weighting $\mu$ come from following $\pi$. On-policy is **load-bearing for stability** (§9.4, deadly-triad foreshadow).

---

## 1. Value-function Approximation = Supervised Learning (§9.1)

Every update can be written as a single training example $s \mapsto u$ ("shift $\hat{v}(s)$ toward target $u$"):

| Method | Update $s \mapsto u$ |
|---|---|
| Monte Carlo | $S_t \mapsto G_t$ |
| TD(0) | $S_t \mapsto R_{t+1} + \gamma\, \hat{v}(S_{t+1}, w_t)$ |
| n-step TD | $S_t \mapsto G_{t:t+n}$ |
| DP | $`s \mapsto \mathbb{E}_\pi[R_{t+1} + \gamma\, \hat{v}(S_{t+1}, w_t) \mid S_t = s]`$  (arbitrary $s$, not experienced) |

**RL imposes two requirements** that rule out off-the-shelf batch methods:
1. **Online / incremental** — learn from data as it arrives.
2. **Nonstationary targets** — target function changes (control: $\pi$ changes; even fixed $\pi$: bootstrap targets shift as $w$ changes).

---

## 2. The Prediction Objective: VE (§9.2)

Tabular has no explicit objective — values are decoupled and can become exactly correct. Under FA, making one state more accurate makes others less accurate. So declare **which states matter** via a state distribution $\mu(s) \geq 0$, $\sum_s \mu(s) = 1$.

### Mean Squared Value Error

$$\text{VE}(w) = \sum_s \mu(s)\bigl[v_\pi(s) - \hat{v}(s, w)\bigr]^2 \qquad \text{(9.1)}$$

- $\mu$ = **on-policy distribution** = fraction of time spent in each state.
  - **Continuing:** stationary distribution under $\pi$.
  - **Episodic:** from start-state probs $h(s)$ and expected visits $\eta(s)$:

$$\eta(s) = h(s) + \sum_{\bar{s}} \eta(\bar{s}) \sum_a \pi(a \mid \bar{s})\, p(s \mid \bar{s}, a), \qquad \mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')} \qquad \text{(9.2–9.3)}$$

### Two honest caveats
- VE may not be the right objective — we really want a better *policy*. But no clearly better one is known.
- **Convergence:** linear → global optimum reachable; nonlinear → local optimum at best; some methods can diverge ($\text{VE}\to\infty$).

---

## 3. Stochastic-Gradient and Semi-Gradient Methods (§9.3)

### SGD with true value known

$$w_{t+1} = w_t + \alpha\bigl[v_\pi(S_t) - \hat{v}(S_t, w_t)\bigr]\, \nabla \hat{v}(S_t, w_t) \qquad \text{(9.5)}$$

Small steps to balance errors across states; local-optimum convergence under usual step-size conditions (2.7).

### General SGD with target $U_t$

$$w_{t+1} = w_t + \alpha\bigl[U_t - \hat{v}(S_t, w_t)\bigr]\, \nabla \hat{v}(S_t, w_t) \qquad \text{(9.7)}$$

Converges (local optimum) **iff $U_t$ is unbiased**: $`\mathbb{E}[U_t \mid S_t = s] = v_\pi(s)`$.

### True-gradient vs. semi-gradient

| | Target $U_t$ | Depends on $w$? | True gradient? | Guarantee |
|---|---|---|---|---|
| **Gradient MC** | $G_t$ | **No** — unbiased | **Yes** | local optimum (global if linear) |
| **Semi-gradient TD(0)** | $R_{t+1} + \gamma\, \hat{v}(S_{t+1}, w)$ | **Yes** — biased | **No** | reliable in important cases (linear) |
| Semi-gradient DP | $`\sum_a \pi(a \mid s)\sum p[r + \gamma\, \hat{v}(s', w)]`$ | **Yes** | **No** | — |

> **Why "semi-gradient":** the bootstrapped target *itself* contains $w$. The SGD derivation assumes the target is independent of $w$; bootstrapping ignores its effect on the target → only part of the gradient. Same bootstrapping that gave TD its speed in the [TD learning notes](rl_td_learning.md) — here the cost shows up as breaking the gradient-descent guarantee.

**Why use it anyway?** Faster learning + online (works on continuing tasks, no waiting for episode end).

### Pseudocode

**Gradient Monte Carlo** (true gradient):
```
Loop per episode:
  generate S0, A0, R1, …, RT, ST using π
  for t = 0..T−1:
    w ← w + α[Gₜ − v̂(Sₜ, w)] ∇v̂(Sₜ, w)
```

**Semi-gradient TD(0)** (bootstraps, online):
```
Loop per episode:
  init S
  for each step:
    A ~ π(·|S); take A, observe R, S′
    w ← w + α[R + γ v̂(S′, w) − v̂(S, w)] ∇v̂(S, w)
    S ← S′
  until S terminal     (v̂(terminal, ·) = 0)
```

### State aggregation
Group states; one weight per group. Special case of SGD where $\nabla \hat{v} = 1$ on the active group, 0 elsewhere. **Example 9.1 — 1000-state Random Walk:** Gradient MC + 10 groups → staircase near the VE optimum; $\mu$ skews each group toward its more-visited member states.

---

## 4. Linear Methods (§9.4)

The theory sweet-spot: $\hat{v}$ linear in $w$ via feature vector $x(s)$:

$$\hat{v}(s, w) = w^\top x(s) = \sum_i w_i\, x_i(s) \qquad \text{(9.8)}$$

Gradient is just the features: $\nabla \hat{v}(s, w) = x(s)$, so the update is $w_{t+1} = w_t + \alpha[U_t - \hat{v}(S_t, w_t)]\, x(S_t)$.

**Why linear:** one global optimum (no bad local minima) → any local-convergence guarantee becomes global. **Gradient MC** under linear FA converges to the **VE global optimum**.

### The TD fixed point

Linear semi-gradient TD(0) converges to a **different** point. Continuing-case expected update:

$$\mathbb{E}[w_{t+1} \mid w_t] = w_t + \alpha\,(b - A w_t), \qquad b = \mathbb{E}[R_{t+1}\, x_t], \quad A = \mathbb{E}\bigl[x_t (x_t - \gamma\, x_{t+1})^\top\bigr] \qquad \text{(9.10–9.11)}$$

$$w_{\text{TD}} = A^{-1} b \qquad \text{(9.12)}$$

**Stability:** $\mathbb{E}[w_{t+1} \mid w_t] = (I-\alpha A)w_t + \alpha b$ — only $A$ matters. Stable iff $A$ is positive definite. For linear TD(0), $A = X^\top D(I-\gamma P)X$; $D(I-\gamma P)$ is PD because its column sums $= (1-\gamma)\mu^\top > 0$ — **this requires $\mu$ to be the on-policy stationary distribution** ($\mu = P^\top \mu$).

**Quality bound:**

$$\text{VE}(w_{\text{TD}}) \;\leq\; \frac{1}{1 - \gamma}\, \min_w \text{VE}(w) \qquad \text{(9.14)}$$

TD's asymptotic error $\leq \tfrac{1}{1-\gamma}$ × MC's. $\gamma$ near 1 → factor can be large. But TD has far lower variance → often faster.

### Understanding the TD fixed point

- $Aw = b$ is the **Bellman expectation equation projected into feature space**. TD lands at the $\mu$-weighted best representable approximation of $v_\pi$, not $v_\pi$ itself.
- **Tabular reduction (Exercise 9.1):** with one-hot features $x(s) = e_s$, $A = D(I-\gamma P)$ and $b = Dr$. The $D$ cancels: $w_{\text{TD}} = (I-\gamma P)^{-1} r = v_\pi$, exact.
- **Why $\mu$ matters in real FA but not tabular:** with $d \ll \lvert\mathcal{S}\rvert$, $D$ doesn't cancel — it stays inside the projection. This single fact explains why $w_{\text{TD}} \neq v_\pi$ in general, why $\mu$ determines both answer and stability, and why tabular is the zero-error corner of all of Ch. 9.

### ⚠ On-policy distribution is load-bearing (deadly-triad foreshadow)

These guarantees require updating states under the on-policy distribution. With a different distribution, **bootstrapping + FA can diverge to $\infty$** (Ch. 11). Dangerous combo: **bootstrapping + FA + off-policy** = the "deadly triad."

### Example 9.2 — Bootstrapping on 1000-state Random Walk
State aggregation $\subset$ linear FA. Semi-gradient TD's asymptotic values are worse than MC's (consistent with 9.14). But **n-step semi-gradient TD** reproduces Ch. 7's pattern: **intermediate $n$ is best**.

### n-step semi-gradient TD

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n\, \hat{v}(S_{t+n}, w_{t+n-1}) \qquad \text{(9.16)}$$

$$w_{t+n} = w_{t+n-1} + \alpha\bigl[G_{t:t+n} - \hat{v}(S_t, w_{t+n-1})\bigr]\, \nabla \hat{v}(S_t, w_{t+n-1}) \qquad \text{(9.15)}$$

---

## 5. Feature Construction for Linear Methods (§9.5)

Linear FA is only as good as $x(s)$. Feature choice = main way to inject prior knowledge and decide **along which dimensions generalization happens**.

**Motivation:** $\hat{v} = w^\top x(s)$ can't capture interactions like "feature $i$ is good only when feature $j$ is absent" unless they're built into the features. Pole-balancing example: angular velocity good at low angle, bad at high — needs **conjunction features**.

### 9.5.1 Polynomials
Order-$n$ basis: $x_i(s) = \prod_j s_j^{c_{ij}}$, exponents in $\{0,\ldots,n\}$ → $(n+1)^k$ features. Grows exponentially in $k$. **Not recommended for online RL** (loses to Fourier, Fig. 9.5).

### 9.5.2 Fourier Basis
Order-$n$ cosine basis. $k$-D: $x_i(s) = \cos(\pi\, s^\top c^i)$, $c^i \in \{0,\ldots,n\}^k$. A 0 entry = constant along that dim; two nonzero entries = an interaction. Suggested step size $\alpha_i = \alpha / \lVert c^i \rVert$.
- **Pros:** often beats polynomials/RBFs; easy priors.
- **Cons:** ringing at discontinuities; features are *global* → poor at local properties.

### 9.5.3 Coarse Coding
Overlapping receptive fields (e.g. circles); binary feature = 1 if state inside. Training one state updates all overlapping fields → generalizes ∝ shared fields. **Key insight (Example 9.3):** field size/shape controls *initial* generalization; **acuity** (finest final discrimination) is set by the **total number of features**, not field size.

### 9.5.4 Tile Coding — the practical workhorse
Coarse coding via multiple overlapping grid tilings, each offset by a fraction of a tile width.
- Every state activates exactly one tile per tiling → **constant active count = # tilings**.
- **Easy step size:** $\alpha = 1/n$ ($n$ tilings) = one-trial learning; $\alpha = 1/(10n)$ = move 1/10 toward target.
- **Cheap compute:** binary → look up $n$ tile indices and sum $n$ weights.
- **Asymmetric offsets** (first odd integers $1,3,\ldots,2k-1$) avoid diagonal artifacts.
- **Hashing** pseudo-randomly collapses a big tiling → memory matches task demand, not exponential in $k$.

### 9.5.5 Radial Basis Functions
Continuous coarse coding via Gaussian response:

$$x_i(s) = \exp\left(-\frac{\lVert s - c_i \rVert^2}{2\sigma_i^2}\right)$$

Smooth + differentiable, but rarely worth it — nonlinear RBF networks often can't beat tile coding while adding cost.

### Cross-cutting theme

| Knob | Controls |
|---|---|
| Feature type | what functions/interactions are representable |
| Receptive-field size/shape (or freq for Fourier) | *initial* generalization direction & breadth |
| Number of features/tilings | asymptotic **acuity** |
| Offsets / conjunctions | avoid artifacts; learn specific combinations |

Local features (coarse/tile/RBF) excel at local structure; global features (poly/Fourier) at smooth global structure.

---

## 6. Selecting Step-Size Parameters (§9.6)

Theory unhelpful: stochastic-approximation conditions converge slowly; $\alpha_t = 1/t$ (sample averages in tabular MC) is wrong for TD, nonstationary problems, or FA.

**Intuition:** $\alpha = 1/\tau$ → estimate approaches mean of targets after $\sim \tau$ experiences.

**Linear FA rule of thumb** (learn over $\sim \tau$ experiences):

$$\alpha = \bigl(\tau\, \mathbb{E}[x^\top x]\bigr)^{-1} \qquad \text{(9.19)}$$

For tile coding with $n$ active features, $x^\top x = n$, recovering $\alpha = 1/(\tau n)$.

---

## 7. Nonlinear Function Approximation: ANNs (§9.7)

ANNs are the standard nonlinear approximator → **deep RL**.

- **Semi-linear units:** weighted sum → nonlinear activation (sigmoid, ReLU, step). Nonlinearity essential (stacked linear = single linear).
- **Universal approximation:** one hidden layer w/ enough sigmoids ≈ any continuous function on a compact set; but **depth** (hierarchical features) is what makes complex AI tractable.
- **Training:** SGD + backpropagation. RL objective: TD error (value learning) or expected reward (policy gradient).
- **Why deep is hard + fixes:**
  - Overfitting → regularization, weight sharing, dropout.
  - Vanishing/exploding gradients → batch norm, residual learning.
- **CNNs** (shared weights + local receptive fields) underlie Atari/Go.
- **Caveat:** RL theory is mostly tabular/linear; nonlinear successes are largely empirical.

---

## 8. Least-Squares TD — LSTD (§9.8)

Since linear semi-gradient TD just converges to $w_{\text{TD}} = A^{-1} b$, why iterate? LSTD estimates $A$ and $b$ directly and solves once.

$$\hat{A}_t = \sum_{k=0}^{t-1} x_k (x_k - \gamma\, x_{k+1})^\top + \varepsilon I, \qquad \hat{b}_t = \sum_{k=0}^{t-1} R_{k+1}\, x_k, \qquad w_t = \hat{A}_t^{-1}\, \hat{b}_t \qquad \text{(9.20–9.21)}$$

- Most data-efficient linear TD(0); no $\alpha$. Inverse maintained via Sherman–Morrison → $O(d^2)$ per step (vs. $O(d)$ for semi-gradient TD).
- **Costs:** $O(d^2)$ prohibitive for large $d$; needs $\varepsilon$; never forgets (no $\alpha$) → bad when $\pi$ changes under GPI.

---

## 9. Memory-Based Function Approximation (§9.9)

**Nonparametric / lazy:** store training examples $s \mapsto g$; on query, retrieve relevant ones and compute on the fly. No fixed form.

- **Local methods:** nearest neighbor, weighted average, locally weighted regression.
- **Why suited to RL:** focuses on **states actually visited**; immediate local effect; memory $\propto k$ per example, linear in #examples (no exponential blowup).
- **Challenge:** fast retrieval → k-d trees, specialized hardware.

---

## 10. Kernel-Based Function Approximation (§9.10)

Distance-based weights of memory methods come from a **kernel** $k(s, s')$:

$$\hat{v}(s, \mathcal{D}) = \sum_{s' \in \mathcal{D}} k(s, s')\, g(s')$$

- **Kernel trick:** any linear method with features $x(s)$ ≡ kernel regression with $k(s,s') = x(s)^\top x(s')$. Lets you work in (possibly infinite) feature spaces at the cost of stored examples.
- Tile coding's generalization patterns *are* implicit kernels.

---

## 11. Interest and Emphasis (§9.11)

Sometimes we care about some states more (e.g. early states in a discounted episode). Two per-step scalars:
- **Interest $I_t \geq 0$:** how much we care about state $t$. Reshapes $\mu$ in VE.
- **Emphasis $M_t \geq 0$:** multiplies the update.

$$w_{t+n} = w_{t+n-1} + \alpha\, M_t\, \bigl[G_{t:t+n} - \hat{v}(S_t, \cdot)\bigr]\, \nabla \hat{v}(S_t, \cdot), \qquad M_t = I_t + \gamma^n\, M_{t-n} \qquad \text{(9.25–9.26)}$$

**Example 9.4** (4-state MRP, shared weights, interest only in leftmost state): plain methods compromise (3.5); interest/emphasis methods value the state of interest exactly (4).

---

## 12. Summary (§9.12)

- Generalization is essential at scale → each update = supervised-learning example; parametric FA with weights $w$ is most suitable.
- $\text{VE}(w)$ ranks approximations under the on-policy $\mu$.
- Workhorse: **n-step semi-gradient TD**, with gradient MC ($n=\infty$) and TD(0) ($n=1$) as endpoints. Semi-gradient methods aren't true gradient but work in the linear case.
- **Features matter most:** avoid polynomials online; prefer Fourier or coarse/tile coding.
- **LSTD** = most data-efficient linear TD but $O(d^2)$. Nonlinear → ANNs + backprop.
- **Some bootstrapping ($n < \infty$) is usually best** — same Ch. 6/7 lesson.

---

## Quick Reference

| Concept | Key formula / fact |
|---|---|
| Approximate value | $\hat{v}(s, w) = w^\top x(s)$ (linear); $\nabla \hat{v} = x(s)$ |
| Objective | $\text{VE}(w) = \sum_s \mu(s)[v_\pi(s) - \hat{v}(s, w)]^2$ |
| General SGD update | $w \leftarrow w + \alpha[U_t - \hat{v}(S_t, w)]\, \nabla \hat{v}(S_t, w)$ |
| Gradient MC | $U_t = G_t$; unbiased → true gradient → VE optimum (linear) |
| Semi-gradient TD | $U_t = R + \gamma \hat{v}'$; biased → semi-gradient → TD fixed point |
| TD fixed point | $w_{\text{TD}} = A^{-1} b$, $A = \mathbb{E}[x_t(x_t - \gamma x_{t+1})^\top]$, $b = \mathbb{E}[R_{t+1} x_t]$ |
| Meaning | Bellman expectation eqn projected in feature space ($Aw = b$) |
| Tabular reduction | one-hot $x(s)$: $D$ cancels → $w_{\text{TD}} = v_\pi$, exact |
| TD error bound | $\text{VE}(w_{\text{TD}}) \leq \tfrac{1}{1-\gamma}\, \min_w \text{VE}(w)$ |
| Stability requires | on-policy distribution (else bootstrapping + FA can diverge) |

---

## Connections to Earlier Chapters

- **Prediction vs. control:** still pure prediction (fixed $\pi$, no max). Control with FA is the next chapter (semi-gradient Sarsa). See [TD learning notes](rl_td_learning.md).
- **Bootstrapping's double identity:** speed/DP-flavor (Ch. 6) ↔ here it makes methods *semi-*gradient (biased, not true SGD).
- **MC vs. TD tradeoff:** VE-optimal but high variance (MC) vs. TD-fixed-point with $\tfrac{1}{1-\gamma}$ bound but lower variance (TD).
- **On-policy distribution** becomes a *stability requirement* → off-policy "deadly triad" of Ch. 11.
