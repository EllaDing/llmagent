# Reinforcement Learning — On-Policy Prediction with Approximation

> **Context:** Personal learning notes on Chapter 9 of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), pp. 197–209. Section numbers (§9.1–9.12) refer to that textbook. Companion to the [Dynamic Programming](rl_dynamic_programming.md) and [TD learning](rl_td_learning.md) notes.
>
> **Theme:** estimating $v_\pi$ for a *fixed* policy $\pi$ when the value function is a *parameterized function* rather than a table.

---

## 0. The Big Picture — Dropping the Table

The central shift of Part II: the value function is no longer a lookup table but a **parameterized functional form**:

$$\hat{v}(s, w) \approx v_\pi(s), \qquad w \in \mathbb{R}^d, \qquad d \ll \lvert\mathcal{S}\rvert$$

- $w$ can be: linear feature weights, neural-net connection weights, decision-tree split points, etc.
- Because $d \ll \lvert\mathcal{S}\rvert$, you cannot store a value per state — and **changing one weight changes the estimate of many states.**

### The defining new property: generalization
Updating one state's estimate **generalizes** to affect many other states. This is:
- **powerful** — learn about states you've barely/never seen,
- but **harder to manage** — you can't make all states correct independently.

### Bonus: partial observability
If $\hat{v}$'s functional form can't depend on some aspect of the state, that aspect is effectively **unobservable** — so all the function-approximation theory applies equally to partially observable problems. (What it *can't* do: add **memory** of past observations.)

### Still prediction, still on-policy
- **Prediction**: $\pi$ is fixed, estimate its value (no improvement step, no max). See [TD learning notes](rl_td_learning.md).
- **On-policy**: the data (and the weighting $\mu$) come from following $\pi$ itself. This on-policy assumption turns out to be **load-bearing for stability** (see §9.4 and the deadly-triad foreshadowing).

---

## 1. Value-function Approximation = Supervised Learning (§9.1)

Every update in the book can be written as a single training example $s \mapsto u$ ("shift $\hat{v}(s)$ toward target $u$"):

| Method | Update $s \mapsto u$ |
|---|---|
| Monte Carlo | $S_t \mapsto G_t$ |
| TD(0) | $S_t \mapsto R_{t+1} + \gamma\, \hat{v}(S_{t+1}, w_t)$ |
| n-step TD | $S_t \mapsto G_{t:t+n}$ |
| DP | $s \mapsto \mathbb{E}_\pi[R_{t+1} + \gamma\, \hat{v}(S_{t+1}, w_t) \| S_t = s]$  (arbitrary $s$, not experienced) |

Each $s \mapsto u$ is fed to a **supervised-learning / function-approximation** method as an input→output example; the function it produces *is* the estimated value function.

**But RL imposes two requirements that rule out off-the-shelf batch methods:**
1. **Online / incremental learning** — must learn from data as it arrives (agent interacting), not multiple passes over a static dataset.
2. **Nonstationary targets** — the target function changes over time:
   - in control (GPI) we chase $q_\pi$ while $\pi$ changes;
   - even with fixed $\pi$, **bootstrapping targets shift as $w$ changes**.

Methods that can't handle nonstationarity are poorly suited to RL.

---

## 2. The Prediction Objective: VE (§9.2)

In the **tabular** case no explicit objective was needed: values were *decoupled* (an update at one state affected no other) and could become *exactly* correct.

With **approximation** this breaks: making one state more accurate makes others **less** accurate. So we must declare **which states we care about** via a state distribution $\mu(s) \geq 0$, $\sum_s \mu(s) = 1$.

### Mean Squared Value Error

$$\text{VE}(w) = \sum_s \mu(s)\bigl[v_\pi(s) - \hat{v}(s, w)\bigr]^2 \qquad \text{(9.1)}$$

- $\sqrt{\text{VE}}$ (root-VE) is the usual plotting measure.
- $\mu$ **= the on-policy distribution** = fraction of time spent in each state.
  - **Continuing tasks:** stationary distribution under $\pi$.
  - **Episodic tasks:** computed from start-state probs $h(s)$ and expected visits $\eta(s)$:

$$\eta(s) = h(s) + \sum_{\bar{s}} \eta(\bar{s}) \sum_a \pi(a \mid \bar{s})\, p(s \mid \bar{s}, a) \qquad \text{(9.2)}$$

$$\mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')} \qquad \text{(9.3)}$$

  (Discounting $\gamma < 1$ is treated as partial termination — insert a $\gamma$ factor in 9.2.)

### Two honest caveats
- **VE may not be the right objective** — our real goal is a *better policy*, and the VE-best value function isn't necessarily the policy-best one. But no clearly better prediction objective is known yet.
- **Convergence is not guaranteed in general:**
  - **linear** approximators → can reach a **global optimum**;
  - **nonlinear** (NN, trees) → at best a **local optimum**;
  - some methods may **diverge** ($\text{VE} \to \infty$).

---

## 3. Stochastic-Gradient and Semi-Gradient Methods (§9.3)

### Stochastic Gradient Descent (SGD)
$\hat{v}(s, w)$ is differentiable in $w$. Adjust $w$ a small step down the squared-error gradient on each example (true value known):

$$w_{t+1} = w_t + \alpha\bigl[v_\pi(S_t) - \hat{v}(S_t, w_t)\bigr]\, \nabla \hat{v}(S_t, w_t) \qquad \text{(9.5)}$$

- **Small** steps (not full corrections) because we want to **balance** errors across states, not zero out one example. Convergence to a local optimum requires $\alpha \downarrow$ per the usual stochastic-approximation conditions (2.7).

### General SGD with an arbitrary target $U_t$
When the true value is unknown, substitute a target $U_t$:

$$w_{t+1} = w_t + \alpha\bigl[U_t - \hat{v}(S_t, w_t)\bigr]\, \nabla \hat{v}(S_t, w_t) \qquad \text{(9.7)}$$

Guaranteed to converge (local optimum) **iff $U_t$ is unbiased**: $\mathbb{E}[U_t \mid S_t = s] = v_\pi(s)$.

### The critical split: true-gradient vs. semi-gradient

| | Target $U_t$ | Depends on $w$? | True gradient? | Guarantee |
|---|---|---|---|---|
| **Gradient MC** | $G_t$ | **No** — unbiased | **Yes** | local optimum (global if linear) |
| **Semi-gradient TD(0)** | $R_{t+1} + \gamma\, \hat{v}(S_{t+1}, w)$ | **Yes** — biased | **No** | reliable only in important cases (e.g. linear) |
| Semi-gradient DP | $\sum_a \pi(a\|s) \sum p[r + \gamma\, \hat{v}(s', w)]$ | **Yes** | **No** | — |

> **Why "semi-gradient":** the bootstrapped target *itself contains $w$*. The derivation 9.4→9.5 assumes the **target is independent of $w$**. Bootstrapping accounts for $w$'s effect on the *estimate* $\hat{v}(S_t, w)$ but **ignores its effect on the target** → it includes only **part** of the gradient.
>
> This is the *same bootstrapping* that gave TD its speed and DP-style behavior in the [TD learning notes](rl_td_learning.md) — here the cost of that DP inheritance becomes visible: it breaks the clean gradient-descent guarantee.

### Why use semi-gradient anyway?
- typically **significantly faster** learning (as in Ch. 6–7);
- **continual & online** — no waiting for episode end → usable on **continuing** tasks, with computational advantages.

### Pseudocode
**Gradient Monte Carlo** (true gradient, unbiased $G_t$):
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
  until S terminal           (v̂(terminal, ·) = 0)
```

### State aggregation
Group states; one weight (one value) per group. Special case of SGD where $\nabla \hat{v} = 1$ for the active group's component, $0$ elsewhere.
- **Example 9.1 — 1000-state Random Walk:** Gradient MC + 10 groups → a **staircase** approximation near the VE global minimum. The state distribution $\mu$ **skews each group's value** toward its more-frequently-visited member states (e.g. leftmost group biased up toward state 100's higher true value).

---

## 4. Linear Methods (§9.4)

The most important special case: $\hat{v}$ is **linear in $w$** via a feature vector $x(s)$:

$$\hat{v}(s, w) = w^\top x(s) = \sum_i w_i\, x_i(s) \qquad \text{(9.8)}$$

- $x(s)$ = feature vector; each $x_i$ is a **basis function**. Choosing $d$ features = choosing a $d$-dim basis.
- Gradient is trivially the features: $\nabla \hat{v}(s, w) = x(s)$, so the update collapses to:

$$w_{t+1} = w_t + \alpha\bigl[U_t - \hat{v}(S_t, w_t)\bigr]\, x(S_t)$$

### Why linear is the theory sweet-spot
- **One optimum** (no bad local optima) ⇒ any local-convergence guarantee becomes a **global** one.
- **Gradient MC** under linear FA → converges to the **VE global optimum** ($\alpha \downarrow$ appropriately).

### The TD fixed point (semi-gradient TD under linear FA)
Linear semi-gradient TD(0) also converges, but to a **different** point that is **not** the VE optimum. Writing the continuing-case expected update:

$$\mathbb{E}[w_{t+1} \mid w_t] = w_t + \alpha\,(b - A w_t) \qquad \text{(9.10)}$$

$$b = \mathbb{E}[R_{t+1}\, x_t] \in \mathbb{R}^d, \qquad A = \mathbb{E}\bigl[x_t (x_t - \gamma\, x_{t+1})^\top\bigr] \in \mathbb{R}^{d \times d} \qquad \text{(9.11)}$$

At convergence $b - A w_{\text{TD}} = 0$, giving the **TD fixed point**:

$$w_{\text{TD}} = A^{-1} b \qquad \text{(9.12)}$$

**Stability / convergence (boxed proof):**
- Rewrite as $\mathbb{E}[w_{t+1} \mid w_t] = (I - \alpha A) w_t + \alpha b$. Only $A$ matters for convergence.
- Stable iff $A$ is positive definite (which also guarantees $A^{-1}$ exists).
- For linear TD(0), $A = X^\top D (I - \gamma P) X$. The key matrix $D(I - \gamma P)$ is positive definite because its **column sums** $= (1 - \gamma)\mu^\top > 0$ — and this relies on $\mu$ being the stationary (on-policy) distribution ($\mu = P^\top \mu$).

**Quality bound at the TD fixed point (continuing case):**

$$\text{VE}(w_{\text{TD}}) \;\leq\; \frac{1}{1 - \gamma}\, \min_w \text{VE}(w) \qquad \text{(9.14)}$$

- TD's asymptotic error $\leq \frac{1}{1-\gamma}$ × the best possible error (the MC limit).
- $\gamma$ near 1 ⇒ this factor can be **large** → real potential loss in asymptotic accuracy.
- **But** TD has **far lower variance** → often faster. Same MC-vs-TD tradeoff as the [TD learning notes](rl_td_learning.md), now in the approximation setting.

### 🔎 Understanding the TD fixed point (deep dive)

**Is this new, or old?** The *fixed point itself* is **new in §9.4** — it can't exist before Ch. 9 because there is no weight vector $w$ to solve for in the tabular chapters. But it is built entirely from the **Chapter 6 TD error**. The bracket in the update (9.9) is literally $\delta_t = R_{t+1} + \gamma\, \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$ with $\hat{v}(S, w) = w^\top x(S)$ swapped in for the tabular $V(S)$. So: same target, new representation.

**Where $A$ and $b$ come from.** Expand the bracket in the update and rearrange:

$$w_{t+1} = w_t + \alpha\,\bigl(R_{t+1}\, x_t - x_t (x_t - \gamma\, x_{t+1})^\top w_t\bigr)$$

Take the steady-state expectation (states $\sim \mu$) ⇒ $\mathbb{E}[w_{t+1} \mid w_t] = w_t + \alpha\,(b - A w_t)$, with $b = \mathbb{E}[R_{t+1}\, x_t]$, $A = \mathbb{E}[x_t (x_t - \gamma\, x_{t+1})^\top]$. The **fixed point is where the expected update stops moving**: $b - A w_{\text{TD}} = 0 \implies w_{\text{TD}} = A^{-1} b$.

**What it *means*: the Bellman expectation equation in feature space.** $A w = b$ is the projected/sampled form of "$\hat{v}$ is self-consistent with $R + \gamma\, \hat{v}(S')$ on average." It's the **same Bellman expectation equation TD(0) always chases** (see [TD learning notes](rl_td_learning.md)) — but since $\hat{v}$ (with $d \ll \lvert\mathcal{S}\rvert$) can't represent $v_\pi$ exactly, TD lands at the **$\mu$-weighted projection** of that solution onto the representable functions, not $v_\pi$ itself. That projection gap is exactly what the $\frac{1}{1-\gamma}$ bound (9.14) measures; it's also why TD's answer differs from Gradient MC's VE-optimum (different criterion: solve $Aw = b$ vs. minimize VE).

**Sanity check — the tabular reduction (Exercise 9.1): $w_{\text{TD}} = v_\pi$ exactly.**
Take one-hot features $x(s) = e_s$ (a 1 in position $s$). Then $d = \lvert\mathcal{S}\rvert$, $\hat{v}(s, w) = w^\top e_s = w_s$ — the weight vector *is* the value table. Computing $A$ and $b$:

$$b = \sum_s \mu(s)\, r(s)\, e_s = D r \qquad \bigl(r(s) = \mathbb{E}[R_{t+1} \mid s]\bigr)$$

$$A = \sum_s \mu(s)\, e_s\bigl(e_s - \gamma \sum_{s'} p(s' \mid s)\, e_{s'}\bigr)^\top = D(I - \gamma P)$$

where $D = \mathrm{diag}(\mu(s))$, $P$ is the on-policy transition matrix. (This is exactly the $D(I - \gamma P)$ "key matrix" from the convergence box.) Then:

$$w_{\text{TD}} = A^{-1} b = [D(I - \gamma P)]^{-1} D r = (I - \gamma P)^{-1} D^{-1} D\, r = (I - \gamma P)^{-1} r$$

**The $\mu$ cancels** ($D^{-1} D = I$), leaving $(I - \gamma P)\, w_{\text{TD}} = r$, i.e. component-wise

$$w_{\text{TD}}(s) = r(s) + \gamma \sum_{s'} p(s' \mid s)\, w_{\text{TD}}(s')$$

— **exactly the Bellman expectation equation** $v_\pi = r + \gamma P v_\pi$. Its unique solution is the true value function, so $w_{\text{TD}} = v_\pi$: in the tabular case the TD fixed point *is* $v_\pi$, with zero error and no $\frac{1}{1-\gamma}$ gap.

**The key takeaway — why $\mu$ matters in general but not here.** In tabular FA the $D$ from $b = D r$ cancels the $D$ in $A = D(I - \gamma P)$, so the distribution is irrelevant and the answer is exact. With genuine FA ($d \ll \lvert\mathcal{S}\rvert$), $D$ does **not** cancel — it stays inside the projection. That single fact explains three things at once:
- **why** $w_{\text{TD}} \neq v_\pi$ in general (you get the $\mu$-weighted best representable approximation),
- **why** the on-policy distribution determines *both* the answer and stability ($A$'s positive-definiteness needed $\mu = P^\top \mu$),
- **why** tabular methods are the zero-error corner of all of Ch. 9 — every gap from $v_\pi$ in real FA comes from $d \ll \lvert\mathcal{S}\rvert$ forcing a projection.

### ⚠ The on-policy distribution is load-bearing (deadly-triad foreshadow)
These convergence guarantees **require updating states under the on-policy distribution**. With a *different* update distribution, **bootstrapping + function approximation can diverge to $\infty$** (Chapter 11). The dangerous combination is **bootstrapping + function approximation + off-policy** — the "deadly triad."

Related results: linear **semi-gradient DP** (on-policy distribution) also converges to the TD fixed point; one-step **semi-gradient Sarsa(0)** (next chapter) has an analogous fixed point and bound.

### Example 9.2 — Bootstrapping on the 1000-state Random Walk
- State aggregation $\subset$ linear FA. Semi-gradient TD's asymptotic values are **worse** than MC's (consistent with the 9.14 bound).
- But **n-step semi-gradient TD** (20 groups of 50) reproduces the tabular Ch. 7 pattern (Fig. 7.2): **intermediate $n$ is best** — TD generalizes/contains MC, retaining the learning-rate advantage.

### n-step semi-gradient TD (natural extension of Ch. 7)

$$w_{t+n} = w_{t+n-1} + \alpha\bigl[G_{t:t+n} - \hat{v}(S_t, w_{t+n-1})\bigr]\, \nabla \hat{v}(S_t, w_{t+n-1}) \qquad \text{(9.15)}$$

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n\, \hat{v}(S_{t+n}, w_{t+n-1}) \qquad \text{(9.16)}$$

> **Exercise 9.1:** tabular methods are a special case of linear FA — the feature vectors are the **one-hot (unit) vectors**, one component per state. (State aggregation generalizes this to one-hot over groups.)

---

## 5. Feature Construction for Linear Methods (§9.5)

Linear methods are only as good as the feature vector $x(s)$. Feature choice = the main way to inject **prior domain knowledge** and decide **along which dimensions generalization happens**.

### The motivation: the linear model can't represent interactions
$\hat{v} = w^\top x(s)$ cannot capture that "feature $i$ is good only in the absence of feature $j$" unless you build the interaction into the features.
- **Pole-balancing example:** high angular velocity is *good* at low angle (pole righting itself) but *bad* at high angle (about to fall). Features coding angle and angular velocity *separately* can't express this → need **combination/conjunction features**.

### 9.5.1 Polynomials
- State as numbers $s = (s_1, \ldots, s_k)$; features like $(1, s_1, s_2, s_1 s_2, \ldots)$. The $1$ → affine functions; products like $s_1 s_2$ → interactions.
- **Order- $n$ polynomial basis** (9.17): $x_i(s) = \prod_j s_j^{c_{ij}}$, exponents in $\{0, \ldots, n\}$ → $(n+1)^k$ features.
- Grows exponentially in $k$ → must subset. **Not recommended for online RL** (loses to Fourier, Fig. 9.5).

### 9.5.2 Fourier Basis
- Order- $n$ **Fourier cosine basis**. 1-D: $x_i(s) = \cos(i \pi s)$, $s \in [0, 1]$.
- $k$-D (9.18): $x_i(s) = \cos(\pi\, s^\top c^i)$, integer vector $c^i \in \{0, \ldots, n\}^k$ sets **frequency per dimension**; a 0 entry = constant along that dim; two nonzero entries = an **interaction** (ratio = direction). → $(n+1)^k$ features.
- Suggested per-feature step size $\alpha_i = \alpha / \lVert c^i \rVert$.
- **Pros:** often beats polynomials/RBFs; easy priors (pick $c^i$ for suspected interactions; cap frequency to filter noise).
- **Cons:** **ringing** at discontinuities; features are **global** (nonzero everywhere) → poor at **local** properties.

### 9.5.3 Coarse Coding
- Overlapping **receptive fields** (e.g. circles); binary feature = 1 if state inside field, else 0.
- Training one state updates all overlapping fields → **generalizes** to nearby states $\propto$ shared fields.
- **Key insight (Example 9.3):** receptive-field **size/shape** controls *initial generalization* (broad vs. bumpy); **acuity** (finest final discrimination) is set by the **total number of features**, NOT field size. Width strongly affects early learning, barely affects asymptotic accuracy.

### 9.5.4 Tile Coding — the practical workhorse
- Coarse coding via **multiple overlapping grid tilings**, each offset by a fraction of a tile width. (One tiling alone = state aggregation.)
- Every state activates **exactly one tile per tiling** → #active features = #tilings, **constant for all states**.
- **Advantages:**
  - **Easy step size:** constant active count → $\alpha = 1/n$ ($n$ tilings) = exact one-trial learning; $\alpha = 1/(10n)$ = move 1/10 toward target.
  - **Cheap compute:** binary → look up $n$ active tile indices and sum $n$ weights (no $d$ multiplications).
  - **Tunable generalization via tile shape:** square = even; stripe = along one axis; **conjunctive** = specific combinations.
- **Offsets matter:** uniform $(1, 1)$ → diagonal artifacts; **asymmetric** offsets (first odd integers $1, 3, 5, \ldots, 2k - 1$, $n$ a power of 2 $\geq 4k$) → homogeneous, well-centered generalization.
- **Hashing:** pseudo-randomly collapse a big tiling into fewer tiles → large memory savings, little performance loss → **breaks the curse of dimensionality** (memory matches task demand, not exponential in $k$).

### 9.5.5 Radial Basis Functions (RBFs)
- Continuous-valued generalization of coarse coding: Gaussian response on distance to center $c_i$, width $\sigma_i$:

$$x_i(s) = \exp\left(-\frac{\lVert s - c_i \rVert^2}{2\sigma_i^2}\right)$$

- **Pro:** smooth, differentiable. **Reality check:** rarely worth it — nonlinear RBF networks (learning centers/widths) often **can't beat tile coding** while adding computation and tuning burden.

### Cross-cutting theme

| Knob | Controls |
|---|---|
| Feature **type** (poly/Fourier/coarse/tile/RBF) | what functions/interactions are representable |
| Receptive-field **size/shape** (or freq for Fourier) | *initial* generalization direction & breadth |
| **Number** of features/tilings | asymptotic **acuity** (finest discrimination) |
| **Offsets / conjunctions** | avoid artifacts; learn specific combinations |

Local features (coarse/tile/RBF) excel at local structure; global features (poly/Fourier) capture smooth global structure but struggle with locality/discontinuities.

---

## 6. Selecting Step-Size Parameters Manually (§9.6)

Theory is little help: the stochastic-approximation conditions (2.7) guarantee convergence but learn too slowly, and $\alpha_t = 1/t$ (which gives sample averages in tabular MC) is **wrong for TD, nonstationary problems, or any FA**.

**Intuition from the tabular case:** $\alpha = 1/\tau$ → a state's estimate approaches the mean of its targets after about $\tau$ experiences (most recent targets weighted most).

**Linear FA rule of thumb** — to learn over $\sim \tau$ experiences with substantially the same feature vector:

$$\alpha = \bigl(\tau\, \mathbb{E}[x^\top x]\bigr)^{-1} \qquad \text{(9.19)}$$

Works best when feature-vector lengths don't vary much (ideally $x^\top x$ constant). For **tile coding** with $n$ active features per step, $x^\top x = n$, so this recovers the clean $\alpha = 1/(\tau n)$ rule from §9.5.4.

---

## 7. Nonlinear Function Approximation: ANNs (§9.7)

ANNs are the standard **nonlinear** approximator → **deep reinforcement learning**.

- **Semi-linear units:** weighted sum → nonlinear **activation** (sigmoid/logistic, ReLU $\max(0, x)$, step). **Nonlinearity is essential** — stacked linear layers collapse to a single linear map.
- **Universal approximation:** one hidden layer with enough sigmoid units approximates any continuous function on a compact set. But **depth** (hierarchical features) is what makes complex AI functions tractable.
- **Training = SGD via backpropagation** (forward pass computes activations; backward pass computes $\partial \text{objective} / \partial \text{weight}$). In RL the objective can be a **TD error** (value learning) or **expected reward** (policy gradient).
- **Why deep is hard + the fixes:**
  - **Overfitting** (many weights) → cross-validation, **regularization**, **weight sharing**, **dropout** (train random "thinned" subnets; approximate their average at test).
  - **Vanishing/exploding gradients** → unsupervised layer-wise pretraining (deep belief nets), **batch normalization**, **deep residual learning** (skip connections learn a residual, no extra weights).
- **Convolutional networks:** feature maps with shared weights + local receptive fields → spatial-invariance via subsampling; trainable by plain backprop. Underpin the Atari/Go results (Ch. 16).
- **Caveat:** RL *theory* is mostly tabular/linear; the big nonlinear successes are largely empirical.

---

## 8. Least-Squares TD — LSTD (§9.8)

Since linear semi-gradient TD just converges to $w_{\text{TD}} = A^{-1} b$, why iterate? **LSTD estimates $A$ and $b$ directly and solves once.**

$$\hat{A}_t = \sum_{k=0}^{t-1} x_k (x_k - \gamma\, x_{k+1})^\top + \varepsilon I, \qquad \hat{b}_t = \sum_{k=0}^{t-1} R_{k+1}\, x_k, \qquad w_t = \hat{A}_t^{-1}\, \hat{b}_t \qquad \text{(9.20–9.21)}$$

- $\varepsilon I$ keeps $\hat{A}$ invertible; the $t$ factors in $\hat{A}$ and $\hat{b}$ cancel.
- **Most data-efficient** linear TD(0) — no step-size $\alpha$. Inverse maintained incrementally via **Sherman–Morrison** (9.22) → $O(d^2)$ **memory & compute/step** (vs. $O(d)$ for semi-gradient TD; naïve inverse would be $O(d^3)$).
- **Costs/caveats:**
  - $O(d^2)$ is prohibitive for large $d$.
  - Still needs $\varepsilon$ (too small → unstable inverses; too large → slow).
  - **Never forgets** (no $\alpha$) → bad when $\pi$ changes under GPI; control use needs an added forgetting mechanism, mooting the "no step size" perk.

---

## 9. Memory-Based Function Approximation (§9.9)

**Nonparametric / lazy learning:** store training examples $s \mapsto g$; on a query, retrieve relevant ones and compute an estimate on the fly. No fixed functional form — the approximation is shaped by the data itself and improves as examples accumulate.

- **Local methods:** retrieve nearest examples by distance. Variants: **nearest neighbor**, **weighted average** (weights $\downarrow$ with distance), **locally weighted regression** (fit a local surface, evaluate, discard).
- **Why suited to RL:** focuses approximation on **states actually visited in trajectories** (no need to approximate unreachable regions); experience has **immediate local effect**; addresses **curse of dimensionality** (memory $\propto k$ per example, linear in #examples, nothing exponential).
- **Challenge:** fast retrieval as memory grows → **k-d trees** and special hardware accelerate nearest-neighbor search.

---

## 10. Kernel-Based Function Approximation (§9.10)

The weights memory-based methods assign by distance come from a **kernel** $k(s, s')$ = strength of generalization from $s'$ to $s$.

**Kernel regression** (9.23):

$$\hat{v}(s, \mathcal{D}) = \sum_{s' \in \mathcal{D}} k(s, s')\, g(s')$$

- **The kernel trick:** any linear parametric method with features $x(s)$ is equivalent to kernel regression with $k(s, s') = x(s)^\top x(s')$ (9.24). So you can **design the kernel directly**, skipping explicit features — often a compact form evaluable **without ever computing in the $d$-dim feature space**. Lets you work in an expansive (even infinite) feature space at the cost of only the stored examples.
- Tile coding's generalization patterns (Fig. 9.11) *are* implicit kernels; any linear FA generalizes according to some kernel.

---

## 11. Interest and Emphasis (§9.11)

Treating all states equally is what gave the on-policy-distribution stability guarantees — but sometimes we **care about some states more** (e.g. early states in a discounted episode; good actions over bad ones). Limited FA resources are better spent where they matter.

Two new per-step scalars:
- **Interest $I_t$** ($\geq 0$): how much we care about accurately valuing state at time $t$. Reshapes $\mu$ in VE (states weighted by interest). Can be set any causal way.
- **Emphasis $M_t$** ($\geq 0$): multiplies the update, emphasizing/de-emphasizing learning at $t$.

$$w_{t+n} = w_{t+n-1} + \alpha\, M_t\, \bigl[G_{t:t+n} - \hat{v}(S_t, \cdot)\bigr]\, \nabla \hat{v}(S_t, \cdot) \qquad \text{(9.25)}$$

$$M_t = I_t + \gamma^n\, M_{t-n} \qquad \text{(9.26)}$$

**Example 9.4** (4-state MRP, $w$ shared across state pairs, interest only in leftmost state): plain methods converge to a *compromise* value (3.5) across the tied states; **interest/emphasis methods value the state-of-interest exactly** (4), spending no updates on states we don't care about.

---

## 12. Summary (§9.12)

- Generalization is essential for RL at scale → treat each update as a **supervised-learning training example**; parametric FA with weight vector $w$ is most suitable.
- $\text{VE}(w)$ ranks approximations under the on-policy distribution $\mu$.
- Workhorse algorithm: **n-step semi-gradient TD**, with **gradient MC ($n = \infty$)** and **semi-gradient TD(0) ($n = 1$)** as endpoints. Semi-gradient (bootstrapping) methods are **not true gradient** → can't lean on classical SGD results, but work well in the **linear** case.
- **Features matter most:** avoid polynomials for online RL; prefer **Fourier basis** or **coarse/tile coding** (tile coding = efficient + flexible); RBFs for smooth 1–2D tasks.
- **LSTD** = most data-efficient linear TD but $O(d^2)$; everything else is $O(d)$. Nonlinear → **ANNs + backprop** (deep RL).
- **Bound (all $n$):** linear semi-gradient $n$-step TD converges within a factor of the optimal VE; the bound **tightens as $n \to \infty$** (toward MC's optimum) but very high $n$ learns slowly → **some bootstrapping ($n < \infty$) is usually best** — the same Ch. 6/7 lesson.

---

## Quick Reference

| Concept | Key formula / fact |
|---|---|
| Approximate value | $\hat{v}(s, w) = w^\top x(s)$ (linear); $\nabla \hat{v} = x(s)$ |
| Objective | $\text{VE}(w) = \sum_s \mu(s)[v_\pi(s) - \hat{v}(s, w)]^2$ |
| General SGD update | $w \leftarrow w + \alpha[U_t - \hat{v}(S_t, w)]\, \nabla \hat{v}(S_t, w)$ |
| Gradient MC ($U_t = G_t$) | unbiased → true gradient → VE optimum (linear) |
| Semi-gradient TD ($U_t = R + \gamma \hat{v}'$) | biased (target has $w$) → semi-gradient → TD fixed point |
| TD fixed point | $w_{\text{TD}} = A^{-1} b$, $A = \mathbb{E}[x_t(x_t - \gamma x_{t+1})^\top]$, $b = \mathbb{E}[R_{t+1} x_t]$ |
| TD fixed point = | the Bellman expectation eqn solved/projected in feature space ($Aw = b$) |
| Tabular reduction | one-hot $x(s)$: $A = D(I - \gamma P)$, $b = Dr$ → $w_{\text{TD}} = (I - \gamma P)^{-1} r = v_\pi$ ($\mu$ cancels, exact) |
| General FA | $D$ doesn't cancel → $w_{\text{TD}} \neq v_\pi$ ($\mu$-weighted projection); $\mu$ sets answer + stability |
| TD error bound | $\text{VE}(w_{\text{TD}}) \leq \frac{1}{1-\gamma}\, \min_w \text{VE}(w)$ |
| Stability requires | on-policy distribution (else bootstrapping + FA can diverge) |

---

## Connections to Earlier Chapters

- **Prediction vs. control:** still pure prediction (fixed $\pi$, no max). Control with approximation is the *next* chapter (semi-gradient Sarsa). See [TD learning notes](rl_td_learning.md).
- **Bootstrapping's double identity:** speed/DP-flavor (Ch. 6) ↔ here it makes methods *semi*-gradient (biased, not true SGD).
- **MC vs. TD tradeoff:** unbiased VE-optimal but high-variance (MC) vs. biased TD-fixed-point with $\frac{1}{1-\gamma}$ bound but low-variance/faster (TD).
- **Sampling-based RL (Ch. 8):** these are all sample updates; DP here is the expected-update counterpart.
- **On-policy distribution** becomes a *stability requirement*, setting up the off-policy "deadly triad" of Chapter 11.
