# Reinforcement Learning — Policy Gradient Methods

> **Context:** Personal learning notes on Chapter 13 of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), pp. 321–336. Section numbers (§13.1–13.7) refer to that textbook. Builds on the [On-Policy Control with Approximation notes](rl_on_policy_control_with_approximation_notes.md) (Ch. 10), the [TD Learning notes](rl_td_learning.md) (Ch. 6), and the [On-Policy Prediction notes](rl_on_policy_prediction_with_approximation.md) (Ch. 9). This is the first method that **parameterizes the policy itself** rather than deriving it from learned values.

---

## 0. The Big Picture — a genuinely new kind of method

Everything before Ch. 13 was an **action-value method**: learn `$`q(s,a)`$`, then *derive* a policy (ε-greedy, etc.). The policy had no existence apart from the values.

Policy gradient methods **parameterize the policy directly**:

$$\pi(a \mid s, \theta) = \Pr\{A_t = a \mid S_t = s, \theta\}, \qquad \theta \in \mathbb{R}^{d'}$$

- A value function may still be learned to *help train* `$`\theta`$`, but it is **not needed to select actions**.
- Optimize a scalar performance measure `$`J(\theta)`$` by **stochastic gradient ascent** (Eq. 13.1):

$$\theta_{t+1} = \theta_t + \alpha\, \widehat{\nabla J(\theta_t)}$$

- **Actor–critic** = learns both a policy (the **actor**) and a value function (the **critic**, used for *bootstrapping*).
- Episodic objective `$`J(\theta) = v_{\pi_\theta}(s_0)`$`; continuing objective `$`J(\theta) = r(\pi)`$` (average reward, §10.3). Both reduce to the **same** gradient expression.

> The single-state bandit version of all this is the **gradient-bandit** algorithm of §2.8 — a good warm-up.

---

## 1. Policy Approximation and its Advantages (§13.1)

**Requirement:** `$`\pi(a \mid s, \theta)`$` differentiable in `$`\theta`$` (so `$`\nabla\pi`$` exists), and kept **stochastic** (`$`\pi \in (0,1)`$`) for exploration.

**Soft-max in action preferences** (discrete actions, Eq. 13.2):

$$\pi(a \mid s, \theta) = \frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}}$$

Preferences `$`h(s,a,\theta)`$` can be linear `$`\theta^\top x(s,a)`$` (Eq. 13.3) or a deep net.

**Advantages over action-value / ε-greedy methods:**

1. **Can approach determinism.** Preferences → ∞ for optimal actions; ε-greedy always keeps ε random, and soft-max-over-*values* plateaus at finite value gaps.
2. **Can represent arbitrary stochastic optimal policies.** Needed for imperfect-information games (poker bluffing) or **state aliasing** under FA.
   - **Example 13.1 (short corridor):** all 3 states look identical to features `$`x(s,\text{right}) = [1,0]^\top`$`, `$`x(s,\text{left}) = [0,1]^\top`$`. ε-greedy is trapped at ≤ −44 / −82; the optimal **stochastic** policy `$`P(\text{right}) \approx 0.59`$` → ≈ −11.6.
3. **The policy may be simpler to approximate** than `$`q`$` (e.g., Tetris).
4. **Inject prior knowledge** about the policy's form (often the most important reason).

**Theoretical edge:** action probabilities change **smoothly** with `$`\theta`$` (vs. ε-greedy's discontinuous jump when the argmax flips) → enables gradient ascent → **stronger convergence guarantees**.

---

## 2. The Policy Gradient Theorem ⭐ (§13.2)

**The problem:** `$`J(\theta)`$` depends on (a) action choices and (b) the **state distribution** — and (b)'s dependence on `$`\theta`$` runs through the *unknown* environment. How to get `$`\nabla J`$` without it?

**The theorem** — an analytic gradient with **no state-distribution derivative** (Eq. 13.5):

$$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a)\, \nabla\pi(a \mid s, \theta)$$

- `$`\mu`$` = on-policy distribution (same as Ch. 9/10).
- Proportionality constant = average episode length (episodic); **= 1** (continuing → exact equality).
- Proof sketch: unroll `$`\nabla v_\pi(s) = \sum_a [\nabla\pi \cdot q_\pi + \pi \cdot \nabla q_\pi]`$` repeatedly → states weighted by `$`\eta(s)`$` → normalize to `$`\mu`$` (Eq. 9.3).

**Why it matters:** this is the **replacement for the policy-improvement theorem** that §10.4 said we lost under function approximation. It gives a usable improvement *direction* even with approximation.

---

## 3. REINFORCE: Monte Carlo Policy Gradient (§13.3)

Make (13.5) sampleable. Write it as `$`\mathbb{E}_\pi[\sum_a q_\pi(S_t,a)\,\nabla\pi(a \mid S_t,\theta)]`$`, introduce the sampled action by multiply-and-divide by `$`\pi`$`, and use `$`\mathbb{E}[G_t \mid S_t, A_t] = q_\pi`$`:

$$\nabla J(\theta) = \mathbb{E}_\pi\!\left[ G_t\, \frac{\nabla\pi(A_t \mid S_t, \theta)}{\pi(A_t \mid S_t, \theta)} \right]$$

→ **REINFORCE update** (Eq. 13.8, using `$`\nabla\ln x = \nabla x / x`$`):

$$\theta_{t+1} = \theta_t + \alpha\, G_t\, \nabla\ln\pi(A_t \mid S_t, \theta)$$

- `$`\nabla\ln\pi(A_t \mid S_t, \theta)`$` = the **eligibility vector** — the *only* place the parameterization appears.
- **Intuition:** move `$`\theta`$` to make `$`A_t`$` more likely, scaled by **return `$`G_t`$`** (favor high-return actions) and **÷ `$`\pi(A_t \mid S_t)`$`** (so merely-frequent actions don't win by frequency).
- True **Monte Carlo**: uses the full `$`G_t`$`, updates after the episode (episodic only). **Unbiased**, converges to a local optimum, but **high variance → slow**.

### Pseudocode (episodic, discounted)
```
Loop per episode:
  generate S₀,A₀,R₁,…,S_{T−1},A_{T−1},R_T  following π(·|·,θ)
  for t = 0…T−1:
    G ← Σ_{k=t+1}^{T} γ^{k−t−1} R_k
    θ ← θ + α γᵗ G ∇ln π(Aₜ|Sₜ,θ)
```
(The `$`\gamma^t`$` factor appears in the boxed general discounted version; the text derives the `$`\gamma = 1`$` case.)

> Ex 13.3: for linear soft-max, `$`\nabla\ln\pi(a \mid s,\theta) = x(s,a) - \sum_b \pi(b \mid s,\theta)\, x(s,b)`$` (observed minus expected feature).

---

## 4. REINFORCE with Baseline (§13.4)

Subtract a state-dependent **baseline** `$`b(s)`$` (Eqs. 13.10–13.11):

$$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a \bigl(q_\pi(s,a) - b(s)\bigr)\, \nabla\pi(a \mid s, \theta)$$

$$\theta_{t+1} = \theta_t + \alpha\, \bigl(G_t - b(S_t)\bigr)\, \nabla\ln\pi(A_t \mid S_t, \theta)$$

- **Purpose = variance reduction, NOT bias.** The baseline lowers the variance of the gradient estimate while leaving its expectation (the true gradient) **unchanged**. That is the single reason it matters — faster, more stable learning.
- **Why it's unbiased — the zero-mean eligibility identity.** The expected eligibility vector, conditioned on the state, is zero:

$$\mathbb{E}_{a \sim \pi}\!\left[\nabla\ln\pi(a \mid s,\theta) \mid s\right] = \sum_a \pi(a \mid s)\frac{\nabla\pi}{\pi} = \sum_a \nabla\pi(a \mid s,\theta) = \nabla\sum_a \pi(a \mid s,\theta) = \nabla(1) = 0$$

  (probabilities always sum to 1 → constant → zero gradient). So for any **action-independent** `$`b(s)`$`, the term `$`\mathbb{E}[b(S_t)\nabla\ln\pi(A_t \mid S_t,\theta) \mid S_t] = b(S_t)\cdot 0 = 0`$` — it rides along the zero-mean vector and vanishes from the expectation. ⚠ Must depend on **state only**; an action-dependent baseline would *not* pull out and would introduce bias.
- **Intuition:** `$`G_t - \hat v(S_t)`$` measures "was `$`A_t`$` better or worse than average **in this state**" instead of the raw, large-magnitude `$`G_t`$` → centered near 0 → lower variance.
- **Why `$`\hat v(s)`$` specifically:** the variance-minimizing baseline is ≈ the state value, so a learned `$`b(s) = \hat v(S_t, w)`$` is the natural choice → `$`\delta = G_t - \hat v(S_t, w)`$`.
- **Still unbiased & still Monte Carlo:** here `$`\hat v`$` is *only* a baseline (subtracted from the full return `$`G_t`$`); nothing is bootstrapped → the estimator stays unbiased. (Contrast with §13.5.)
- Two step sizes `$`\alpha^\theta, \alpha^w`$`; `$`\alpha^w`$` easy (Ch. 9 rule), `$`\alpha^\theta`$` harder. Figure 13.2: the baseline makes REINFORCE learn **much faster**.

---

## 5. Actor–Critic Methods (§13.5)

**⭐ Baseline ≠ Critic — the key distinction (and the common confusion).** Both use `$`\hat v(s,w)`$`, but in *different roles*:

| | role of `$`\hat v(s,w)`$` | bootstrapping? | bias? | variance |
|---|---|---|---|---|
| **REINFORCE + baseline** (§13.4) | **baseline** — subtracted from full MC return `$`G_t`$` | **No** | **Unbiased** | reduced (still MC-ish) |
| **Actor–critic** (§13.5) | **critic** — used *inside* the bootstrapped target `$`R_{t+1} + \gamma\hat v(S_{t+1},w)`$` | **Yes** | **Biased** | much lower, fully online |

- **Baseline role** = subtract `$`\hat v(S_t)`$` → pure variance reduction, **no bias** (the §13.4 zero-mean argument).
- **Critic role** = bootstrap with `$`\hat v(S_{t+1})`$` → variance reduction *via* **bias** (asymptotic dependence on FA quality), and enables **online, incremental** learning.
- So REINFORCE-with-baseline is **not** actor–critic: it has the baseline role but *no bootstrapping*. Only bootstrapping makes it a "critic." Same **MC↔TD tradeoff** as Ch. 6, now for policies.

**One-step actor–critic:** replace the full `$`G_t`$` with the one-step TD target (Eq. 13.14):

$$\theta_{t+1} = \theta_t + \alpha\, \bigl(R_{t+1} + \gamma\hat v(S_{t+1},w) - \hat v(S_t,w)\bigr)\, \nabla\ln\pi(A_t \mid S_t,\theta) = \theta_t + \alpha\, \delta_t\, \nabla\ln\pi(A_t \mid S_t,\theta)$$

- The TD error `$`\delta_t = R_{t+1} + \gamma\hat v(S_{t+1},w) - \hat v(S_t,w)`$` carries **both roles at once**: the `$`+\gamma\hat v(S_{t+1},w)`$` is the critic (bootstrap target), the `$`-\hat v(S_t,w)`$` is the baseline. Together `$`\delta`$` estimates the **advantage** `$`q(S_t,A_t) - v(S_t)`$` — "how much better than average was `$`A_t`$` here."
- Critic learned by **semi-gradient TD(0)**. Fully online — states/actions/rewards used once, never revisited.

### Pseudocode (one-step actor–critic, episodic)
```
Loop per episode:
  init S; I ← 1
  while S not terminal:
    A ~ π(·|S,θ); take A, observe S′, R
    δ ← R + γ v̂(S′,w) − v̂(S,w)        (v̂(terminal)=0)
    w ← w + α^w δ ∇v̂(S,w)
    θ ← θ + α^θ I δ ∇ln π(A|S,θ)
    I ← γI;  S ← S′
```
- Generalizes to **n-step** (replace target with `$`G_{t:t+n}`$`) and **λ-return / eligibility traces** (separate traces `$`z^\theta, z^w`$` for actor & critic — Ch. 12 patterns).

---

## 6. Policy Gradient for Continuing Problems (§13.6)

Continuing case: performance = **average reward** (from §10.3, Eq. 13.15):

$$J(\theta) = r(\pi) = \lim_{h\to\infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t \mid A_{0:t-1} \sim \pi] = \sum_s \mu(s) \sum_a \pi(a \mid s) \sum_{s',r} p(s',r \mid s,a)\, r$$

- `$`\mu`$` = steady-state distribution (ergodicity assumed); invariant under `$`\pi`$` (Eq. 13.16).
- Values use the **differential return** `$`G_t = \sum_k (R_{t+k} - r(\pi))`$` (Eq. 13.17).
- **The policy gradient theorem (13.5) holds unchanged** in the continuing case (boxed proof) — forward & backward views identical.

### Pseudocode (actor–critic w/ traces, continuing)
```
init R̄, w, θ, S
z^w ← 0; z^θ ← 0
Loop forever:
  A ~ π(·|S,θ); take A, observe S′, R
  δ ← R − R̄ + v̂(S′,w) − v̂(S,w)        # DIFFERENTIAL TD error
  R̄ ← R̄ + α^{R̄} δ
  z^w ← λ^w z^w + ∇v̂(S,w)
  z^θ ← λ^θ z^θ + ∇ln π(A|S,θ)
  w ← w + α^w δ z^w
  θ ← θ + α^θ δ z^θ
  S ← S′
```
> **This is the direct payoff of Chapter 10:** average reward isn't just for value-based control — it's the natural, *differentiable* objective for continuing policy gradient. The differential-TD-error + `$`\bar R`$` machinery from the [Ch. 10 control notes](rl_on_policy_control_with_approximation_notes.md) now drives the policy.

---

## 7. Policy Parameterization for Continuous Actions (§13.7)

Policy methods handle **infinite/continuous** action spaces by learning the **statistics of a distribution** rather than a probability per action. Gaussian policy (Eq. 13.19):

$$\pi(a \mid s,\theta) = \frac{1}{\sigma(s,\theta)\sqrt{2\pi}} \exp\!\left(-\frac{(a - \mu(s,\theta))^2}{2\,\sigma(s,\theta)^2}\right)$$

with split `$`\theta = [\theta_\mu, \theta_\sigma]`$` (Eq. 13.20):

$$\mu(s,\theta) = \theta_\mu^\top x_\mu(s) \quad \text{(linear)}, \qquad \sigma(s,\theta) = \exp\!\left(\theta_\sigma^\top x_\sigma(s)\right) \quad \text{(exp → stays positive)}$$

All chapter algorithms then apply to real-valued actions. (Ex 13.4: the Gaussian eligibility vector splits into a mean part `$`\frac{a-\mu}{\sigma^2}\, x_\mu`$` and a std part `$`\left(\frac{(a-\mu)^2}{\sigma^2} - 1\right) x_\sigma`$`.)

---

## Quick Reference

| Concept | Key formula |
|---|---|
| Objective (episodic) | `$`J(\theta) = v_{\pi_\theta}(s_0)`$` |
| Objective (continuing) | `$`J(\theta) = r(\pi)`$` (average reward) |
| Soft-max policy | `$`\pi(a \mid s,\theta) = e^{h(s,a,\theta)} / \sum_b e^{h(s,b,\theta)}`$` |
| **Policy gradient theorem** | `$`\nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a)\, \nabla\pi(a \mid s,\theta)`$` |
| Eligibility vector | `$`\nabla\ln\pi(a \mid s,\theta) = \nabla\pi/\pi`$` (linear soft-max: `$`x(s,a) - \sum_b \pi(b \mid s)\,x(s,b)`$`) |
| REINFORCE | `$`\theta \leftarrow \theta + \alpha\, G_t\, \nabla\ln\pi(A_t \mid S_t,\theta)`$` |
| REINFORCE w/ baseline | `$`\theta \leftarrow \theta + \alpha\,(G_t - \hat v(S_t,w))\, \nabla\ln\pi(A_t \mid S_t,\theta)`$` |
| One-step actor–critic | `$`\theta \leftarrow \theta + \alpha\,\delta_t\, \nabla\ln\pi(A_t \mid S_t,\theta)`$`,  `$`\delta_t = R_{t+1} + \gamma\hat v(S_{t+1},w) - \hat v(S_t,w)`$` |
| Continuing `$`\delta`$` | `$`\delta = R - \bar R + \hat v(S',w) - \hat v(S,w)`$`,  `$`\bar R \leftarrow \bar R + \alpha^{\bar R}\delta`$` |
| Gaussian policy | `$`\mu = \theta_\mu^\top x_\mu(s)`$`, `$`\sigma = \exp(\theta_\sigma^\top x_\sigma(s))`$` |

## The method spectrum (variance ↔ bias)
```
REINFORCE ──── REINFORCE+baseline ──── n-step actor–critic ──── one-step actor–critic
(MC, unbiased,                                              (TD bootstrapping,
 high variance,                                              biased, low variance,
 offline/episodic)                                           fully online)
```
Same MC↔TD axis as Ch. 6/7/9, now for **policies** instead of values.

## Connections to earlier chapters
- **Restores what §10.4 lost:** the policy improvement theorem fails under FA; the **policy gradient theorem** is its replacement — works *because* `$`\pi`$` depends smoothly on `$`\theta`$`. See the [Ch. 10 control notes](rl_on_policy_control_with_approximation_notes.md).
- **Average reward returns (§13.6):** differential return + `$`\bar R`$` from Ch. 10 = the continuing actor–critic; `$`J(\theta) = r(\pi)`$` is differentiable. See the [Ch. 10 control notes](rl_on_policy_control_with_approximation_notes.md).
- **MC↔TD tradeoff:** REINFORCE (MC) → actor–critic (TD/bootstrapping). See the [Ch. 6 TD notes](rl_td_learning.md).
- **Features from Ch. 9:** preferences / `$`\mu`$` / `$`\sigma`$` are built from the same feature vectors `$`x(s,a)`$`. See the [Ch. 9 prediction notes](rl_on_policy_prediction_with_approximation.md).
- **Gradient bandits (§2.8):** the single-state special case of this whole chapter.
- **Singh LIRPG paper:** an A2C/PPO (actor–critic policy-gradient) agent with a meta-gradient on the reward — its core is exactly the `$`\nabla\ln\pi \cdot \text{advantage}`$` update here.
