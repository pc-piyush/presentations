# Towards Causal Representation Learning: The Concepts, Explained

A companion to the journal club deck on **Schölkopf, Locatello, Bauer, Ke, Kalchbrenner, Goyal & Bengio (2021), *Toward Causal Representation Learning*** ([arXiv:2102.11107](https://arxiv.org/abs/2102.11107); *Proceedings of the IEEE* 109(5)). It is built from the paper itself ([HTML version](https://arxiv.org/html/2102.11107v1)) and follows the paper's section structure. The matching slides are noted where helpful.

Each concept is explained twice:

- **The idea** gives the precise, conceptual version, staying close to the paper's wording.
- **In plain words** says the same thing without jargon, usually with an everyday analogy.

When something comes from outside the paper (standard textbook background, or a detail from a study the paper only cites), it is marked *(background)*. A glossary at the end collects every term, and a short final section lists where the slides and the paper differ.

---

## Contents

0. [The paper in one paragraph](#0-the-paper-in-one-paragraph)
1. [Introduction: why i.i.d. machine learning isn't enough](#1-introduction-why-iid-machine-learning-isnt-enough) (paper §I)
2. [Levels of causal modeling](#2-levels-of-causal-modeling) (§II)
3. [Causal models and inference](#3-causal-models-and-inference) (§III)
4. [Independent causal mechanisms](#4-independent-causal-mechanisms) (§IV)
5. [Causal discovery and machine learning](#5-causal-discovery-and-machine-learning) (§V)
6. [Learning causal variables](#6-learning-causal-variables) (§VI)
7. [Implications for machine learning](#7-implications-for-machine-learning) (§VII)
8. [Conclusion and open problems](#8-conclusion-and-open-problems) (§VIII)
9. [Critique and clinical relevance](#9-critique-and-clinical-relevance) (journal club discussion)
10. [Key takeaways](#10-key-takeaways)
11. [Glossary](#11-glossary)
12. [Where the slides and the paper differ](#12-where-the-slides-and-the-paper-differ)

---

## 0. The paper in one paragraph

**The idea.** Causal inference and machine learning grew up separately. Causal inference can reason about interventions and counterfactuals, but it assumes the variables are already given. Deep learning turns raw, high-dimensional data into useful representations, but those representations are purely statistical. The paper names the central open problem as *"causal representation learning, the discovery of high-level causal variables from low-level observations."*

**In plain words.** One field knows how to answer "what happens if I do X?" but needs a tidy spreadsheet of variables first. The other can read messy images and text but only learns which things tend to appear together. The paper asks how a machine could look at raw data, work out the meaningful "things" in it, *and* work out how those things cause one another.

It is a **perspective / review paper**, not an experimental one. It lists five contributions:

1. Levels of modeling, and how causal models differ from statistical ones (§II–III).
2. The **Independent Causal Mechanisms (ICM)** principle and the **Sparse Mechanism Shift (SMS)** hypothesis (§IV).
3. A review of causal discovery, classical and deep-learning based (§V).
4. How causal variables might be learned from data (§VI).
5. Implications for ML practice: semi-supervised learning, adversarial robustness, OOD generalization, pre-training, RL, science, multi-task learning (§VII).

The paper extends Schölkopf's 2019 essay *Causality for Machine Learning* and partly overlaps with it.

---

## 1. Introduction: why i.i.d. machine learning isn't enough

*Slides: "Paper at a Glance", "Motivation", "Why Conditional Probabilities Aren't Enough".*

### 1.1 The i.i.d. assumption

**The idea.** Most ML successes are large-scale pattern recognition on **independent and identically distributed (i.i.d.)** data: every example is drawn independently from one fixed distribution, and test data comes from that same distribution. The paper names four trends behind ML's success: (1) massive data, (2) high-capacity models, (3) high-performance computing, and (4) i.i.d. problems. It calls the last one *"often ignored, but crucial when it comes to causality."* Even RL is often made approximately i.i.d., for example by experience replay.

ML is *"rather limited"* at transferring knowledge from one problem to the next. The paper calls that kind of transfer horizontal, strong, or **out-of-distribution (OOD) generalization**.

**In plain words.** i.i.d. means "the exam questions come from the same pool as the practice questions." If that holds, memorizing the patterns in the practice questions works. The trouble starts when the exam is written by someone else.

### 1.2 Issue 1: Robustness

**The idea.** Vision systems can be thrown off by blur, noise, compression, shifts, rotations, or new viewpoints. Benchmarks such as ObjectNet (which controls background and rotation) test exactly these simple, algorithmically generated interventions. Data augmentation, pre-training, self-supervision, and architectural inductive biases help, but *"there has been no definitive consensus"* on how to solve the problem. The authors argue that generalizing beyond i.i.d. requires learning an underlying causal model, one that can describe distribution shifts as **interventions**.

The paper gives real failure cases:
- objects shown in a context that was negatively correlated with them in training;
- a diabetic retinopathy detector affected by socio-economic factors in clinics in Thailand;
- adversarial perturbations *"imperceptible to humans"* that cause errors such as misreading traffic signs.

**In plain words.** A model that learned "cows are on green grass" may fail to recognize a cow on a beach. Showing it more beaches helps, but it never learns *why* a cow is a cow. A causal model would treat the background as a separate factor that can change without changing what the animal is.

### 1.3 Issue 2: Learning reusable mechanisms

**The idea.** Infants' understanding of physics relies on objects that can be tracked over time and behave consistently. If a model of the world were built from modules that correspond to physical causal mechanisms, many of those modules would behave the same way across tasks and environments. Facing a new environment, the model would only need to adapt a few modules, and so *"one should thus require fewer examples to adapt."* Today's networks are trained end-to-end and adapted as a whole.

**In plain words.** When you move to a new city you don't relearn how to walk, how gravity works, or what a door is. You only learn the new street layout. Current neural networks are more like someone who has to relearn everything after moving.

### 1.4 Seeing vs. doing

**The idea.** Causation cannot be captured by logic or probabilistic inference alone; it needs the notion of **intervention**. Conditional probabilities cannot predict what happens when you act. The paper's example: seeing people open umbrellas suggests that it is raining, but *"closing umbrellas does not stop the rain."* Causal knowledge supports predictions far outside the observed data, including hypothetical situations and ones that require deliberate thinking. Statistical learning gives *"a rather superficial description of reality"* that holds only while conditions stay fixed.

The paper also mentions storks: across Europe, the number of storks is *"a reasonable predictor for human birth rates."* A statistical model can use this, since $P(\text{births} \mid \text{storks})$ really is predictive when nothing changes. But releasing more storks would not raise the birth rate. That is a question about $P(\text{births} \mid do(\text{storks}))$, which observational data alone cannot answer.

The paper's goal for AI is *"thinking in the sense of Konrad Lorenz, i.e., acting in an imagined space"*: being able to try out actions in your head before doing them.

**In plain words.** "Seeing" and "doing" are different questions. Seeing umbrellas tells you about the weather. Closing them doesn't change it. Data collected by passive watching usually only answers the "seeing" question.

> **The $do(\cdot)$ operator** *(background notation; the paper uses it in §II)*. $P(Y \mid X = x)$ means "among cases where we *happened to see* $X = x$, how is $Y$ distributed?" $P(Y \mid do(X = x))$ means "if we *forced* $X$ to be $x$, how would $Y$ be distributed?" They coincide only when nothing else links $X$ and $Y$.

### 1.5 What animals learn from, and ML throws away

**The idea.** ML often *"disregards information that animals use heavily"*: interventions, domain shifts, and temporal structure. Standard practice treats these as nuisances to be removed so the data looks i.i.d.

**In plain words.** A child learns by poking things and watching what happens. Standard ML throws away exactly that kind of information.

---

## 2. Levels of causal modeling

*Slides: "A Hierarchy of Models (Table I)", "The Gold Standard: Physical Models", "Four Things We Might Want From a Model".*

### 2.1 A hierarchy of models (Table I)

| Model | Predict i.i.d. | Predict under shift / intervention | Counterfactuals | Physical insight | Learn from data |
|---|:-:|:-:|:-:|:-:|:-:|
| Mechanistic / physical | yes | yes | yes | yes | ? |
| Structural causal model | yes | yes | yes | ? | ? |
| Causal graphical model | yes | yes | no | ? | ? |
| Statistical model | yes | no | no | no | yes |

**The idea.** Models form a ladder. Physical models (differential equations) can do everything but usually need *"an intelligent human"* to write them. Statistical models can be learned from observational data but only predict when conditions stay the same. Causal models *"lie in between, abstracting away from physical realism while retaining the power to answer certain interventional or counterfactual questions."* **Causal discovery** tries to learn such models from data, replacing expert knowledge with *"weak and generic assumptions."*

**In plain words.** A physics textbook explains everything but someone has to write it. A pattern-matcher learns by itself but can't explain or adapt. The goal is a model that learns by itself *and* understands cause and effect.

### 2.2 The gold standard: physical models

**The idea.** A physical model is a system of coupled differential equations:

$$\frac{d\mathbf{x}}{dt} = f(\mathbf{x}), \qquad \mathbf{x} \in \mathbb{R}^d, \qquad \mathbf{x}(t_0) = \mathbf{x}_0 .$$

If $f$ is Lipschitz, the Picard–Lindelöf theorem guarantees a unique solution locally, so the system's immediate future is determined by its past. A first-order discretization makes this explicit:

$$\mathbf{x}(t + dt) = \mathbf{x}(t) + dt \cdot f(\mathbf{x}(t)).$$

If the equations capture the real mechanisms, causal structure can be read directly off $f$ (which components drive which), and any intervention can be simulated by changing a parameter and re-running. The paper adds a caution: equations that reproduce how a system evolves but don't capture its mechanisms (because of hidden confounding, or a choice of variables that doesn't respect causal structure) may not be causally meaningful.

**In plain words.** A physics simulation of a ball says exactly where the ball will be next, given where it is now. Want to know what happens on the Moon? Change gravity and run it again. The catch is that someone had to work out the equations, and for most real problems (a patient, an economy) nobody has them.

### 2.3 Four questions a model might answer (§II-A to §II-D)

**A. Predict in the i.i.d. setting.**
*Idea:* estimate $P(Y \mid X)$, for example *"what is the probability that this particular image contains a dog?"*, or the probability of heart failure given blood pressure and other measurements. With enough i.i.d. data, statistical models answer these provably. Spurious correlations like storks–births are fine here. But under intervention, predictions can become *"(arbitrarily) inaccurate."*
*Plain words:* if the world stays the same, any reliable pattern is useful, causal or not.

**B. Predict under distribution shift.**
*Idea:* interventional questions go beyond i.i.d. *"Is increasing the number of storks in a country going to boost its human birth rate?" "Would fewer people smoke if cigarettes were more socially stigmatized?"* Interventions need not be deliberate: preferences drift, and deployed models meet data unlike their training data. The paper argues that predicting under shift *"should not be reduced to just the accuracy on a test set"*: a trustworthy model's predictions stay valid when conditions change.
*Plain words:* will the model still work at a new hospital, in a new season, with a new camera?

**C. Answer counterfactual questions.**
*Idea:* counterfactuals ask why something happened and what would have happened under a different action. They are harder than interventional questions. The paper's heart-failure contrast:
- interventional: *"how does the probability of heart failure change if we convince a patient to exercise regularly?"*
- counterfactual: *"would a given patient have suffered heart failure if they had started exercising a year earlier?"*

The question types form a continuum, written $P(A \mid B, do(C))$. If $B$ is empty, it's a plain intervention. If $B$ pins down *all* the unobserved noise terms, it's a counterfactual. In between, $B$ is partly informative. RL Q-functions are sometimes called counterfactual but sit closer to interventions. Counterfactual reasoning is described as *"especially critical"* in RL.
*Plain words:* an interventional question is "what happens to *people like this* if we give the drug?" A counterfactual is "this *specific* patient didn't get the drug and got worse; would *they* have recovered if they had?"

**D. Learn from the data we have.**
*Idea:* data varies along two axes:
- *observational vs. interventional:* purely observational i.i.d. data is *"seldom strictly available"*; the middle ground is data with shifts or unknown interventions;
- *hand-engineered vs. raw:* tidy semantic variables vs. pixels, which don't expose causal information.

Statistical models learn efficiently from raw observational data. Causal structure learning *"frequently requires collecting data from multiple environments, or the ability to perform interventions."* It also often assumes **causal sufficiency** (no unobserved common causes). Much prior knowledge is hidden in *which variables get measured*; the paper hopes inductive biases, meta-learning, and self-supervision can replace some of that expert choice.
*Plain words:* the easiest data to get (passively recorded, raw) is the hardest to learn causes from.

---

## 3. Causal models and inference

*Slides: "Reichenbach's Common Cause Principle", "Structural Causal Models", "Interventions", "Graphical Models vs. SCMs".*

### 3.1 Why direction gets lost: the laptop and the rucksack

**The idea.** The paper's shopping example: Alice searches for a laptop rucksack and the recommender suggests buying a laptop. Owning a laptop *causes* wanting a rucksack, not the other way round. But statistically, learning that a customer has a laptop reduces uncertainty about the rucksack by exactly as much as the reverse (the mutual information is symmetric), so the direction is invisible in the statistics. The direction lives in the *mechanism*. Also, recommending an item is itself an intervention, which pushes the system outside the i.i.d. setting the recommender was trained in.

**In plain words.** Correlation is a two-way street; causation is one-way. A system that only sees correlations will sometimes recommend things backwards.

### 3.2 Reichenbach's common cause principle

**The idea.** If two observables $X$ and $Y$ are statistically dependent, there exists a variable $Z$ that causally influences both and explains all of their dependence, meaning they become independent once you condition on $Z$ ($X \perp\!\!\!\perp Y \mid Z$). $Z$ may coincide with $X$ or $Y$.

Storks again, with $X$ = stork frequency and $Y$ = birth rate:
1. $X \to Y$: storks bring babies;
2. $X \leftarrow Y$: babies attract storks;
3. $X \leftarrow Z \to Y$: a third variable, such as economic development, drives both.

Observational data alone cannot distinguish these. Hence: *"A causal model thus contains genuinely more information than a statistical one."* With only two variables, the direction is hard to determine. With more variables, causal structure implies **conditional independence** patterns that can be used.

**In plain words.** Every correlation has a causal story behind it, but the correlation doesn't tell you which one. Ice-cream sales and drownings rise together because summer drives both. Once you hold the season fixed, the link disappears; that is what "explains the dependence" (or "screens off") means.

### 3.3 Structural Causal Models (SCMs)

**The idea.** Variables sit at the nodes of a directed acyclic graph (DAG). Each one is produced by an *assignment*:

$$X_i := f_i(\mathbf{PA}_i, U_i), \qquad i = 1, \dots, n,$$

where $f_i$ is deterministic, $\mathbf{PA}_i$ are $X_i$'s parents in the graph, and $U_i$ is an unexplained random noise term. The noises $U_1, \dots, U_n$ are **jointly independent**. (If two noises were dependent, the common cause principle would demand another variable to explain that, and the model would not be causally sufficient.) The ":=" means "is computed from," not mathematical equality.

Applying the assignments in order produces the observational joint distribution. It satisfies the **causal Markov condition**: given its parents, each variable is independent of its non-descendants.

The paper offers a "gossip" analogy: the independent noises act like *information probes* that spread through the graph. The pattern of dependencies they leave behind is a footprint that independence tests can pick up. That footprint may not single out one graph, and with only two variables it says nothing about direction, so extra assumptions are needed. The functions $f_i$ leave a footprint too, which is where ML's theory of function classes becomes useful (§5).

**In plain words.** An SCM is a recipe. Each ingredient is made from earlier ingredients plus some randomness specific to it. For example: `fitness := f(genes, noise₁)`, `exercise := g(fitness, noise₂)`, `heart_health := h(fitness, exercise, noise₃)`. The noise covers everything about this particular case that the model doesn't spell out.

### 3.4 Causal vs. entangled factorization

**The idea.** The DAG plus independent noises gives the **causal (disentangled) factorization**:

$$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid \mathbf{PA}_i).$$

Any other ordering of the chain rule, such as

$$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid X_{i+1}, \dots, X_n),$$

is an **entangled factorization**. Both describe the same joint distribution. Only the causal one decomposes into pieces that match the structural assignments, one per **mechanism**. The difference doesn't show up in observational data. It shows up when something *changes*, and the rest of the paper builds on that.

**In plain words.** You can describe a cake as separate steps (make batter, bake, frost) or as one tangled description of "how frosting relates to batter given oven temperature." Both describe the same cake. Switch to a different frosting, and the step-by-step version needs one line changed while the tangled version needs rewriting throughout.

### 3.5 Latent variables and confounders

**The idea.** An unobserved variable can make two variables look related when neither causes the other, or it can distort a real causal relation, as when a **confounder** causes both. Confounders may or may not be known to the modeller.

**In plain words.** People who carry lighters get lung cancer more often. Lighters don't cause cancer; smoking causes both. If smoking isn't recorded, an analysis can wrongly conclude that lighters are dangerous.

### 3.6 Interventions

**The idea.** An intervention changes some of the structural assignments and leaves the rest alone. The paper (following Eaton & Murphy) distinguishes:

- **No intervention:** only observational data.
- **Hard / perfect:** *"the function in the structural assignment … is set to a constant (implying that the value of the variable is fixed)."* The variable is cut off from its parents.
- **Soft / imperfect:** *"the structural assignment … is modified by changing the function or the noise term."* The variable still depends on its parents, but differently.
- **Uncertain:** *"the learner is not sure which mechanism/variable is affected by the intervention."*

The paper adds that a causal model should come with a set of *allowed* interventions. Which causal model is "right" can depend on which interventions you care about, so models should be judged by their usefulness for a given set of tasks.

**In plain words.**
- Hard: "Everyone takes exactly 10 mg." Whatever would normally decide the dose no longer matters.
- Soft: "Doctors now prescribe a bit more than before." The dose still depends on the patient, but the rule has shifted.
- Uncertain: "The new hospital is different in some way we can't pin down."

### 3.7 Statistical models vs. causal graphical models vs. SCMs (Fig. 1)

**The idea.** Fig. 1 contrasts the two: a statistical model *"specifies a single probability distribution,"* while *"a causal model represents a set of distributions, one for each possible intervention."*

Arrows in an ordinary graphical model need not be causal. $X_1 \to X_2 \to X_3$ and $X_1 \leftarrow X_2 \leftarrow X_3$ imply the same independence ($X_1 \perp\!\!\!\perp X_3 \mid X_2$). They are **Markov equivalent**, so the Markov condition alone can't determine causal direction. A graphical model becomes *causal* when its arrows are causal; then you can compute interventions: disconnect the intervened variable from its parents, fix its value, and sample its descendants.

|  | Interventions | Counterfactuals |
|---|:-:|:-:|
| Causal graphical model | yes | no |
| SCM | yes | yes |

Only SCMs support counterfactuals, because *"to compute counterfactuals, we need to fix the value of the noise variables."* Many different combinations of function and noise produce the same conditional distribution, so a causal graph with its conditionals isn't enough on its own.

*(Background.)* Pearl's standard three-step recipe for counterfactuals spells out what "fixing the noise" means:
1. **Abduction:** from what actually happened, infer this case's noise values.
2. **Action:** apply the hypothetical intervention.
3. **Prediction:** recompute the outcome with the *same* noise values.

**Causal learning and reasoning.** Statistical learning works with a joint distribution and assumptions about function classes. Causal learning also uses the causal factorization and the interventions it admits. Given a causal model (learned or supplied by experts), **causal reasoning** answers questions about interventions, counterfactuals, and potential outcomes.

**In plain words.** A causal graph tells you which way the influence flows, which is enough for "what happens on average if we do X?" A counterfactual about *one specific person* also needs that person's individual quirks: their noise terms. An SCM keeps track of those, so it can replay the same person's story with one thing changed.

---

## 4. Independent causal mechanisms

*Slides: "The ICM Principle", "Example: Altitude and Temperature", "The Sparse Mechanism Shift (SMS) Hypothesis".*

### 4.1 Where the intuition comes from: vision

**The idea.** Our visual system assumes that an object is independent of the light falling on it, of our viewpoint, and of the illumination. This is the **generic viewpoint assumption**: no accidental coincidences, such as 3D structures lining up perfectly in a 2D image or shadow edges landing exactly on texture edges. When such coincidences do happen, we see optical illusions. **Structure from motion** relies on the same idea: moving around an object changes the viewpoint while lighting, position, and shape stay the same, which lets us infer 3D shape even with one eye.

**In plain words.** Your brain bets that "what the thing is" and "how you're looking at it" are separate facts. That bet is usually right, and when it's wrong you get an optical illusion.

### 4.2 Example: altitude and temperature

**The idea.** Altitude $A$ of weather stations causes their average annual temperature $T$. Data from Austria and from Switzerland have different joint distributions $P(A, T)$, because the distribution of station altitudes $P(A)$ differs. But $P(T \mid A)$ encodes physics and should be nearly the same in both.

- Causal factorization $P(A)\,P(T \mid A)$: one factor changes by country, the other generalizes.
- Entangled factorization $P(T)\,P(A \mid T)$: both factors mix geography with physics, so both change.

Without knowing the causal structure, the cross-country similarity is lost.

**In plain words.** Air gets colder as you climb at the same rate in every country. What differs is how many mountains each country has. Store your knowledge as "how many mountains here" plus "how cold it gets with height," and you only update the first part when you travel. Store it the other way round, and both parts are wrong in the new country.

### 4.3 The ICM principle

**The idea.** The causal generative process of a system's variables is *"composed of autonomous modules that do not inform or influence each other."* For the causal factorization this means:

- **(a) Influence:** changing one mechanism $P(X_i \mid \mathbf{PA}_i)$ does not change any other mechanism.
- **(b) Information:** knowing some mechanisms tells you nothing about the others.

This entails that causal variables can be intervened on separately, that subsystems are modular and autonomous, and that mechanisms are invariant. With two variables, it says the distribution of the cause is independent of the mechanism that produces the effect from it.

Two clarifications from the paper:
- **Independent mechanisms ≠ independent variables.** Variables can be strongly dependent even when all mechanisms are independent. Independence of the noise terms $U_i$ is a special case of ICM.
- **Algorithmic formulation.** Because statistical information is awkward to define for deterministic mechanisms, the paper also states ICM in terms of **Kolmogorov complexity** (the length of the shortest program that outputs a string). Write each mechanism as a bit string; the mechanisms are independent if *"joint compression of these strings does not save space relative to independent compressions,"* i.e. their mutual algorithmic information is zero. In such **algorithmic graphical models**, each $X_i$ is the output of a fixed computer running program $U_i$ on input $\mathbf{PA}_i$, and analogues of the Markov condition and the causal factorization still hold. The paper concludes that *"causality is not intrinsically bound to statistics."*

**History.** The idea has deep roots: Haavelmo (1944) assumed that changing one structural equation leaves the others intact; Simon's invariance criterion holds that the true causal order is invariant under the right interventions; Aldrich (1989) quotes Frisch asking of a relation *"How autonomous is it?"*; Pearl (2009) notes that causal discovery works under slightly varying conditions, where accidental independencies break but structural ones persist.

**In plain words.** Nature is built from separate, swappable parts. (a) Changing one part doesn't break the others: replacing a car's tyres doesn't change how the engine works. (b) Studying one part tells you nothing about another: the tyre brand gives no clue about engine design. The compression version says the same thing: if two rules are truly separate, describing them together takes no less space than describing them one at a time.

### 4.4 The Sparse Mechanism Shift (SMS) hypothesis

**The idea.** *"Small distribution changes tend to manifest themselves in a sparse or local way in the causal/disentangled factorization."* They should usually not affect all factors at once. Since every real-world distribution is a product of causal mechanisms, any change must come from a change in at least one mechanism. In an entangled factorization, by contrast, a single change typically spreads across many or all terms.

The paper notes that SMS has been used as a learning signal: for learning causal models, for modular architectures, and for disentanglement. If the right representation is the one under which observed shifts look sparse, then searching for sparse-shift representations gives supervision that doesn't need labels.

**In plain words.** When something changes in the world, usually only one or two things actually change. A lab installs a new microscope; patient biology stays the same. If your model is organised the right way, that change shows up as one small edit. If it's organised the wrong way, it shows up as a mess everywhere. So "which description makes changes look small and local?" is a clue to the right description.

---

## 5. Causal discovery and machine learning

*Slide: "Causal Discovery Meets ML".*

### 5.1 The conditional-independence route

**The idea.** Under assumptions such as **faithfulness**, conditional independence (CI) tests recover part of the graph from observational data, up to its **Markov equivalence class**: graphs with the same skeleton (edges without directions) and the same **immoralities** ($X_i \to X_j \leftarrow X_k$ with $X_i$ and $X_k$ not directly connected).

Two problems:
1. CI testing is hard with finite data, especially with continuous, multi-dimensional conditioning sets. Kernel methods are the main modern approach.
2. With only two variables there are no conditional independencies to test, so the Markov condition says nothing about direction.

*(Background.)* Faithfulness means every independence in the data is implied by the graph, not produced by a coincidental cancellation of effects. The paper cites but does not define it.

**In plain words.** CI tests are detective work: "if A and C stop being related once I know B, then B probably sits between them." This only narrows things down to a family of candidate graphs, needs lots of data, and doesn't work at all for a single pair of variables.

### 5.2 Restricting the function class: additive noise models

**The idea.** Both problems can be addressed by restricting the class of functions, which is where ML's learning theory comes in. In the general model $X := U$, $Y := f(X, V)$, the noise $V$ can select among arbitrarily different functions, so a finite sample says little. The **additive noise model (ANM)** restricts this:

$$X := U, \qquad Y := f(X) + V, \qquad U \perp\!\!\!\perp V.$$

Key result: given data from an ANM in one direction, you generically cannot fit an ANM in the reverse direction. That breaks the symmetry between cause and effect. The exception is the linear case with Gaussian noise, where both directions fit equally well. Extensions cover linear non-Gaussian models (LiNGAM), non-linear rescalings, loops, confounders, and multivariate settings. Other methods that beat chance at finding direction use compression, generative networks, or train a classifier to label pairs of variables as cause→effect or effect→cause.

**In plain words.** If $Y$ is "a function of $X$ plus independent random wobble," then trying to explain $X$ from $Y$ leaves wobble that *does* depend on $Y$. That mismatch gives away the wrong direction. If everything is linear and bell-curve shaped, the fingerprint disappears.

### 5.3 Multiple environments, and meta-learning

**The idea.** Under SMS, most of the causal structure stays fixed across shifts. So observing a system in several **environments** or **contexts** can reveal causal structure. Contexts can come from interventions, non-stationary time series, or multiple views of the same system. Contexts can also be *tasks*, which links causal discovery to meta-learning:
- Bengio et al. relate meta-learning to causal invariance: a correct causal model should adapt *faster* to an intervention than a merely predictive one.
- Ke et al. extend this to many variables and unknown interventions, turning the discrete search over graphs into continuous optimization.
- Dasgupta et al. have an RL agent learn a causal model.

All of these still assume the meaningful variables are given; none learns them from pixels.

**In plain words.** Data from several settings works like a natural experiment: whatever stays stable across settings is likely a real mechanism. And the right causal model is the one that needs the fewest adjustments when the setting changes, so "adapts fastest" is itself a clue.

---

## 6. Learning causal variables

*Slides: "The Setting: Causal Variables Aren't Given", "Problem Formulation", "SMS in Pixel Space", "Three Problems for CRL".*

### 6.1 The setting

**The idea.** Causal discovery assumes its units are random variables already connected by a graph. Real observations aren't structured that way; images don't come labelled with their objects. Fig. 2 describes perceptual data as *"entangled views of the state of an unknown causal system,"* usually with no labels except perhaps a task label. Full recovery of the causal system may be unrealistic, so the aim is a representation that exposes *enough* causal structure to support causal statements for downstream tasks. Example: to detect objects a robot could manipulate, separate an object's intrinsic properties from its pose and appearance, so the detector is robust when pose and appearance shift.

Choosing the level of description (granularity) is hard. Fields like economics, medicine, and psychology use coarse variables whose interventions are not well defined. Causal variables can be coarse-grained summaries of finer models (microscopic equations, ODEs, time series aggregated over time).

The paper's proposed architecture: embed an SCM inside a larger neural network whose inputs and outputs are high-dimensional. Its modules can then be fine-tuned and reused, and SMS can be used to shape the structure.

**In plain words.** A camera doesn't record "ball," "table," "gravity." It records a million coloured dots. Causal tools need the ball and the table as named things. Causal representation learning is about getting from the dots to the things, choosing things that actually cause one another.

### 6.2 Problem formulation

**The idea.** Observations $X$ (say, an image with $d$ pixels) are generated from $n \ll d$ latent causal variables through a non-linear map $G$:

$$X = G(S_1, \dots, S_n), \qquad S_i := f_i(\mathbf{PA}_i, U_i), \qquad P(S_1, \dots, S_n) = \prod_{i=1}^{n} P(S_i \mid \mathbf{PA}_i).$$

Architecture: an **autoencoder with an SCM in the bottleneck**.
- An **encoder** $q: \mathbb{R}^d \to \mathbb{R}^n$ maps $X$ to noise variables $U_1, \dots, U_n$.
- The **causal mechanisms** $f$ turn noises into causal variables $S_1, \dots, S_n$.
- A **decoder** $p: \mathbb{R}^n \to \mathbb{R}^d$ maps the causal variables back to an image.
- Train by reconstruction error so that $p \circ f \circ q \approx \mathrm{id}$.

If the graph is known, $f$'s structure can be fixed. If not, $f$ gets absorbed into the decoder ($\tilde p = p \circ f$), and the causal graph is implicit in the decoder. Leeb et al. propose a structured decoder along these lines.

**The catch is identifiability.** From i.i.d. data alone, recovering the true noise variables is ill-defined: infinitely many solutions give the same observational distribution. Extra inductive biases are needed to prefer the right one.

**In plain words.** Compress an image into a few causal "knobs," let the knobs influence each other according to learned cause-and-effect rules, then redraw the image. If the redrawing matches, the knobs capture what matters. The problem is that many different sets of knobs redraw the image equally well, and nothing in the images alone forces the *true* causal ones. Extra hints are needed, such as data from changing environments.

### 6.3 SMS in pixel space (Fig. 3)

**The idea.** Fig. 3 shows an intervention (*"which may or may not be intentional/observed"*) that moves one finger of a robotic hand; as a consequence, an object (a cube) falls. In the right causal variables the change is **sparse**: the finger's position changed, and the cube's position changed as a result. In pixel space the change is **dense**, or entangled: many pixels change at once. In the extreme, *"all pixels may change as a result of a sparse intervention,"* for example if the camera angle or lighting changes.

**In plain words.** To a camera, a cube falling off a hand changes much of the picture. To a person, only one thing happened: a finger moved and gravity did the rest. A good representation sees it the way the person does.

### 6.4 Problem 1: Learning disentangled representations

**The idea.** Find an encoder whose latent variables satisfy the disentangled factorization $\prod_i P(S_i \mid \mathbf{PA}_i)$, with each conditional separately manipulable and largely invariant across related problems.

- Most disentanglement work assumes fully *independent* factors, the special case where every $\mathbf{PA}_i$ is empty. ICM is more general: a fork and a knife are statistically dependent (they appear together), yet we want them as separate entities.
- **Object-centric** representations are another special case: objects can be intervened on separately, so they make natural building blocks, complemented by orientation, viewpoint, and lighting.
- To make the latents *causal*, use ICM: independent noises and independent mechanisms, enforced through invariance across problems, sparse changes, or independent interventions.
- Locatello et al. showed that SMS is theoretically *sufficient* for disentanglement given suitable training data. SMS can supervise learning even when variables have parents.
- Which factors can be disentangled depends on which interventions are observed; different supervision signals identify different subsets of factors.

**In plain words.** Learn separate knobs for separate real-world factors, so turning one doesn't secretly turn others, and use "the world changed, but only a little" as the clue for where the knobs are.

### 6.5 Problem 2: Learning transferable mechanisms

**The idea.** Real agents have limited data and limited compute (brains have a size limit; robots run on batteries). They can't rely on *"large-scale labeling work done by humans."* So robust AI will need to reuse components across tasks and environments, ideally with a modular structure that mirrors the world's modularity.

The paper's lighting example: a visual system shouldn't train a separate face recogniser for each lighting condition. It should have a gain-control module that factors out lighting variations spanning orders of magnitude. That module need not mimic the physics of light, but it plays the corresponding role, which may push models toward a structure that mirrors the world. Looking for independent mechanisms, possibly through competitive training between modules, is a sensible inductive bias.

**In plain words.** Build a toolkit of reusable parts instead of a custom machine for every job. One "adjust for brightness" tool serves every face, every room, every task.

### 6.6 Problem 3: Learning interventional world models and reasoning

**The idea.** Deep learning captures statistics but *"does not care about the interventional properties of the variables it analyzes or reconstructs."* Causal representation learning should produce models that support *"intervention, planning, and reasoning,"* realizing Lorenz's thinking as acting in an imagined space. That requires reflecting on one's actions and imagining alternatives, possibly needing "(the illusion of) free will." The paper speculates that self-consciousness may relate to having a variable that represents *oneself* in that imagined space, and that free will may be a way to communicate about actions, which matters for social and cultural learning.

**In plain words.** The model should let an agent imagine "if I push this, that happens" before doing it, the way people rehearse actions in their heads, including picturing themselves in the scene.

---

## 7. Implications for machine learning

*Slides: "Semi-Supervised Learning", "Adversarial Examples and Robustness", "Re-reading Today's Tricks", "Science, Medicine and Multi-Task Learning".*

The paper drops the i.i.d. assumption and assumes test data may come from a different distribution *"but involving (mostly) the same causal mechanisms."* It names four challenges:

- (a) abstract causal variables often have to be inferred from low-level inputs;
- (b) there is no consensus on which aspects of data reveal causal relations;
- (c) the usual train/test protocol may be insufficient, and new benchmarks may need environment information and interventions;
- (d) scalable, numerically sound algorithms are often missing.

### 7.1 Semi-supervised learning: direction matters

**The idea.**
- **Causal direction** $X \to Y$: $P(X, Y) = P(X)\,P(Y \mid X)$. By ICM, $P(X)$ carries no information about $P(Y \mid X)$, so semi-supervised learning (SSL) should be futile, to the extent it relies on unlabeled data about $P(X)$ to improve $P(Y \mid X)$.
- **Anticausal direction** $Y \to X$ (predicting against the generative direction): Daniušis et al. define a measure of dependence between $P(X)$ and $P(Y \mid X)$ that is zero in the causal direction but strictly positive in the anticausal one. So unlabeled $X$ carries information about the conditional, and SSL can work.

The study Schölkopf et al. (2012), *On Causal and Anticausal Learning*, *"empirically corroborated these predictions"*, and about 17 follow-up papers built on it. The idea also inspired invariance-based causal discovery in statistics.

The standard SSL assumptions fit this picture: the **cluster assumption** (labels don't change within a cluster of $P(X)$), **low-density separation** (the decision boundary lies where $P(X)$ is small), and **semi-supervised smoothness**. **Co-training** assumes two views are independent given the label, exactly what you expect when the label causes both views, an anticausal setting.

*(Background examples.)* Causal: predicting a protein from its mRNA sequence. Anticausal: classifying a handwritten digit from its image, where the intended digit causes the ink.

**In plain words.** If the label *causes* the input (the writer chose "7" and then drew it), then piles of unlabeled images naturally fall into clusters, one per digit, and those clusters help draw boundaries. If the input *causes* the label, the way inputs are spread out says nothing about the rule that maps them to outputs, so extra unlabeled inputs don't help.

### 7.2 Adversarial vulnerability

**The idea.** Adversarial examples are minute, human-invisible changes to an input that change the output. They violate i.i.d., since the modified inputs don't come from the training distribution. Current defenses approximate human robustness with a small $\ell_p$-ball around each input in pixel space.

The paper's **writer example** shows why the generative direction matters. Three setups yield the same statistical classifier:
1. a writer is given labels $y$ and draws images $x$ (anticausal: changing $y$ changes $x$);
2. a writer picks digits and records the labels (confounded: one effect predicts another);
3. a person is shown images and labels them.

If a predictor approximates a causal mechanism that is robust and transferable, adversarial examples *"should be harder to find."* Supporting defenses: **analysis by synthesis**, which classifies by checking the input against a generative model of each class, and reconstructing the input with an autoencoder before classifying. Attacks might still exploit flaws in how the mechanism is approximated.

**In plain words.** Adversarial attacks nudge pixels in ways no real-world process would. A model that understands how a real "panda" image comes about would notice that the tampered image couldn't have been produced that way. Analysis by synthesis asks: "if this really were a gibbon, would it look like this?"

### 7.3 Robustness and strong generalization

**The idea.** Standard ML minimizes empirical risk on the training distribution:

$$g^\star = \arg\min_{g \in \mathcal{H}} \hat R_{P(X,Y)}(g), \qquad \hat R_{P(X,Y)}(g) = \hat{\mathbb{E}}_{P(X,Y)}\big[\mathrm{loss}(Y, g(X))\big].$$

OOD risk measures the same loss under a different distribution $P^\dagger$. **Environments** are *"the collection of different circumstances that give rise to the distribution shifts such as locations, times, experimental conditions, etc."* In a causal model, environments correspond to interventions on variables or mechanisms. Taking the worst case over a set $\mathcal{E}$ of observed environments gives a min-max problem:

$$g^\star = \arg\min_{g \in \mathcal{H}} \max_{P^\dagger \in \mathcal{E}} \hat{\mathbb{E}}_{P^\dagger}\big[\mathrm{loss}(Y, g(X))\big].$$

Here $\mathcal{E}$ is the observed subset of all interventional distributions $P_\mathcal{G}$ that the causal graph allows. Learning models that are stable across training environments and generalize to new ones is *"nontrivial (and in some cases provably hard)."* If the observed environments don't cover the possible ones, the extra error *"may be arbitrarily large in the worst case."*

The paper expects modular, causally factorized models to be relatively robust to swapping or modifying single components. Its credit-scoring example: paying off debt plausibly *causes* better repayment, but moving to a richer neighbourhood probably doesn't. A score that uses only causal features may therefore be harder to game. The paper presents this as intuition, not a proof.

*(Background.)* The related literature it cites (e.g., invariant causal prediction, invariant risk minimization) shows that when environments arise from interventions on variables other than the target, a predictor using only the target's causal parents performs equally well in all of them, which is why causal predictors are natural candidates for the min-max solution.

**In plain words.** Instead of being best on average, be *never terrible*, even in the worst setting you'll face. Relying on true causes is stable across settings; shortcuts look good in some settings and fail badly in others. But you can only guard against settings that resemble the ones you've seen.

### 7.4 Pre-training, data augmentation, and self-supervision

**The idea.** The min-max problem is hard to solve directly, so the paper reads common techniques as approximations to it:
- **Pre-training on a large, diverse corpus** is still ordinary empirical risk minimization. It helps only if the extra data happens to cover other environments in $P_\mathcal{G}$.
- **Data augmentation** applies artificial interventions (rotations, small translations, flips) to teach invariances, e.g. that shifting an image doesn't change its class. It effectively *defines* the set of interventions the model should be robust to. Augmentation policies can be searched for (e.g. with RL).
- **Self-supervision** solves pretext tasks with artificial labels, which forces representations to capture information about $P(X)$. ICM might help *automate* the design of pretext tasks. (Following §7.1, information about $P(X)$ is most useful in anticausal settings.)
- **Adversarial training** solves the min-max explicitly, with the "environments" being attacks instead of natural interventions.

Suggested direction: combine large-scale training, augmentation, self-supervision, and robust fine-tuning on data from several, possibly simulated, environments.

**In plain words.** Rotating training images is a way of telling the model "rotation doesn't change what this is." That works, but only for the changes you thought to simulate. A causal view makes the list of protected changes explicit.

### 7.5 Reinforcement learning

**The idea.** RL is closer to causality than most ML: on-policy RL effectively estimates do-probabilities for the agent's own interventions. Off-policy and batch settings make things subtler. The paper splits RL's causal side into *causal induction* (learning a causal model of the environment, made harder because variables aren't given) and *causal inference* (planning with that model).

- **World models.** Model-based RL learns the effects of actions; generative world models serve as Lorenzian imagined spaces in which to train agents. Structured world models break the environment into entities. Many world models are partial, so unobserved parts of the environment can confound actions and rewards; the paper suggests adjusting with the backdoor criterion, conditioning on the policy.
- **Generalization, robustness, fast transfer.** RL needs huge numbers of samples and is brittle to small visual or mechanical changes. Learning invariances in a causal graph needs interventions, which matches developmental psychology's emphasis on experimentation. A causal agent could re-learn only the mechanisms that changed. The right formalization is still open.
- **Counterfactuals.** They improve data efficiency and performance, and support communicating about past experience among agents. Future RL should treat counterfactual reasoning *"as a critical component."*
- **Offline RL.** Learning a policy from a fixed log of trajectories requires inferring the consequences of actions that were never taken, which *"essentially makes it a counterfactual inference problem."* The mismatch between the logging policy and the new policy is a departure from i.i.d. Factoring knowledge into independent mechanisms may help.

**In plain words.** An agent that tries things and sees what happens is running experiments. An agent learning only from someone else's logs is back to correlations, which can mislead: maybe the expert only took that action when things were already going well.

### 7.6 Scientific applications

**The idea.**
- **Physics simulation.** Neural simulators can be faster than hand-built ones. When conditions are tightly controlled, strong OOD generalization may not be needed, but the simulator must be retrained if conditions change.
- **Personalized medicine.** Combining electronic health records with genetic data is promising, but a system trained on doctors' decisions in controlled settings may add little and *"may fail in surprising ways when deployed."* Reliable personalized treatment needs causal understanding.
- **Covid-19.** *"Causal mediation analysis helps disentangle different effects contributing towards case fatality rates when a textbook example of Simpson's paradox was observed."* *(Background, from the cited study by von Kügelgen et al.: comparing China and Italy early in the pandemic, Italy's overall case-fatality rate was higher, yet within every age group it was lower. Italy's cases skewed much older.)*
- **Exoplanets.** A planet passing in front of its star dims it slightly. In Kepler telescope data, instrument effects confound this signal. *"Shared patterns in measurement noise across stars light-years apart can be removed"*: stars that far apart can't share astrophysical signals, so what they share must come from the instrument. The method (named in the cited 2016 PNAS paper as **half-sibling regression**) *"lead to the discovery of 36 planet candidates, of which 21 were subsequently validated as bona fide exoplanets."* One of them, K2-18b, was later found to have water in its atmosphere, the first such finding for an exoplanet in the habitable zone.

**In plain words.**
- *Simpson's paradox:* a trend can reverse when you split data into groups. One country can do better in every age bracket and still look worse overall because more of its patients are elderly.
- *Half-sibling regression:* if a hundred unrelated stars all flicker at the same moment, the telescope is to blame, not the stars. Subtract the shared flicker, and what remains is each star's own behaviour, including a planet passing in front of it.

### 7.7 Multi-task learning

**The idea.** A multi-task system solves many tasks across environments, which share traits it can reuse. The paper proposes learning a shared generative causal model whose components satisfy SMS; causal models adapt faster to sparse interventions.

It answers an obvious objection: *"Why can't we just train a huge model that learns environments' dynamics…?"* With insufficiently diverse data, worst-case error on unseen shifts can stay arbitrarily high (§7.3). Bigger models may beat OOD benchmarks in the short term, but all generalization rests on assumptions, and causal assumptions are explicit and aligned with physics and cognition. A learner that ignores valid assumptions should do worse. ICM suggests decomposing knowledge into independent, recombinable pieces, which motivates modular architectures.

(The section heading also says "continual learning," but the body discusses only multi-task and meta-learning.)

**In plain words.** Learning to drive a truck after a car means adjusting a few skills, not starting over. Modular causal models would allow the same for machines, while a single giant model has to re-fit everything and has no guarantee about situations unlike its training data.

---

## 8. Conclusion and open problems

*Slide: "Open Problems Named by the Authors".*

The paper's four open problems:

1. **Learning non-linear causal relations at scale.** When are non-linear causal relations learnable? Which training frameworks scale? Do causal modules beat non-causal representations at generalization, re-use, and transfer on real tasks?
   *Plain words:* we don't yet know how to do this on large, realistic problems, or how much it really helps.

2. **Learning causal variables.** Current disentangled representations are fixed-size vectors with an arbitrary order of dimensions, so they can't handle, for example, a changing number of objects in a scene. Structured, modular representations should come from task-specific training, not only from autoencoding.
   *Plain words:* a scene can contain two objects or twenty, and current methods assume a fixed number of slots.

3. **Understanding the biases of existing deep learning.** Scale, augmentation, and self-supervision all improve robustness, but their separate contributions are hard to untangle. Testing strong generalization against defined sets of interventions could map out which inductive biases do what, and how pre-training choices affect transfer and robustness *"in a causal sense."*
   *Plain words:* we know these tricks help but not exactly against what. Test each against specific kinds of change.

4. **Learning causally correct models of the world and the agent.** RL rarely comes with clean, abstract states, so deriving causal variables from pixels and recovering their graph matters. Causal descriptions of both the agent and its environment would support robust, versatile model-based RL.
   *Plain words:* a robot has to work out from camera images what the meaningful things are, including itself.

---

## 9. Critique and clinical relevance

*Slides: "Critical Appraisal", "Why This Matters for Clinical Data". This section is journal-club discussion, not content from the paper.*

### 9.1 Strengths
- Unifies two communities with a shared vocabulary (SCMs ↔ representations).
- ICM and SMS are **concrete, testable inductive biases**, not vague appeals to "causality."
- Re-interprets familiar phenomena (SSL, adversarial examples, augmentation) in a way that actually explains them.
- Agenda-setting: much of the later identifiable-CRL literature traces back to it.

### 9.2 Limitations
- A **position paper**: no new experiments or benchmarks.
- Identifiability is acknowledged but not solved. Without some supervision, causal variables may be unrecoverable.
- SMS is a *hypothesis*. When do real shifts actually look sparse?
- Little guidance on evaluation: how would we *know* a learned variable is causal?
- The scaling argument (§7.7) predates the foundation-model era. Does scale already buy much of the promised robustness?

### 9.3 Clinical data

**The idea.**
- **Distribution shift is the default.** Multi-site networks (e.g., PCORnet or OMOP-based federations) are effectively **many environments**, with different populations, coding practices, and EHR vendors. Under SMS, a new site should change *a few* mechanisms (case mix, documentation habits), not the underlying physiology. Robust models should keep those apart. The paper's own warnings are directly relevant: the diabetic retinopathy detector affected by clinic conditions, and clinical systems that *"may fail in surprising ways when deployed."*
- **Variables aren't given here either.** Phenotypes, disease stages, and severity are *latent* variables built from codes, labs, and notes. Phenotyping is, in effect, a representation learning problem, and the paper's granularity point (§6.1) applies: coarse clinical variables may not have well-defined interventions.

**Discussion prompt:** in our own pipelines, which "variables" are we treating as given that are really *learned*, and possibly not causal? Which learned phenotypes would survive an intervention?

**In plain words.** Every hospital is a slightly different world, but the body works the same way in all of them. A good model should separate "how this hospital records things" from "how the disease works." And "has diabetes" in a dataset is often a definition someone built from billing codes and lab values, so it deserves the same scrutiny as any learned representation.

---

## 10. Key takeaways

1. **Seeing vs. doing.** Statistical models answer *"what will I observe?"* Causal models answer *"what happens if I act?"* Generalizing under shift needs the second.
2. **Modularity and sparse change.** The ICM principle says causal mechanisms neither influence nor inform each other. The SMS hypothesis says real-world changes touch only a few of them.
3. **Learning the variables.** Causal inference assumes the variables are given. Causal representation learning is about learning them from raw data, using sparse shifts and multi-environment data as supervision.
4. **A new lens on old puzzles.** SSL working only sometimes, adversarial fragility, and the value of augmentation and pre-training make more sense from a causal point of view.

---

## 11. Glossary

| Term | Precise meaning | Plain meaning |
|---|---|---|
| **i.i.d.** | Independent and identically distributed: all samples drawn independently from one fixed distribution. | Test data looks like training data. |
| **Distribution shift / OOD** | Test distribution differs from training distribution. | The world changed after you trained. |
| **Environment** | A circumstance (location, time, experimental condition) giving rise to a distribution shift; in causal terms, an intervention. | One "setting" the data came from. |
| **Intervention, $do(X = x)$** | Replace the assignment for $X$ (e.g., fix its value) and leave all others intact. | Actively setting something, instead of watching it. |
| **Hard / soft / uncertain intervention** | Set the function to a constant / change the function or noise / unknown target. | Force it / nudge it / something changed, not sure what. |
| **Counterfactual** | A query about what *would* have happened to a specific case under a different action, given what actually happened. Requires fixing the noise variables. | "What if *this* person had done otherwise?" |
| **SCM** | Assignments $X_i := f_i(\mathbf{PA}_i, U_i)$ over a DAG, with jointly independent noises. | A recipe saying how each variable is produced. |
| **Parents, $\mathbf{PA}_i$** | Direct causes of $X_i$ in the graph. | The things that directly affect $X_i$. |
| **Noise term, $U_i$** | Unexplained, independent source of variation for $X_i$. | Everything about this case the model doesn't spell out. |
| **DAG** | Directed acyclic graph. | A flowchart of causes with no loops. |
| **Causal Markov condition** | Given its parents, each variable is independent of its non-descendants. | Once you know a thing's direct causes, its more distant causes add nothing. |
| **Causal sufficiency** | No unobserved common causes of the modelled variables. | Nothing important is hidden. |
| **Causal (disentangled) factorization** | $\prod_i P(X_i \mid \mathbf{PA}_i)$, one factor per mechanism. | Knowledge stored as separate, swappable parts. |
| **Entangled factorization** | Any other valid factorization of the joint; its factors mix mechanisms. | Knowledge stored as one tangled lump. |
| **Mechanism** | A single conditional $P(X_i \mid \mathbf{PA}_i)$ / assignment $f_i$. | One cause-and-effect rule. |
| **Common cause principle** | Dependence implies a causal common cause that explains it away (possibly $X$ or $Y$ itself). | Every correlation has a causal story behind it. |
| **Confounder** | A common cause of two variables, possibly unobserved. | A hidden third factor creating a fake link. |
| **Markov equivalence (class)** | DAGs implying the same conditional independencies; they share a skeleton and immoralities. | Graphs that observational data can't tell apart. |
| **Immorality / v-structure** | $X_i \to X_j \leftarrow X_k$, with $X_i$ and $X_k$ not adjacent. | Two separate causes of a shared effect. |
| **Faithfulness** | Every independence in the data is implied by the graph. | No coincidental cancellations hide real links. |
| **Additive noise model** | $Y := f(X) + V$, $X \perp\!\!\!\perp V$. | Effect = function of cause + independent wobble; reveals direction. |
| **ICM principle** | Causal mechanisms are autonomous modules that neither influence nor inform each other. | Nature is built from independent, swappable parts. |
| **Kolmogorov complexity** | Length of the shortest program that outputs a given string. | How short the shortest description is. |
| **SMS hypothesis** | Small distribution changes act sparsely or locally in the causal factorization. | When the world changes, usually only one or two things change. |
| **Generic viewpoint assumption** | No accidental alignments between object, viewpoint, and lighting. | What you're looking at is independent of how you're looking at it. |
| **Causal representation learning** | Learning high-level causal variables (and their mechanisms) from low-level observations. | Going from pixels to meaningful things and how they affect each other. |
| **Identifiability** | Whether the true latent variables can be uniquely recovered from data (up to trivial transformations). | Whether the data contains enough clues to find the *real* answer. |
| **Disentangled representation** | Latents that correspond to separate generative factors. | One knob per real-world factor. |
| **Object-centric representation** | A representation built from separate object slots. | Describe the scene as a list of things. |
| **Causal / anticausal learning** | Predicting effect from cause ($X \to Y$) vs. cause from effect ($Y \to X$). | Forward along the arrow vs. backward against it. |
| **Analysis by synthesis** | Classify by checking consistency with a generative model of each class. | "Could this class really have produced this image?" |
| **Empirical risk minimization (ERM)** | Choose the model with lowest average loss on training data. | Do well on the examples you've seen. |
| **Min-max (worst-case) risk** | $\min_g \max_{P^\dagger \in \mathcal{E}}$ of the expected loss. | Be never terrible, rather than best on average. |
| **Offline RL** | Learning a policy from a fixed log of trajectories. | Learning to act only from someone else's records. |
| **Simpson's paradox** | A trend in aggregate data reverses within every subgroup. | Splitting into groups can flip the story. |
| **Half-sibling regression** | Remove noise shared with variables that can share *only* the noise. | Subtract what everything has in common to reveal what's unique. |

---

## 12. Where the slides and the paper differ

Checked against the paper's text. You may want to fix these on the slides before presenting.

| Slide | Slide says | Paper says |
|---|---|---|
| Semi-Supervised Learning | "A **meta-analysis** cited by the authors found SSL gains concentrated in anticausal tasks." | No meta-analysis. The prediction comes from theory (Daniušis et al.), and one study (Schölkopf et al. 2012) *"empirically corroborated these predictions."* Suggested wording: "Schölkopf et al. (2012) found empirical support: SSL gains appear mainly in anticausal tasks." |
| Science, Medicine and Multi-Task Learning | Exoplanets: "36 candidates, 21 later validated." | The text does say 36 and 21. But the paper's own reference for the validation (Montet et al. 2015) is titled "…validation of 17 planets." Worth checking before quoting 21. |
| Science, Medicine and Multi-Task Learning | Covid-19 Simpson's paradox "resolved by causal reasoning about age structure." | The paper only says causal mediation analysis disentangled the effects; it doesn't name countries or the direction. The China/Italy detail comes from the cited study (von Kügelgen et al.). |
| Science, Medicine and Multi-Task Learning | "learn new mechanisms without forgetting old ones" (continual learning). | "Continual learning" appears in the heading only; the body discusses multi-task learning and meta-learning. |
| Four Things We Might Want From a Model | Framed as four desiderata. | These are the paper's §II-A to §II-D: four *questions* (i.i.d. prediction, prediction under shift, counterfactuals, nature of data). Close enough, but not a named list. |
| Adversarial Examples and Robustness | "the causal predictor solves this min-max." | The paper sets up the min-max and expects causal models to be more robust, but frames this as motivation, citing the invariance literature; it adds that learning such models is "in some cases provably hard." |
| SMS in Pixel Space | "one finger moves and the object falls." | Matches; the body text specifies the object is a cube. |

---

## Further reading

- Schölkopf et al. (2021). *Toward Causal Representation Learning.* Proceedings of the IEEE, 109(5). arXiv:2102.11107
- Schölkopf (2019). *Causality for Machine Learning.* arXiv:1911.10500
- Peters, Janzing & Schölkopf (2017). *Elements of Causal Inference.* MIT Press
- Pearl (2009). *Causality.* Cambridge University Press
- Schölkopf et al. (2012). *On Causal and Anticausal Learning.* ICML
- Locatello et al. (2019). *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML
- Locatello et al. (2020). *Weakly-Supervised Disentanglement Without Compromises.* ICML
- Bengio et al. (2020). *A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms.* ICLR
- Arjovsky et al. (2019). *Invariant Risk Minimization.* arXiv:1907.02893
- Schölkopf et al. (2016). *Modeling Confounding by Half-Sibling Regression.* PNAS
- von Kügelgen et al. (2021). *Simpson's Paradox in COVID-19 Case Fatality Rates: A Mediation Analysis of Age-Related Causal Effects.* IEEE TAI
