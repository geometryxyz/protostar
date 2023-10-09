# Halo2+Protostar

A research implementation of [Protostar](https://eprint.iacr.org/2023/620) folding scheme for halo2-PSE. 

The goal of this project was to study the feasibility of implementing a folding scheme supporting circuits built using the halo2 API. 
In what follows, we detail our findings and provide ideas and suggestions that would help future efforts. 

## Report

### Why Folding?

It is important to remember that folding is only a performance improvement over existing SNARK recursion/accumulation techniques, 
and is limited to the case where the prover is iteratively proving the **same** circuit (IVC, or batch proving). 
The two main improvements are
- **Efficient Prover**: The prover performs fewer MSMs, there are no FFTs, and only needs to prove active constraints.
- **Smaller Recursive Verifier**: The circuit verifying the folding iteration has fewer MSMs and hashes, leading to fewer constraints in the recursive verifier circuit.

When standard recursion techniques are too slow, folding schemes can drastically speed up a specific application.
The resulting proof system can be strongly customized to ensure the overhead of proving each iteration is only a multiplicative factor slower than actually evaluating the iterated function.

Given the current state of proof systems, the efficiency of STARKs seems unbeatable by SNARKs for any sort of uniform computation.
This is due to the impressive research into small fields, custom hash functions, and better codes.
The benefits of SNARKs at this time are mainly their smaller proofs and efficient verifiers, 
though their provers are unfortunately limited by the requirements of large 256-bit fields.

Folding schemes seem like they can bridge this gap in prover performance, 
as they are able to take full advantage of the fact that the circuit is known in advance and is the same in every iteration. 


### Arithmetization Choice

Folding as described in Protostar is a generalization of Nova for R1CS and Sangria for standard PlonK, 
and can be adapted to any PlonK-ish arithmetization and even CCS.
As with any proof system, its performance will depend heavily on the chosen arithmetization, 
which in turn requires circuits that take full advantage of it. 

In practice, the arithmetization should still be provable by an efficient SNARK
in order to create succinct proofs of the entire computation. 
This means that the description of the accumulation Decider algorithm
should be describable using a somewhat efficient SNARK arithmetization like PlonK-ish. 
We therefore require that the set of constraints used in Protostar 
can be efficiently encoded into a uniform constraint system. 

Converting between arithmetizations will usually lead to a loss of semantic information about the original program/circuit. 
This is akin to converting source code into assembly.
Since Protostar is able to take advantage of more properties of the circuit, 
we would want to preserve this information to design an optimal prover.
Moreover, knowing what types of operations are required and **how many times they are used** in the circuit
can help guide the design of the prover. 

Overall, a semantically richer PlonK-ish arithmetization still seems like the best approach to build efficient folding provers. 
The folded circuit should then also be designed to take advantage of the broader design space offered by Protostar,
while also taking into account the arithmetization of the decider. 

Unfortunately, this also makes it a lot more complicated to consider what the exact arithmetization would look like, 
since there are many more parameters to take into account, some of which are not considered by existing PlonK-ish DSLs like halo2. 
These may require large modifications to the DSL, and it is important to consider whether these are actually beneficial. 

A good starting point for this evaluation would be to first consider a R1CS implementation of the circuit and benchmark with Nova. 
From this baseline, we would want to consider
- Number of witness variables, and time to commit to them.
- Total witness generation time. 
- Number of degree 2 constraints, and time to prove them.
- Number of operations, and for each type how many variables they require.
- Decider complexity.

From there, a Protostar implementation could be defined to specifically target that circuit.
- Minimize number of witnesses by including custom gates for highly algebraic constraints (hashes, ECC).
- Measure impact on recursive verifier when including additional committed columns in the IOP.
- Decide which variant of the compressed-verifier technique makes sense to handle higher degree gates. 
- Consider the impact of including lookup arguments.
- Real-world cost of implementing these primitives.

This becomes more of an engineering problem than a cryptography one. 
A fully featured library allowing customization of all these parameters would be incredibly complex,
which in turn make it harder to extend or audit.
The main reason to consider Protostar over Nova would be speed, 
but even then only a couple of these optimizations might actually make sense. 
In this case, we would want to ensure the prover isn't wasting time on features which are not relevant to the circuit. 
Moreover, ensuring fast witness generation can be hard when we lose the ability to reason about variables as actual computed values 
rather than generic field elements. 

### Circuit design differences

We highlight the main differences that could affect circuit design when optimizing for prover folding efficiency. 

#### Trace Layout 

Since Protostar does not rely on FFTs, it is not necessary to limit the length of columns
by splitting the trace into multiple chunks. 
In fact, due to the asymptotics of the Pippenger algorithm for MSMs, longer columns are more efficient to commit to.
Reducing the number of columns also strictly improves the overhead induced by the recursive folding verifier since each column must be hashed to the Fiat-Shamir transcript and induces an additional scalar multiplication. 

A "single column per round" layout strategy would also ensure during the ErrorCheck phase of folding,
values belonging to the same constraint are located next to each other in RAM. 
This could lead to better memory access pattern and more efficient parallelism.

Accessing almost all witness values via rotations may require some modifications to the decider SNARK, 
as these may not be supported natively by the proof system (for example if the decider is implemented with Sumcheck).
During the deciding phase, it is possible to "unzip" the single column of length $n$ into $k$ smaller ones of size $n/k$. 
This can minimize the need for as many shifts, and potentially speed up the PCS phase of the decider.
Even though $k$ is not relevant to the folding prover, we can ensure that constraints are applied to regions of the trace with "alignment $k$". 
This is similar to physical CPUs where incorrect alignment can lead worse performance or even crashes. 

#### Constraints 

The complexity of proving constraints is very different in Protostar.

While a PlonK prover would determine the degree of a constraint by treating fixed and witness columns of the trace as variables, 
Protostar considers fixed columns as constants while treating challenges as variables. 
When the polynomial is evaluated in a given row during proving, 
the values coming from fixed columns are the actual ones defined by the circuit.

Generally, fixed columns are used to specialize a generic constraint for validating multiple types of operations.
For example, 
the standard PlonK constraint $q_L \cdot w_L + q_R \cdot w_R + q_O \cdot w_O + q_M \cdot w_L \cdot w_R + q_C = 0$
can be treated as either an addition, multiplication, inversion, or boolean check constraint depending on the fixed values $q_L, q_R, q_O, q_M, q_C$. 
The constraint can be more efficiently evaluated if the prover is told which of these operations is actually being proved.
Moreover, we will see later that linear constraints (when considering the "folding degree") are "free". 
Therefor when $q_M = 0$, the evaluation of the constraint $q_L \cdot w_L + q_R \cdot w_R + q_O \cdot w_0  + q_C$ could be ignored when folding. 

Selectors take on a much more important role in Protostar, 
since the expression they select can be completely cancelled out when they are turned off. 
While they ultimately need to be encoded as fixed columns in their PlonK-ish arithmetization in the decider phase, 
they are only used during folding to indicate which constraints to evaluate.
We can also use them to enforce linear independence of a group of constraints. 
For example, consider a group of constraints $G_1, \ldots, G_m$ such that only one of them is active at any given time. 
We can write the sum $\sum_{j=1}^m s_j \cdot G_j$ using pre-processed selectors $s_1,\ldots,s_m \in \{0,1\}$ such that $\sum_{j=1}^m s_j = 1$. 
This removes the requirement that 

When challenges are considered as variables contributing to the degree of a constraint, 
we need to ensure that the expressions they are part of are written in a way that does not accidentally blow up the degree. 
For example, a common usage of them is for batch zero-testing, where we might write a constraint like $w_0 + c \cdot (w_1 + c \cdot (w_2 + c \cdot w_3))$. 
The degree here would be equal to 4, due to the $c^3 \cdot w_3$ term in the expansion $w_0 + c \cdot w_1 + c^2 \cdot w_2 + c^3 \cdot w_3$.
The trick is to let the verifier define intermediary variables $c_0 = 1$, $c_i = c\cdot c_{i-1}$ for $i=1,2,3$. These can then be considered as independent challenges which all have degree 1, ensuring that the constraint $w_0 + c_1 \cdot w_1 + c_2 \cdot w_2 + c_3 \cdot w_3$ has total degree 2. 

#### Gates

In halo2, a gate is an abstraction over a collection of linearly independent constraints which must all be satisfied in a given row. 
A chip however may apply multiple gates to a region of the trace, and the gates may have been defined with the assumption that they will always be applied in a certain order. 
This is achieved by using "rotation" of columns, in order to reference values from neighboring rows.

This level of abstraction unfortunately discards a lot of valuable information about the potential relationship between different gates. 
An optimized Protostar prover would consider an entire region where the variables require overlapping constraints. 
In most cases, these variables are only used within this specific region as they represent variables internal to the scope of a specific operation.
They need to be "interpolated" (explained later in the implementation section) up to the degree of the constraint they participate in.
Therefor we can ensure each variable is only interpolated once, and only up to its required degree.

As we explain in the next section, permutation are a zero-cost abstraction when considering only the folding prover.
We can exploit this property to completely remove the need to consider rotations during the decider. 
If we assume that each constraint only constrains up to $k$ variables,
the trace as viewed from the perspective of the decider would correspond to exactly one constraint per row.
To ensure the folding prover does not pay the cost of committing to the same variable multiple times, 
we can pre-process the SRS during the prover key generation to sum up all the basis elements of the variables 
which are constrained to be equal by the permutation argument. 
Taking this idea even further, this means that the order of the constraints for the decider 
can be completely independent of the order we apply them in the folding prover. 

This can be beneficial for a Sumcheck-based prover which can also take advantage of the sparsity of selectors,
as long as all constraints of the same type are grouped together. 

#### Permutations 

Permutations are interpreted as strictly linear constraints by Protostar,
so they have zero performance overhead for the folding prover.
It is therefore advantageous to make as much use of them. 
Large permutation arguments can have a big impact on the decider prover 
due to the fact that the grand-product column computation is hard to parallelize.
This can be mitigated by using the GKR protocol as in Lasso.


#### Lookups

The logUp technique for tackling lookup arguments in SNARKs is a fundamental building block of Protostar,
as it can remove the dependency on the table size for the runtime of the folding prover. 
Lookups several types of useful primitives for writing more efficient circuits, 
though these use-cases are often special case of the following:

- **Memoized function lookup**: For a given function $y \gets f(x)$, a table stores the evaluations $(x,y)$ for all possible inputs $x$. 
- **ROM/RAM**: Simulate access to read-only/random-access memory inside a circuit by proving the correctness of each access over a sorted permutation of each access.
- **Set membership**: A value belongs to a given set of values, for example a set of fixed public keys.
- **Range constraints**: A value belongs to a certain range of valid values, for example $x \in [0,2^16]$.
- **Shuffle constraints**: Two regions of a trace contain the same values, albeit in a different order.
- **Sharing values between columns**: Two columns from different traces contain the same rows, for example two columns from consecutive IVC iterations. 

All of these primitives can be built using the same basic logUp technique, but their implementations can be drastically different, 
in particular in the way the witnesses would be laid out. 
For example:
- Memoized functions need to consider the index of $x$ in the table. Rather than inefficiently looking it up during proving,
it would be possible to compute the index by looking only at $x$.
- Shuffle constraints over values and rows do not require a commitment to the multiplicities vector. 
- Lookups may be performed over the single or multiple columns, and over fixed or committed columns. 
- For range constraints, an implementation with a decider based on Sumcheck could leverage the fact that the column containing all values of a range does not need to committed to. 
- Since constraints can be selectively applied to different parts of the trace, there is no need to share values between columns as is often the case with SNARKs. 
- Many of these arguments could be batched together using selectors, therefor minimizing the number of commitments sent by the folding prover.
- The optimization proposed in the Protostar paper may only make sense for certain of the above use cases, but not all. 

Only certain of the above primitives may make sense for a specific circuit, and may only make sense if implemented efficiently.
Especially for witness generation, it is important for the proof system to be able to allocate the variables efficiently. 
A solution would be to integrate these primitives directly as part of the circuit builder, 
while also describing a corresponding efficient (and possible customizable) proving strategy for both folding and deciding.


### Prover Design 

The implementation in this repo did not manage to implement all the optimizations detailed in the section above.
We opted instead for a simpler approach in order to understand what the structure of an optimized folding prover could look like,
given enough engineering resources.

#### Accumulator

A Protostar accumulator represents both the instance and witness for a satisfying "relaxed" PlonK-ish circuit. 
In particular, it contains all messages sent by both the prover and the verifier over the course of the IOP and folding protocol.

The structure is approximately 
- `[Instance]`: Instance values know by both parties
- `[Advice]`: Committed advice columns sent by the prover 
- `[Challenges]`: Challenges usable as variables in gate equations
- `[Lookups(r, [thetas], m, g, h)]`: For each lookup argument, the required challenges and witness columns used by logUp
- `Beta, BetaError`: Commitments to the "powers-of-beta" column used for the compressed verifier, along with an error vector for proving its correctness.
- `[y]`: Powers of a challenge $y$ to ensure linear independence of all constraints.
- `error`: Evaluation of all constraints, initialized to 0 for initial transcripts.

The fields of this structure are the same for both P and V, though the prover would represent witness columns as the triple `(column, commitment, blind)`, while the verifier would only consider the `commitment`.
Except for the error terms (`BetaError` and `error`), all fields are homomorphic.

The folding prover/verifier make no distinction between a "fresh" accumulator or one that was the result of a previous folding.
The prover accumulator can be "decided" be ensuring all commitments are correct, and recomputing the errors. 
Theoretically, this decider could then be implemented as a SNARK using any appropriate proof system. 

#### Gates and Constraints

In order to unify the description of constraints over both gate and lookup constraints, we chose to implement an extended `Expression` where leaves 
are references to columns (with potential rotations).
The goal was also to make it reusable across different situations (row-wise, interpolated over two accumulators, over cosets, etc.) though 
this proved to be hard to generalize over as many types.
The actual shape of a constraint is given by
$$
G(f_1, \ldots, f_{m_1}; w_1, \ldots, w_{m_2}, c_1, \ldots, c_{m_3})
$$ where
- $f_j$ represents a variable from a constant column (either `Selector` or `Fixed`)
- $w_j$ represents a variable from a witness column (either `Instance`, `Advice`, `Beta` or `Lookup(m,g,h)`)
- $c_j$ represents a variable from a verifier challenge (either from `Challenges`, `[y]` or `Lookup(r, [theta])`)


For simplicity, we reduced all constraints $G_1, \ldots, G_m$ over all gates and all lookup arguments as a single one (for a given row $i$)
$$
G_i = \sum_{j=1}^m y_j \cdot G_{j,i},
$$
where 
- $y_j$ is the $j$-th power of a verifier challenge $y$
- $G_{j,i}$ is the partial evaluation of the constraint $j$ with the values of the constant columns at row $i$. That is 
$$
G_{j,i}(w_1, \ldots, w_{m_2}, c_1, \ldots, c_{m_3}) = G_j(f_{1,i}, \ldots, f_{m_1,i}; w_1, \ldots, w_{m_2}, c_1, \ldots, c_{m_3})
$$

We consider an IOP transcript to be valid if 
$$
G_i(w_{1,i}, \ldots, w_{m_2,i}, c_1, \ldots, c_{m_3}) = 0, \quad \forall i \in [n].
$$


#### Compressed Verifier

We implemented a simple version of the "compressed verifier" optimization from the Protostar paper. 
Rather than sending two commitments with powers of $\beta$ with size $\sqrt{n}$, we send a single one of size $n$. 
The vector $\vec{\beta}$ must satisfy the constraint $\beta \cdot \beta_i - \beta_{i+1} =0$ for all $i \in [n]$, and $\beta_0 - \beta =0$. 
The reason why this optimization was skipped was due to the commitment scheme parameters only supports committing to vectors of size $n$. 

This allows us to consider the single constraint $G_i' = \beta_i \cdot G_i$, thereby reducing the need for the prover to send commitments to the error evaluations.
An accumulator is considered valid if $\mathsf{error} = G'$ where $G' = \sum_{i\in [n]} G_i'$ is evaluated over all variables contained in the accumulator.

In order to validate the constraints induced by the $\vec{\beta}$ column, we noticed that during folding,
the verifier is able to compute it by themselves by linearly combining the two previous error vector commitments with the commitments to the beta columns.
This introduces extra scalar multiplications, while reducing the number of messages in the transcript.

#### ErrorCheck

The ErrorCheck is a core component of the folding prover and is the one where the most optimizations are applicable. 
However, many of these optimizations were complex to implement in a simple way, so we chose the following simpler approach:

We consider as input only the constraint $G' = \sum_{i\in [n]} G_i'$ as defined above. 
As an `Expression` represented as a binary tree, where nodes are operations `+,*,-` and leaves represent variables (constant, witness or challenge). 
In this case, the leaves are actually
- References to columns in the proving key for constant columns $f$
- Pairs of references to the same column from two different accumulators $[w^{(0)}, w^{(1)}]$
- Pairs of references to that same challenge from two different columns $[c^{(0)}, c^{(1)}]$

Let $d$ be the total degree of the constraint $G'$. 
We consider $e_i(X) = G_i'(w_{1,i}(X), \ldots, w_{m_2,i}(X), c_1(X), \ldots, c_{m_3}(X))$, where the variables correspond to 
- $w_{j,i}(X) = (1-X)\cdot w_{j,i}^{(0)} + X\cdot w_{j,i}^{(1)}$ for witness columns
- $c_{j}(X) = (1-X)\cdot c_{j}^{(0)} + X\cdot c_{j}^{(1)}$ for challenge variables. 

Intuitively, this corresponds to defining a line passing between the values from accumulators. Notice that setting $X=0,1$ returns the values from the respective variables.
Since $G_i'$ has degree $d$, and each variable is a linear polynomial, the degree of $e_i(X)$ is also $d$. 
We define $D = \{0,1,\ldots,d\}$ as an interpolation set, such that $f(D) = \{f(0), \ldots, f(d)\}$. 
The goal is to evaluate $e_i(D)$ for all rows $i$, and then compute $e(D) = \sum_i e_i(D)$.

In order to evaluate $e_i(D)$, we first evaluate all $w_{j,i}(D)$. This can be done using only field additions since for any line $p(X) = (1-X)\cdot p_0 + X\cdot p_1$, we have $p(X) = p_0 + X\cdot(p_1 - p_0)$ and therefore $p(j) = p(j-1) + \Delta_p$ where $\Delta_p = p_1 - p_0$. 
Next, we simply need to evaluate 
$$
e_{i}(j) = G_i'(w_{1,i}(j), \ldots, w_{m_2,i}(j), c_1(j), \ldots, c_{m_3}(j)),\quad \forall j \in D.
$$

Note that the evaluations of the challenges are reused across all rows, so they only need to be interpolated once. 

Finally we can use Lagrange interpolation to compute the coefficients of $e(X)$ form $e(D)$.
Note that $e(0)=e_0$ and $e(1)=e_1$ correspond to the evaluations of the error term of both respective accumulators.
The prover sends the quotient 
$$
e'(X) = \frac{e(X) - (1-X)\cdot e_0 - X\cdot e_1} {(1-X)\cdot X},
$$

The verifier then samples $\alpha$ and computes the linear combination of all witness column commitments and challenges as $w' = (1-\alpha)\cdot w^{(0)} + \alpha \cdot w^{(1)}$ and $c' = (1-\alpha)\cdot c^{(0)} + \alpha \cdot c^{(1)}$, and computes the new error
$$
e' = (1- \alpha)\cdot \alpha \cdot e'( \alpha) + (1- \alpha)\cdot e_0 + \alpha\cdot e_1
$$

### Limitations


#### IVC support 

In its current form, halo2 does not support IVC out of the box and requires integration with external libraries like `snark-verifier`.
We were not able to implement this in time, though it's implementation can be adapted from the native verifier included in the library.

#### Decider support

Initially, the goal was to reuse the PlonK implementation as-is to implement the decider. Unfortunately we realized this would have required essentially rewriting much of it from scratch, and opted instead to leave it unimplemented rather than hacking together existing code. 

Moreover, the structure of the decider is fundamentally better suited for a Sumcheck based prover as we have argued earlier. 
Conceptually, the techniques used for error check are very similar to those that would be used for a Sumcheck prover, therefore it would be nice to reuse some of the structure for a complete implementation. 

#### Unoptimized ErrorCheck

Given a set of constraints $\{G^{(j)}\}$ and a grand constraint $G = \sum_j y_j \cdot G^{(j)}$, we can compute individual evaluations $e^{(j)}(D_j) = \sum_i G^{(j)}(D_j)$ by invoking independent ErrorCheck computations. 
Each constraint $G^{(j)}$ need only to be evaluated over $\deg(G^{(j)})+1$ points. 

We can then interpolate these to obtain $e^{(j)}(X)$. To obtain the final error polynomial $e(X) = \sum_j y_j(X)\cdot e^{(j)}(X)$, we can perform the multiplication by the challenge $y_j(X)$ over the coefficient domain. 
This has negligible cost for the prover, and allow it to save one additional evaluation.

Moreover, if the prover caches the intermediary errors $e^{(j)}$ for each constraint $\{G^{(j)}\}$, it can avoid evaluating $e^{(j)}(0), e^{(j)}(1)$ saving two evaluations of the constraint. 

If a constraint is given as $G_j = s_j\cdot G_j'$ where $s_j$ is a selector, the prover can check in each row if $s_{j,i} = 0$ and skip the evaluation entirely. Otherwise it simply evaluated $G_j'$ knowing that $s_{j,i}=1$. 

#### Parallelization

Many loops are missing parallelization support, though thanks to the data-parallel nature of the folding prover subroutines, these should be rather straightforward to implement. 

For ErrorCheck, the naive way of parallelizing over all rows could lead to each thread having very different work loads if the selector checking optimization is used. 
Indeed, consider the case of two threads and a circuit containing only arithmetic gates in the first half, and ECC operations in the second one. 
The second thread would likely take much more time since ECC constraints are a lot more costly to evaluate. 
To handle this efficiently, it may be useful to rely on more complex multithreading primitives like "work stealing" like in Rayon. 
This would allow the prover to adapt to varying circuit sizes. 

### Conclusion

We think there are two real-world situations where one might want to currently implement Protostar.
We explain how they both would be implemented with significantly different approaches.

#### Optimal performance

Given the enormous amount of flexibility offered by Protostar, it is hard to imagine a single implementation supporting all possible configurations. 
More importantly, whether or not to include these optimizations depends heavily on the circuit used.
While it is always possible to estimate the impact the cost/benefit of including certain primitives, they come at a tradeoff over:
- Number of cells to commit to
- Recursive verifier size and time to prove 
- Constraint evaluation time
- Real-world engineering cost and code complexity

When compared to Nova, many of the advantages of an optimized Protostar implementation are already available by construction:
- Single column layout
- Free (+unlimited) linear combinations
- Free permutations
- Only active constraints are evaluated at compile-time
- Linear-time decider (Spartan)

The following is missing though:
- Lookups
- Custom gates

There is technically nothing preventing these techniques from being implemented in Nova,
since they can already be done in a single column layout with Protostar. 
With a concrete circuit to optimize for, 
a specific application of these could be considered if there is a clear need to improve on current performance. 
By building specific extensions with clear use cases, it would be easier to share them across circuits, 
while allowing circuit developpers to only activate those relevant their application.
From an engineering perspective, building these proof-system-level optimizations would be guided by benchmarks and profiling.

Optimal folding requires much more customization at the proof system level. 
Implementing all permutations to accommodate all use case can lead to a system that is optimal for nobody.
Therefore, in case where performance is the critical to the application (VMs for example), 
and where there is the option of designing the proof system for very specific circuits,
it would make sense to build these abstraction on top of an exisiting simpler scheme.
This type of development is closer to STARKs, where each VM team has built their own prover stacks which are tailored exactly to their computation model. 


#### Backwards compatibility 

We cannot ignore the fact that many circuits today are already written in halo2, and there would be a huge cost associated to rewriting it all in Nova and then figuring out how each optimization would fit it. 
Moreover, there are several existing circuits (for example zkEVM) for which folding would improve performance.

Our implement focused on **full** backwards compatibility with the halo2 API, but this proved to limit the optimizations that could be applied from the paper.
We suggest some small (but potentially breaking) changes that could be applied to the existing API to make folding more efficient:
- Forcing every `Gate` to have a queryable binary `Selector` ensuring only active gates are evaluated during folding. 
  - These should be treated separately from the `Expression` they select, though they can be treated as `Fixed` columns by the decider and re-multiplied at that point.
- Consider implementing a new `Layouter` which would focus on:
  - Limiting the number of columns sent by the prover.
  - Merging multiple fixed table columns into a single one.
- Allowing variable-sized and sparse columns could speed up commitment time.
  - Remove all padding requirements.
  - Enabling support for `sqrt(n)` compressed-verifier strategy.
  - Lower memory requirements for `Instance` columns.
- Specifying the power of a challenge inside an expression to prevent degree blow-up. 
- Build up more lookup primitives based on logUp. 
  - Distinguish fixed and online lookups to apply different proving strategies. 
  - Shuffles using fixed multiplicities columns.
- Augmenting the `Circuit` API to support IVC would 

Note that not all the above points are required, though some of them may necessitate larger architectural changes that would be break compatibility with exisiting circuits. 
These types of changes should be discussed by different members of the community to ensure compatibility across different projects. 


Moreover, a Sumcheck prover seems like a better choice for implementing the decider, due to its similarity with the folding prover.
A Sumcheck-based prover would be useful on it's own without folding,
as it would enable faster proving of standalone circuits as well.
- The lack of degree bound in Sumcheck decouples the need to fit this specific requirement inside of the `ConstraintSystem` and provides more flexibility in terms of gate design. 
- Native Sumchecks required for logUp do not require commitments to "running product/sum columns", whose computation is inherently sequential. 
- Selector optimizations can still be applied as long as gates of the same type are adjacent. 


Overall, a generic implementation of Protostar which takes advantage of the many optimizations available in folding will require changes to the way circuits are defined, and a large rewrite of most of the proof system. 
This will require coordination between different groups and members of the community and to take into account the requirements of each project.

## Minimum Supported Rust Version

Requires Rust **1.56.1** or higher.

Minimum supported Rust version can be changed in the future, but it will be done with a
minor version bump.

## Controlling parallelism

`halo2` currently uses [rayon](https://github.com/rayon-rs/rayon) for parallel computation.
The `RAYON_NUM_THREADS` environment variable can be used to set the number of threads.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
