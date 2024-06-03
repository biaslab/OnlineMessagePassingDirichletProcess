# Online Structure Learning with Dirichlet Processes through Message Passing
*By Bart van Erp, Wouter W. L. Nuijten and Bert de Vries*

---
**Abstract**
Generative and probabilistic modeling is crucial for developing intelligent agents that can reason about their environment.
However, designing these models manually for complex tasks is often infeasible due to their complexity.
Structure learning addresses this challenge by automating model creation based on sensory observations.
Central to structure learning is Bayesian model comparison, which provides a principled framework for evaluating models based on their evidence, enabling model reduction and expansion.
This paper focuses on model expansion and introduces an online message-passing procedure using Dirichlet processes, a prominent prior in non-parametric Bayesian methods.
Our approach builds on previous work by automating Bayesian model comparison through message passing and variational free energy minimization.
We derive novel message-passing update rules to emulate Dirichlet processes, offering a flexible and scalable method for online structure learning.
Our method generalizes to arbitrary models and treats structure learning just like state estimation and parameter learning.
The experimental results validate the effectiveness of our approach on the infinite mixture model, highlighting its benefits over existing methods.
This paper contributes a modular and generic approach for online structure learning, enhancing model accuracy and managing complexity over time.


---
This repository contains all experiments of the paper.

## Installation instructions
1. Install [Julia](https://julialang.org/)

2. Start a Pluto.jl server
```sh
>> make pluto
```


# License

[MIT License](LICENSE) Copyright (c) 2024 BIASlab
