### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 7ce1bec4-f238-407e-aefb-c633ee2fadd5
begin
	using PlutoUI

	md"""
	# Markov Decision Processes
	##### Julia Academy: _Decision Making Under Uncertainty with POMDPs.jl_

	An introduction to MDPs using [`QuickPOMDPs`](https://github.com/JuliaPOMDP/QuickPOMDPs.jl) that's part of the `POMDPs.jl` ecosystem.

	-- Robert Moss (Stanford University) as part of [_Julia Academy_](https://juliaacademy.com/) (Github: [mossr](https://github.com/mossr))
	"""
end

# ╔═╡ 8477c27b-ea40-4c59-b1ca-da8b641eb884
using POMDPs          # for MDP type

# ╔═╡ ae71c096-311f-4da5-8a6c-3829242af1f9
using POMDPModelTools # for SparseCat distribution

# ╔═╡ d91bcc7c-0941-4d6c-b695-026c33f67329
using POMDPPolicies   # for Policy type

# ╔═╡ 232d7d72-94f8-4dfe-abdf-2b6e712847f7
using QuickPOMDPs     # for QuickMDP

# ╔═╡ e1850dbb-30ea-4e1e-94f7-10582f89fb5d
using Parameters, Random

# ╔═╡ 43a7c406-3d59-4b80-8d34-7b6119e9c936
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# ╔═╡ 142f0646-541e-453b-a3b1-4b8fadf709cc
using DiscreteValueIteration

# ╔═╡ 80866699-58e9-4c32-a440-c5433c56a0ad
using Reel

# ╔═╡ 50f78e55-0433-438e-aaee-46c34eba8ba5
using TabularTDLearning

# ╔═╡ 79ad2ccc-615e-4b02-9f9d-cf29aaabe7fc
using MCTS

# ╔═╡ 73126c4f-108e-4e09-8e37-82b4dbb4ffb5
using D3Trees

# ╔═╡ 6e6d983c-7cb3-4f85-ac0c-f1daf8ee3fee
using POMDPSimulators

# ╔═╡ 720fe4d3-5801-49b4-a7b9-4923be051220
using ColorSchemes, Colors

# ╔═╡ 012cb4ad-7e30-477b-9d30-0bc020f12606
using Latexify

# ╔═╡ db0265cd-ebe0-4bf2-9e70-c0f978b91ff6
md"""
## Lecture Outline
- Markov Decision Process (MDP) Definition
- Grid World Problem
  - State Space
  - Action Space
  - Transition Function
  - Reward Function
- Solution Methods
    - Offline Algorithms
        - Value Iteration
        - Q-Learning
        - SARSA
    - Online Algorithms
        - Monte Carlo Tree Search (MCTS)
- Simulations
    - Rollout Simulator Visualization
"""

# ╔═╡ faf0ca6e-693d-4217-9e3c-e9e60339d416
md"""
## Markov Decision Process (MDP)

A Markov decision process (MDP) is a 5-tuple consisting of:

$\langle \mathcal{S}, \mathcal{A}, T, R, \gamma \rangle$

Variable | Description | `POMDPs` Interface
:----------- | :-------------- | :------:
$\mathcal{S}$ | State space | `POMDPs.states`
$\mathcal{A}$ | Action space | `POMDPs.actions`
$T$ | Transition function | `POMDPs.transition`
$R$ | Reward function | `POMDPs.reward`
$\gamma \in [0,1]$ | Discount factor | `POMDPs.discount`
"""

# ╔═╡ a84c5d59-ebc6-465e-8d92-87618b5712f0
md"""
- State space $\mathcal{S}$: `POMDPs.states`
- Action space $\mathcal{A}$: `POMDPs.actions`
- Transition function $T$ (sometimes called $P$): `POMDPs.transition`
- Reward function $R$: `POMDPs.reward`
- Discount factor $\gamma \in [0, 1]$: `POMDPs.discount`
"""

# ╔═╡ 8d30248e-b3c3-4f37-8296-392898790283
md"""
State spaces $\mathcal{S}$ and action spaces $\mathcal{A}$ represent all possible states an agent can be in and actions an agent can take. These spaces can either be _discrete_ (i.e., a finite number of possible values) or _continuous_ (i.e., infinite possible values).

In this notebook we only consider **_discrete_** states and actions.

We refer to the _transition function_ and the _reward function_ as "_the model_".
"""

# ╔═╡ 5e9b28a4-50b9-4c44-8d3c-48eb72bda3a5
md"""
## Grid World Problem (MDP)
In the _Grid World_ problem, an _agent_ moves around a grid attempting to collect as much reward as possible, trying to avoid negative rewards.
"""

# ╔═╡ e2a84ebf-a259-43c1-b512-f6c6b6e02d14
md"""
### Environment Parameters (setup)
First we set some parameters that help us define the Grid World environment (these could be global variables, but we choose to consolidate them to a single `GridWorldParams` structure).

These parameters defines the _size_ of the grid, a _null state_ for convenience, and the probability of transitioning to the chosen cell $p_\text{transition}$.
"""

# ╔═╡ 31ae33aa-5f25-4cd8-8e63-8e77c2233208
md"""
### States

A state $s$ in the Grid World problem is a discrete $(x,y)$ value in a $10\times10$ grid.
"""

# ╔═╡ b83aceeb-4360-43ab-9396-ac57a9416791
struct State
	x::Int
	y::Int
end

# ╔═╡ 07846f69-2f7a-4e12-9f4b-6fed8659e9ed
md"""
#### State space
The state space $\mathcal{S}$ for the Grid World problem is the set of all $(x,y)$ values in the $10\times10$ grid, including a null state at $(-1, -1)$.
"""

# ╔═╡ 99acb099-742c-4d13-abd8-c588217e4466
md"""
> **Note**: type `\scrS` then `<TAB>` to generate `𝒮` (example of LaTeX-style unicode characters).
"""

# ╔═╡ 581376af-21eb-4cc8-91af-7b671ebf4e71
md"""
We also define the `==` function so we can directly compare `State` types.
"""

# ╔═╡ c1d07fca-1fbd-4450-96b1-c829d7ad8306
Base.:(==)(s1::State, s2::State) = (s1.x == s2.x) && (s1.y == s2.y)

# ╔═╡ c092511d-c2e7-4b8c-8104-b4b10893cb02
@with_kw struct GridWorldParameters
	size::Tuple{Int,Int} = (10, 10)   # size of the grid
	null_state::State = State(-1, -1) # terminal state outside of the grid
	p_transition::Real = 0.7 # probability of transitioning to the correct next state
end	

# ╔═╡ 13dbf845-14a7-4c98-a1db-b3a83c9ce37c
params = GridWorldParameters();

# ╔═╡ 4a14aee4-12f1-4d55-9532-9b88e4c465f8
𝒮 = [[State(x,y) for x=1:params.size[1], y=1:params.size[2]]..., params.null_state]

# ╔═╡ dcfc1975-04e8-4d8e-ab46-d1e0846c071e
md"""
### Actions

The possible actions $s$ are movements in the cardinal directions, using Julia's built-in `@enum` macro.
"""

# ╔═╡ bcc5e8a3-1e3a-40cf-a306-13599a4952ac
@enum Action UP DOWN LEFT RIGHT

# ╔═╡ 52b96024-9f52-4f07-926b-2297ed7dd166
# create policy grid showing the best action in each state
function policy_grid(policy::Policy, xmax::Int, ymax::Int)
    arrows = Dict(UP => "↑",
                  DOWN => "↓",
                  LEFT => "←",
                  RIGHT => "→")

    grid = Array{String}(undef, xmax, ymax)
    for x = 1:xmax, y = 1:xmax
        s = State(x, y)
        grid[x,y] = arrows[action(policy, s)]
    end

    return grid
end

# ╔═╡ d66edb3a-7ccc-4e75-8d42-8d5b1ff5afbb
md"""
#### Action space
The action space $\mathcal{A}$ is made up of all possible actions. Note, `\scrA<TAB>` produces 𝒜.
"""

# ╔═╡ bc541507-61db-4084-9712-1c57d139e17f
𝒜 = [UP, DOWN, LEFT, RIGHT]

# ╔═╡ b2856919-5529-431b-8025-0b7f3f3081b0
md"""
For convenience, we use a dictionary to map actions to their movements applied to states. We overload the `+` operator from `Base` to define how to add two states together, used in the `transition` function below.
"""

# ╔═╡ 1303be2a-d18c-44b0-afb9-06a6b4ce5c08
begin
	const MOVEMENTS = Dict(UP    => State(0,1),
						   DOWN  => State(0,-1),
						   LEFT  => State(-1,0),
						   RIGHT => State(1,0));

	Base.:+(s1::State, s2::State) = State(s1.x + s2.x, s1.y + s2.y)
end

# ╔═╡ 268e2bb2-e6e2-4198-ad83-a93fcfa65b80
md"""
### Transition Function
The dynamics to transition the agent live in the transition function $T(s^\prime \mid s, a)$. The transition function returns a **distribution** over next states $s^\prime$ given the current state $s$ and an action $a$.
"""

# ╔═╡ 148d8e67-33a4-4065-911e-9ee0c33d8822
md"We define a boundry helper function to ensure the agent stays within the grid."

# ╔═╡ 49901c66-db64-48a2-b122-84d5f6b769db
inbounds(s::State) = 1 ≤ s.x ≤ params.size[1] && 1 ≤ s.y ≤ params.size[2]

# ╔═╡ 51796bfc-ee3c-4cab-9d58-359608fd4106
md"""
### Reward Function
The reward functions $R(s)$ and $R(s,a)$ return the rewards for any given `State`. Note, certain problem formulations may use $R(s)$ or $R(s,a)$, or even $R(s,a,s')$ to compute the rewards. The Grid World problem only cares about $R(s)$.
"""

# ╔═╡ f7814a66-23c8-4782-ba06-755397af87db
function R(s, a=missing)
	if s == State(4,3)
		return -10
	elseif s == State(4,6)
		return -5
	elseif s == State(9,3)
		return 10
	elseif s == State(8,8)
		return 3
	else
		return 0
	end
end

# ╔═╡ 27e554ff-9861-4a41-ad65-9d5ae7727e45
function T(s::State, a::Action)
	if R(s) != 0
		return Deterministic(params.null_state)
	end

	Nₐ = length(𝒜)
	next_states = Vector{State}(undef, Nₐ + 1)
	probabilities = zeros(Nₐ + 1)
	p_transition = params.p_transition

	for (i, a′) in enumerate(𝒜)
		prob = (a′ == a) ? p_transition : (1 - p_transition) / (Nₐ - 1)
		destination = s + MOVEMENTS[a′]
		next_states[i+1] = destination

		if inbounds(destination)
			probabilities[i+1] += prob
		end
	end
	
	# handle out-of-bounds transitions
	next_states[1] = s
	probabilities[1] = 1 - sum(probabilities)

	return SparseCat(next_states, probabilities)
end

# ╔═╡ e5286fa6-1a48-4020-ab03-c24a175c8c04
md"""
### Discount Factor
For an infinite horizon problem, we set the discount factor $\gamma \in [0,1]$ to a value where $\gamma < 1$ to discount future rewards. We bind $\gamma$ to a `PlutoUI` slider so we can adjust the value in real-time.
"""

# ╔═╡ 87a6c45e-6f3e-428e-8301-3b0c4166a84b
@bind γ Slider(0:0.05:1, default=0.95, show_value=true)

# ╔═╡ fd5b8960-933a-4ca0-9a7e-5003821ccfe3
md"""
### Termination
When the agent is in the `null_state` (which we've arbitrarily defined), then it is in a terminal state. Note for this problem, that termination cannot be _"when you're in a reward state"_, but has to be a separate `null_state` that we transition to _after_ we taken an action in a reward state (thus, _after_ we collect the reward).
"""

# ╔═╡ 6970821b-2b87-4e66-b737-512e83627998
termination(s::State) = s == params.null_state

# ╔═╡ 7c2c2733-eb28-4d85-9074-99e64074e414
md"""
### MDP Formulation (`QuickPOMDPs.jl`)
We can use the convinient package `QuickPOMDPs` to setup the MDP problem formulation for the Grid World problem. We set the _initial state distribution_ to be the entire state space $\mathcal{S}$, which will be sampled uniformly. A separate notebook is provided that creates the MDP in the "traditional" `POMDPs` manner, but we recommend `QuickPOMDPs` to get started.

We define the `GridWorld` abstract `MDP` type so we can reference it in other methods.
"""

# ╔═╡ 49b140ad-641f-436c-9492-cf3efbadd8d2
abstract type GridWorld <: MDP{State, Action} end

# ╔═╡ d9755f26-3f30-48ba-91d7-266c0204237d
# helper functions to get all the values from a policy
begin
	function one_based_policy!(policy)
		# change the default action in the policy (all zeros) to all ones (if needed)
		if all(iszero, policy.policy)
			policy.policy[:] = ones(eltype(policy.policy), length(policy.policy))
		end
	end

    function get_rewards(mdp::QuickMDP{GridWorld}, policy::Policy)
        null_state = params.null_state
        valid_states = setdiff(states(mdp), [null_state])
        U = map(s->reward(mdp, s), valid_states)
    end

    function values(mdp::QuickMDP{GridWorld}, policy::Policy)
        null_state = params.null_state
        valid_states = setdiff(states(mdp), [null_state])
        U = map(s->value(policy, s), valid_states)
    end

    function values(mdp::QuickMDP{GridWorld}, planner::MCTSPlanner)
        null_state = params.null_state
        valid_states = setdiff(states(mdp), [null_state])
        U = []
        for s in valid_states
            u = 0
            try
                u = value(planner, s)
            catch
                # state not in tree
            end
            push!(U, u)
        end
        return U
    end

    function values(mdp::QuickMDP{GridWorld}, policy::ValuePolicy)
        maxU = mapslices(maximum, policy.value_table, dims=2)
        return maxU[1:end-1] # remove null_state
    end

    struct NothingPolicy <: Policy end
    
    # Use this to get a stationary grid of rewards
    function values(mdp::QuickMDP{GridWorld}, policy::Union{NothingPolicy, FunctionPolicy})
        null_state = params.null_state
        valid_states = setdiff(states(mdp), [null_state])
        rewards = map(s->reward(mdp, s), valid_states)
    end
end

# ╔═╡ a3f76a77-bb6d-4a6b-8e5a-170dcc867c07
md"""
## Solutions (Offline)
Depending on the type of problem (discrete/continuous states or actions, model-free vs. model-based), we can pick an MDP solver that meets our needs. For a list of MDP (and POMDP) solvers based on online/offline solutions and continuous/discrete spaces, see the [README](https://github.com/JuliaPOMDP/POMDPs.jl#mdp-solvers) of POMDPs.jl.

Solution methods typically follow the defined `POMDPs.jl` interface syntax:
```julia
solver = FancyAlgorithmSolver() # inputs are the parameters of said algorithm
policy = solve(solver, mdp)     # solves the MDP and returns a policy
```
"""

# ╔═╡ 9f5dc78e-8183-4a03-9282-1aebf1af218c
md"""
### Value Iteration

*Value iteration* is an algorithm to solve discrete MDPs that uses dynamic programming and the Bellman equation to compute the optimal *value* (or *utility*) $U^*$ iteratively.$^2$

$$U^*(s) = \max_a\Biggl(\underbrace{R(s,a)}_{\substack{\text{immediate}\\\text{reward}}} + \overbrace{\gamma \sum_{s^\prime} T(s^\prime \mid s,a) U^*(s^\prime)}^{\text{discounted future reward}}\Biggr) \quad \text{for all states } s$$

The optimal policy $\pi^*$ can be extracted using the value function.

$$\pi^*(s) = \mathop{\rm arg\,max}_a\left(R(s,a) + \gamma \sum_{s^\prime} T(s^\prime \mid s, a) U^*(s^\prime)\right)$$

But, of course, we don't have to implement these algorithms ourselves, we can use `POMDPs.jl` 🙃!
"""

# ╔═╡ c67f7fc6-7af8-4e4f-a341-133c70f879bc
md"Value iteration is *model-based* because it relies on the transition model $T$ and reward model $R$ (also called _transition function_ and _reward function_)."

# ╔═╡ 48c966cb-6b79-42a6-8ff0-2fe3261f3981
solver = ValueIterationSolver(max_iterations=30);

# ╔═╡ 12501ad4-b42d-4fc4-b54b-30f4b929c0ab
md"""
#### Policy
We set the discount factor to the variable $\gamma$ which is bound to a slider, and solve the MDP to obtain a policy $\pi$ mapping states $s$ to actions $a$.

$$\pi(s) = a$$
"""

# ╔═╡ 90b507bd-8cab-4c30-816e-a4b264e903a6
md"### Example: _Transition Probability and State-Value_
What does the transition probability distribution look like from some current state $s_r$? Also, what is the _value_ of a particular state (shown in the color gradient)?"

# ╔═╡ 73182581-fdf4-4252-b64e-34f39e1f96da
function plot_transition_probability(distr)
	if distr isa Deterministic
		vals = [distr.val]
		probs = [1]
	else
		vals = distr.vals
		probs = distr.probs
	end
	xtick_states = [(v.x,v.y) for v in vals]
	@info xtick_states
	@info probs
	cmap_probs = ColorScheme([colorant"#B3BCDB", colorant"#4063D8"])
	bar(probs,
		xformatter = x->1 <= x <= length(xtick_states) ? xtick_states[Int(x)] : "",
		label=false,
		aspect_ratio=5,
		group=xtick_states,
		color=map(v->get(cmap_probs, v), probs))
	title!("Transition Probability\nDistribution")
	xlabel!("next state 𝑠′")
	ylabel!("probability")
	ylims!(0, 1.1)
	xlims!(0, 6)
	yticks!(0:0.1:1)
end

# ╔═╡ 786b27eb-129f-4538-beca-7e8b69fd40e4
md"""
 $(x,y)$ = $(@bind state_x NumberField(1:10, default=8)) | $(@bind state_y NumberField(1:10, default=3))
"""

# ╔═╡ 9cb6e19b-25f4-44b5-8155-d55ad3ba617c
sᵣ = State(state_x, state_y)

# ╔═╡ da9926ae-4e49-4ff3-abc2-d8249bddb0f2
md"And can look at the distribution over next states $s^\prime$ via $T(s^\prime \mid s, a)$."

# ╔═╡ b942fd56-13c3-4729-a701-63f103b13638
md"""
### Example: _Using the Policy_
Given some current state $s$, we can pull out an action $a$ using the policy $\pi$:

$$\pi(s) = a$$

"""

# ╔═╡ 4de6845e-a555-4147-86e8-d623e399c22a
md"We can query the policy $\pi$ for an action using the `action(π, s)` function."

# ╔═╡ 887f90ce-98eb-4262-894b-e14a0a53fa50
md"We can look at the value $U(s)$ at an individual state given the policy."

# ╔═╡ a4e4d65f-a734-404d-8478-029b0017651c
md"""
And also look at the state-action value, or $Q$-value, over all possible actions, where

$$Q(s,a) = \underbrace{R(s,a)}_{\substack{\text{immediate}\\\text{reward}}} + \overbrace{\gamma \sum_{s^\prime} T(s^\prime \mid s,a) U(s^\prime)}^{\text{discounted future reward}}.$$

We use the `value(π, s)` function for $U(s)$ and `value(π, s, a)` for $Q(s,a)$, where

$$U(s) = \max_a Q(s,a)$$
"""

# ╔═╡ 73cde70f-17f9-4ccd-ae4e-0cf050c2915e
Q(π, s, a) = value(π, s, a) # Q-value (i.e., state-action value)

# ╔═╡ cd4d33c8-bc24-4327-a65e-ed2d46af766b
md"## Visualization"

# ╔═╡ d3b0aeb2-11c9-4d4e-aa53-bb831ccd74a2
md"""
### Value Iteration Policy

The arrows in each cell (i.e. state) show the policy, and the color represents the discounted utility $U(s)$ (where green is positive utility and red is negative utility).
"""

# ╔═╡ a8817d6e-2302-4f39-8b93-66d550ca09ef
@bind vi_iterations Slider(0:30, default=0, show_value=true)

# ╔═╡ 5024e6c5-39e2-4bd6-acca-05267cf8639e
vi_solver = ValueIterationSolver(max_iterations=vi_iterations);

# ╔═╡ 3590b4b0-5a05-4052-aa58-f48d3912ce77
@bind γ_vi Slider(0:0.05:1, default=0.95, show_value=true)

# ╔═╡ 4d698cad-a570-4608-b6dd-20de5d7dbe33
vi_mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ_vi, # custom discount for visualization of Value Iteration policy
    initialstate = 𝒮,
    isterminal   = termination);

# ╔═╡ 1540a649-b238-498e-a8fb-5a29461194b5
vi_policy = solve(vi_solver, vi_mdp);

# ╔═╡ 6d024b5b-faa3-4075-babd-c6b260cef55e
one_based_policy!(vi_policy); # handles the case when iterations = 0

# ╔═╡ dd5e9a13-4297-4717-a150-e1908faea2ca
md"""
Here's an example where we can save the rendering as an SVG file using the `Plots.jl` package.
"""

# ╔═╡ b4dd0437-b945-4b4e-a504-1ac0fca54a75
md"""
### Animated GIF
"""

# ╔═╡ 38af3571-9b0a-4b19-b33a-573101b597a0
md"Create value iteration GIF? $(@bind create_gif CheckBox())"

# ╔═╡ a2b7e745-8b15-42c6-89ca-e97aef1c9a0f
md"""
## Reinforcement Learning Solvers
"""

# ╔═╡ a94529cf-1ab3-4b28-887a-04be9d103869
html"""
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="369.699pt" height="185.742pt" viewBox="0 0 369.699 185.742" version="1.1">
<defs>
<g>
<symbol overflow="visible" id="rl-glyph0-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-1">
<path style="stroke:none;" d="M 12.984375 -5.40625 L 12.375 -5.40625 C 11.859375 -2.1875 11.53125 -0.875 8.109375 -0.875 L 5.453125 -0.875 C 4.515625 -0.875 4.609375 -0.875 4.609375 -1.53125 L 4.609375 -6.734375 L 6.28125 -6.734375 C 8.203125 -6.734375 8.28125 -6.21875 8.28125 -4.40625 L 9.046875 -4.40625 L 9.046875 -9.9375 L 8.28125 -9.9375 C 8.28125 -8.09375 8.203125 -7.609375 6.28125 -7.609375 L 4.609375 -7.609375 L 4.609375 -12.25 C 4.609375 -12.90625 4.515625 -12.921875 5.453125 -12.921875 L 8.03125 -12.921875 C 11.078125 -12.921875 11.46875 -11.953125 11.796875 -9.0625 L 12.578125 -9.0625 L 12 -13.8125 L 0.515625 -13.8125 L 0.515625 -12.921875 L 1.140625 -12.921875 C 2.671875 -12.921875 2.5625 -12.828125 2.5625 -12.109375 L 2.5625 -1.671875 C 2.5625 -0.953125 2.671875 -0.875 1.140625 -0.875 L 0.515625 -0.875 L 0.515625 0 L 12.25 0 L 13.125 -5.40625 Z M 12.984375 -5.40625 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-2">
<path style="stroke:none;" d="M 10.78125 -0.125 L 10.78125 -0.875 C 9.625 -0.875 9.25 -0.734375 9.21875 -1.328125 L 9.21875 -5.140625 C 9.21875 -6.859375 9.1875 -7.5625 8.5625 -8.28125 C 8.28125 -8.625 7.546875 -9.0625 6.390625 -9.0625 C 4.9375 -9.0625 3.859375 -8.0625 3.3125 -6.828125 L 3.5625 -6.828125 L 3.5625 -9.078125 L 0.5 -8.84375 L 0.5 -7.96875 C 2.03125 -7.96875 2.046875 -7.953125 2.046875 -6.96875 L 2.046875 -1.640625 C 2.046875 -0.734375 1.96875 -0.875 0.5 -0.875 L 0.5 0.015625 L 2.890625 -0.0625 L 5.234375 0.015625 L 5.234375 -0.875 C 3.78125 -0.875 3.6875 -0.734375 3.6875 -1.640625 L 3.6875 -5.296875 C 3.6875 -7.375 4.984375 -8.359375 6.25 -8.359375 C 7.515625 -8.359375 7.59375 -7.40625 7.59375 -6.28125 L 7.59375 -1.640625 C 7.59375 -0.734375 7.515625 -0.875 6.03125 -0.875 L 6.03125 0.015625 L 8.421875 -0.0625 L 10.78125 0.015625 Z M 10.78125 -0.125 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-3">
<path style="stroke:none;" d="M 10.234375 -8.09375 L 10.234375 -8.859375 C 9.65625 -8.8125 9.078125 -8.78125 8.625 -8.78125 L 6.75 -8.84375 L 6.75 -7.96875 C 7.625 -7.953125 7.703125 -7.609375 7.703125 -7.234375 C 7.703125 -7.046875 7.671875 -6.96875 7.59375 -6.75 L 5.5625 -1.671875 L 5.8125 -1.671875 L 3.578125 -7.234375 C 3.46875 -7.484375 3.421875 -7.484375 3.421875 -7.484375 L 3.46875 -7.5625 C 3.46875 -8.09375 4.125 -7.96875 4.609375 -7.96875 L 4.609375 -8.84375 L 2.3125 -8.78125 C 1.765625 -8.78125 0.96875 -8.8125 0.234375 -8.859375 L 0.234375 -7.96875 C 1.640625 -7.96875 1.578125 -7.96875 1.828125 -7.34375 L 4.703125 -0.28125 C 4.828125 0 5 0.21875 5.265625 0.21875 C 5.515625 0.21875 5.71875 -0.078125 5.796875 -0.28125 L 8.40625 -6.75 C 8.59375 -7.21875 8.8125 -7.953125 10.234375 -7.96875 Z M 10.234375 -8.09375 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-4">
<path style="stroke:none;" d="M 5.046875 -0.125 L 5.046875 -0.875 C 3.609375 -0.875 3.640625 -0.84375 3.640625 -1.609375 L 3.640625 -9.078125 L 0.59375 -8.84375 L 0.59375 -7.96875 C 2.03125 -7.96875 2.078125 -7.96875 2.078125 -7 L 2.078125 -1.640625 C 2.078125 -0.734375 1.984375 -0.875 0.515625 -0.875 L 0.515625 0.015625 L 2.84375 -0.0625 C 3.546875 -0.0625 4.25 -0.015625 5.046875 0.015625 Z M 3.9375 -12.15625 C 3.9375 -12.6875 3.359375 -13.34375 2.765625 -13.34375 C 2.09375 -13.34375 1.546875 -12.65625 1.546875 -12.15625 C 1.546875 -11.609375 2.15625 -10.984375 2.75 -10.984375 C 3.421875 -10.984375 3.9375 -11.65625 3.9375 -12.15625 Z M 3.9375 -12.15625 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-5">
<path style="stroke:none;" d="M 7.375 -7.703125 C 7.375 -8.34375 6.640625 -9.0625 5.78125 -9.0625 C 4.328125 -9.0625 3.46875 -7.59375 3.1875 -6.734375 L 3.453125 -6.734375 L 3.453125 -9.078125 L 0.421875 -8.84375 L 0.421875 -7.96875 C 1.953125 -7.96875 1.96875 -7.953125 1.96875 -6.96875 L 1.96875 -1.640625 C 1.96875 -0.734375 1.890625 -0.875 0.421875 -0.875 L 0.421875 0.015625 L 2.828125 -0.0625 C 3.625 -0.0625 4.5625 -0.0625 5.484375 0.015625 L 5.484375 -0.875 L 4.9375 -0.875 C 3.46875 -0.875 3.546875 -0.953125 3.546875 -1.671875 L 3.546875 -4.734375 C 3.546875 -6.71875 4.265625 -8.359375 5.78125 -8.359375 C 5.921875 -8.359375 5.859375 -8.40625 5.890625 -8.390625 L 6 -8.609375 C 5.9375 -8.59375 5.40625 -8.203125 5.40625 -7.6875 C 5.40625 -7.125 5.953125 -6.71875 6.390625 -6.71875 C 6.75 -6.71875 7.375 -7.078125 7.375 -7.703125 Z M 7.375 -7.703125 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-6">
<path style="stroke:none;" d="M 9.5 -4.375 C 9.5 -6.9375 7.390625 -9.1875 4.984375 -9.1875 C 2.484375 -9.1875 0.421875 -6.875 0.421875 -4.375 C 0.421875 -1.8125 2.625 0.21875 4.953125 0.21875 C 7.375 0.21875 9.5 -1.859375 9.5 -4.375 Z M 7.59375 -4.546875 C 7.59375 -3.828125 7.625 -2.84375 7.1875 -1.96875 L 7.15625 -1.875 C 6.71875 -0.96875 5.96875 -0.53125 4.984375 -0.53125 C 4.125 -0.53125 3.328125 -0.921875 2.78125 -1.828125 C 2.296875 -2.703125 2.328125 -3.828125 2.328125 -4.546875 C 2.328125 -5.3125 2.296875 -6.3125 2.765625 -7.1875 C 3.3125 -8.109375 4.15625 -8.484375 4.953125 -8.484375 C 5.84375 -8.484375 6.59375 -8.09375 7.109375 -7.234375 C 7.625 -6.375 7.59375 -5.296875 7.59375 -4.546875 Z M 7.59375 -4.546875 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-7">
<path style="stroke:none;" d="M 16.3125 -0.125 L 16.3125 -0.875 C 15.15625 -0.875 14.78125 -0.734375 14.765625 -1.328125 L 14.765625 -5.140625 C 14.765625 -6.859375 14.71875 -7.5625 14.109375 -8.28125 C 13.828125 -8.625 13.09375 -9.0625 11.9375 -9.0625 C 10.265625 -9.0625 9.25 -7.734375 8.90625 -6.96875 L 9.15625 -6.96875 C 8.890625 -8.703125 7.296875 -9.0625 6.390625 -9.0625 C 4.9375 -9.0625 3.859375 -8.0625 3.3125 -6.828125 L 3.5625 -6.828125 L 3.5625 -9.078125 L 0.5 -8.84375 L 0.5 -7.96875 C 2.03125 -7.96875 2.046875 -7.953125 2.046875 -6.96875 L 2.046875 -1.640625 C 2.046875 -0.734375 1.96875 -0.875 0.5 -0.875 L 0.5 0.015625 L 2.890625 -0.0625 L 5.234375 0.015625 L 5.234375 -0.875 C 3.78125 -0.875 3.6875 -0.734375 3.6875 -1.640625 L 3.6875 -5.296875 C 3.6875 -7.375 4.984375 -8.359375 6.25 -8.359375 C 7.515625 -8.359375 7.59375 -7.40625 7.59375 -6.28125 L 7.59375 -1.640625 C 7.59375 -0.734375 7.515625 -0.875 6.03125 -0.875 L 6.03125 0.015625 L 8.421875 -0.0625 L 10.78125 0.015625 L 10.78125 -0.875 C 9.328125 -0.875 9.21875 -0.734375 9.21875 -1.640625 L 9.21875 -5.296875 C 9.21875 -7.375 10.515625 -8.359375 11.796875 -8.359375 C 13.046875 -8.359375 13.125 -7.40625 13.125 -6.28125 L 13.125 -1.640625 C 13.125 -0.734375 13.046875 -0.875 11.578125 -0.875 L 11.578125 0.015625 L 13.96875 -0.0625 L 16.3125 0.015625 Z M 16.3125 -0.125 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-8">
<path style="stroke:none;" d="M 8.390625 -2.484375 C 8.390625 -2.6875 8.109375 -2.875 8.015625 -2.875 C 7.828125 -2.875 7.65625 -2.609375 7.609375 -2.453125 C 6.90625 -0.390625 5.265625 -0.53125 5.0625 -0.53125 C 4.0625 -0.53125 3.34375 -1.09375 2.890625 -1.828125 C 2.296875 -2.78125 2.328125 -4 2.328125 -4.609375 L 7.765625 -4.609375 C 8.203125 -4.609375 8.390625 -4.71875 8.390625 -5.140625 C 8.390625 -7.109375 7.1875 -9.1875 4.703125 -9.1875 C 2.390625 -9.1875 0.421875 -7 0.421875 -4.5 C 0.421875 -1.828125 2.65625 0.21875 4.9375 0.21875 C 7.375 0.21875 8.390625 -2.109375 8.390625 -2.484375 Z M 6.953125 -5.28125 L 2.375 -5.28125 C 2.46875 -8.109375 4.03125 -8.484375 4.703125 -8.484375 C 6.75 -8.484375 6.8125 -5.921875 6.8125 -5.28125 Z M 6.953125 -5.28125 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-9">
<path style="stroke:none;" d="M 6.734375 -2.59375 L 6.734375 -3.859375 L 5.96875 -3.859375 L 5.96875 -2.625 C 5.96875 -1.15625 5.515625 -0.53125 4.78125 -0.53125 C 3.453125 -0.53125 3.5625 -2.21875 3.5625 -2.546875 L 3.5625 -7.96875 L 6.421875 -7.96875 L 6.421875 -8.84375 L 3.5625 -8.84375 L 3.5625 -12.515625 L 2.8125 -12.515625 C 2.78125 -10.734375 2.328125 -8.75 0.234375 -8.671875 L 0.234375 -7.96875 L 1.9375 -7.96875 L 1.9375 -2.59375 C 1.9375 -0.140625 3.921875 0.21875 4.640625 0.21875 C 6.0625 0.21875 6.734375 -1.3125 6.734375 -2.59375 Z M 6.734375 -2.59375 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-10">
<path style="stroke:none;" d="M 14.40625 -0.125 L 14.40625 -0.875 L 13.921875 -0.875 C 12.734375 -0.875 12.578125 -0.875 12.34375 -1.53125 L 8.046875 -13.984375 C 7.953125 -14.25 7.796875 -14.515625 7.46875 -14.515625 C 7.15625 -14.515625 6.953125 -14.265625 6.859375 -13.984375 L 2.734375 -2.078125 C 2.375 -1.0625 1.71875 -0.890625 0.5 -0.875 L 0.5 0.015625 L 2.671875 -0.0625 L 5.078125 0.015625 L 5.078125 -0.875 C 3.96875 -0.875 3.578125 -1.234375 3.578125 -1.75 C 3.578125 -1.8125 3.5625 -1.9375 3.625 -2.015625 L 4.5 -4.546875 L 9.25 -4.546875 L 10.265625 -1.609375 C 10.28125 -1.53125 10.3125 -1.421875 10.3125 -1.328125 C 10.3125 -0.734375 9.34375 -0.875 8.671875 -0.875 L 8.671875 0.015625 C 9.515625 -0.0625 10.921875 -0.0625 11.671875 -0.0625 L 14.40625 0.015625 Z M 9.125 -5.421875 L 4.828125 -5.421875 L 7 -11.75 L 6.734375 -11.75 L 8.921875 -5.421875 Z M 9.125 -5.421875 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-11">
<path style="stroke:none;" d="M 9.78125 -8.171875 C 9.78125 -8.5 9.421875 -9.28125 8.640625 -9.28125 C 8.25 -9.28125 7.265625 -9.125 6.515625 -8.390625 C 5.78125 -8.96875 4.859375 -9.0625 4.421875 -9.0625 C 2.5625 -9.0625 1.0625 -7.546875 1.0625 -6.015625 C 1.0625 -5.140625 1.53125 -4.296875 1.9375 -3.96875 C 1.765625 -3.765625 1.375 -3.015625 1.375 -2.3125 C 1.375 -1.6875 1.671875 -0.859375 2.078125 -0.59375 C 1.1875 -0.34375 0.421875 0.65625 0.421875 1.453125 C 0.421875 2.890625 2.53125 4.109375 4.953125 4.109375 C 7.3125 4.109375 9.5 2.96875 9.5 1.421875 C 9.5 0.71875 9.1875 -0.390625 8.171875 -0.953125 C 7.109375 -1.515625 5.875 -1.546875 4.65625 -1.546875 C 4.15625 -1.546875 3.3125 -1.546875 3.171875 -1.578125 C 2.53125 -1.65625 2.234375 -2.125 2.234375 -2.765625 C 2.234375 -2.84375 2.1875 -3.234375 2.46875 -3.53125 C 3.125 -3.0625 4.046875 -2.96875 4.421875 -2.96875 C 6.28125 -2.96875 7.765625 -4.46875 7.765625 -6 C 7.765625 -6.734375 7.40625 -7.5625 7.015625 -7.921875 C 7.625 -8.53125 8.265625 -8.59375 8.625 -8.59375 L 8.703125 -8.625 C 8.703125 -8.625 8.765625 -8.59375 8.828125 -8.5625 L 8.828125 -8.828125 C 8.609375 -8.75 8.359375 -8.390625 8.359375 -8.140625 C 8.359375 -7.8125 8.765625 -7.453125 9.078125 -7.453125 C 9.28125 -7.453125 9.78125 -7.703125 9.78125 -8.171875 Z M 6.015625 -6.015625 C 6.015625 -5.484375 6.03125 -4.9375 5.734375 -4.4375 C 5.578125 -4.203125 5.21875 -3.6875 4.421875 -3.6875 C 2.6875 -3.6875 2.8125 -5.53125 2.8125 -6 C 2.8125 -6.53125 2.78125 -7.09375 3.09375 -7.59375 C 3.25 -7.828125 3.625 -8.34375 4.421875 -8.34375 C 6.15625 -8.34375 6.015625 -6.46875 6.015625 -6.015625 Z M 8.203125 1.453125 C 8.203125 2.53125 6.9375 3.390625 4.984375 3.390625 C 2.96875 3.390625 1.71875 2.515625 1.71875 1.453125 C 1.71875 0.53125 2.34375 -0.078125 3.234375 -0.140625 L 4.40625 -0.140625 C 6.109375 -0.140625 8.203125 -0.265625 8.203125 1.453125 Z M 8.203125 1.453125 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-12">
<path style="stroke:none;" d="M 14.484375 -6.875 C 14.484375 -10.9375 11.359375 -14.296875 7.734375 -14.296875 C 4.15625 -14.296875 0.96875 -10.984375 0.96875 -6.875 C 0.96875 -2.78125 4.1875 0.4375 7.734375 0.4375 C 11.359375 0.4375 14.484375 -2.84375 14.484375 -6.875 Z M 12.171875 -7.15625 C 12.171875 -1.875 9.578125 -0.34375 7.75 -0.34375 C 5.84375 -0.34375 3.28125 -1.953125 3.28125 -7.15625 C 3.28125 -12.3125 6.078125 -13.546875 7.734375 -13.546875 C 9.46875 -13.546875 12.171875 -12.25 12.171875 -7.15625 Z M 12.171875 -7.15625 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-13">
<path style="stroke:none;" d="M 10.5 -4.421875 C 10.5 -6.953125 8.421875 -9.0625 6.15625 -9.0625 C 4.609375 -9.0625 3.640625 -8.09375 3.328125 -7.734375 L 3.546875 -7.625 L 3.546875 -14.109375 L 0.421875 -13.859375 L 0.421875 -12.984375 C 1.953125 -12.984375 1.96875 -12.96875 1.96875 -12 L 1.96875 0 L 2.671875 0 L 3.328125 -1.109375 C 3.53125 -0.8125 4.46875 0.21875 5.9375 0.21875 C 8.3125 0.21875 10.5 -1.859375 10.5 -4.421875 Z M 8.59375 -4.4375 C 8.59375 -3.703125 8.59375 -2.609375 8.015625 -1.71875 C 7.59375 -1.09375 6.9375 -0.484375 5.859375 -0.484375 C 4.953125 -0.484375 4.328125 -0.921875 3.84375 -1.65625 C 3.5625 -2.078125 3.609375 -2.03125 3.609375 -2.390625 L 3.609375 -6.5 C 3.609375 -6.875 3.5625 -6.8125 3.78125 -7.125 C 4.5625 -8.25 5.578125 -8.359375 6.0625 -8.359375 C 6.953125 -8.359375 7.5625 -7.890625 8.046875 -7.125 C 8.5625 -6.3125 8.59375 -5.265625 8.59375 -4.4375 Z M 8.59375 -4.4375 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-14">
<path style="stroke:none;" d="M 7.296875 -2.671875 C 7.296875 -3.71875 6.65625 -4.421875 6.421875 -4.65625 C 5.75 -5.296875 4.90625 -5.5 4.0625 -5.65625 C 2.953125 -5.875 1.734375 -6 1.734375 -7.15625 C 1.734375 -7.84375 2.125 -8.546875 3.84375 -8.546875 C 6.03125 -8.546875 6 -6.875 6.03125 -6.25 C 6.0625 -6.078125 6.5 -6 6.5 -6 L 6.421875 -5.953125 C 6.671875 -5.953125 6.796875 -6.171875 6.796875 -6.546875 L 6.796875 -8.5625 C 6.796875 -8.90625 6.671875 -9.1875 6.453125 -9.1875 C 6.359375 -9.1875 6.21875 -9.140625 5.953125 -8.90625 C 5.890625 -8.828125 5.703125 -8.640625 5.703125 -8.671875 C 5.046875 -9.140625 4.140625 -9.1875 3.84375 -9.1875 C 1.421875 -9.1875 0.515625 -7.703125 0.515625 -6.59375 C 0.515625 -5.890625 0.875 -5.265625 1.421875 -4.828125 C 2.046875 -4.296875 2.703125 -4.140625 4.140625 -3.859375 C 4.578125 -3.78125 6.078125 -3.578125 6.078125 -2.15625 C 6.078125 -1.140625 5.515625 -0.484375 3.96875 -0.484375 C 2.296875 -0.484375 1.6875 -1.46875 1.3125 -3.171875 C 1.25 -3.421875 1.109375 -3.640625 0.921875 -3.640625 C 0.65625 -3.640625 0.515625 -3.359375 0.515625 -3.015625 L 0.515625 -0.375 C 0.515625 -0.046875 0.65625 0.21875 0.875 0.21875 C 0.96875 0.21875 1.078125 0.15625 1.453125 -0.21875 C 1.5 -0.265625 1.5 -0.296875 1.765625 -0.578125 C 2.546875 0.15625 3.546875 0.21875 3.96875 0.21875 C 6.25 0.21875 7.296875 -1.234375 7.296875 -2.671875 Z M 7.296875 -2.671875 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-15">
<path style="stroke:none;" d="M 9.734375 -1.890625 L 9.734375 -3.140625 L 8.984375 -3.140625 L 8.984375 -1.890625 C 8.984375 -0.734375 8.625 -0.75 8.40625 -0.75 C 7.75 -0.75 7.796875 -1.515625 7.796875 -1.609375 L 7.796875 -5.59375 C 7.796875 -6.4375 7.75 -7.3125 7.03125 -8.046875 C 6.25 -8.828125 5.171875 -9.1875 4.21875 -9.1875 C 2.59375 -9.1875 1.078125 -8.109375 1.078125 -6.796875 C 1.078125 -6.203125 1.609375 -5.734375 2.125 -5.734375 C 2.6875 -5.734375 3.171875 -6.25 3.171875 -6.765625 C 3.171875 -7.015625 2.953125 -7.8125 2.296875 -7.828125 C 2.65625 -8.3125 3.546875 -8.484375 4.1875 -8.484375 C 5.15625 -8.484375 6.15625 -7.828125 6.15625 -6.0625 L 6.15625 -5.453125 C 5.28125 -5.40625 3.890625 -5.34375 2.625 -4.734375 C 1.140625 -4.0625 0.5 -2.890625 0.5 -2.015625 C 0.5 -0.390625 2.5625 0.21875 3.828125 0.21875 C 5.140625 0.21875 6.171875 -0.703125 6.546875 -1.640625 L 6.296875 -1.640625 C 6.375 -0.84375 7.046875 0.125 7.984375 0.125 C 8.40625 0.125 9.734375 -0.28125 9.734375 -1.890625 Z M 6.15625 -2.90625 C 6.15625 -1.015625 4.859375 -0.484375 3.96875 -0.484375 C 2.984375 -0.484375 2.296875 -1.03125 2.296875 -2.03125 C 2.296875 -3.125 3.015625 -4.65625 6.15625 -4.765625 Z M 6.15625 -2.90625 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-16">
<path style="stroke:none;" d="M 6.71875 4.65625 C 6.71875 4.609375 6.671875 4.46875 6.328125 4.125 C 3.84375 1.609375 3.25 -2.046875 3.25 -5.09375 C 3.25 -8.5625 3.96875 -11.953125 6.421875 -14.4375 C 6.671875 -14.6875 6.71875 -14.796875 6.71875 -14.859375 C 6.71875 -15 6.515625 -15.203125 6.390625 -15.203125 C 6.203125 -15.203125 4.265625 -13.703125 3.09375 -11.171875 C 2.078125 -8.984375 1.828125 -6.765625 1.828125 -5.09375 C 1.828125 -3.546875 2.046875 -1.140625 3.140625 1.109375 C 4.34375 3.5625 6.203125 4.984375 6.390625 4.984375 C 6.515625 4.984375 6.71875 4.796875 6.71875 4.65625 Z M 6.71875 4.65625 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-17">
<path style="stroke:none;" d="M 5.875 -5.09375 C 5.875 -6.65625 5.65625 -9.0625 4.5625 -11.3125 C 3.359375 -13.765625 1.53125 -15.203125 1.328125 -15.203125 C 1.21875 -15.203125 1 -14.984375 1 -14.859375 C 1 -14.796875 1.03125 -14.6875 1.421875 -14.328125 C 3.359375 -12.34375 4.46875 -9.265625 4.46875 -5.09375 C 4.46875 -1.6875 3.765625 1.71875 1.296875 4.21875 C 1.03125 4.46875 1 4.609375 1 4.65625 C 1 4.78125 1.21875 4.984375 1.328125 4.984375 C 1.53125 4.984375 3.453125 3.5 4.625 0.96875 C 5.640625 -1.21875 5.875 -3.421875 5.875 -5.09375 Z M 5.875 -5.09375 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph0-18">
<path style="stroke:none;" d="M 8.390625 -2.484375 C 8.390625 -2.6875 8.0625 -2.828125 8.015625 -2.828125 C 7.828125 -2.828125 7.65625 -2.609375 7.609375 -2.484375 C 7.03125 -0.640625 5.875 -0.53125 5.140625 -0.53125 C 4.078125 -0.53125 2.453125 -1.25 2.453125 -4.46875 C 2.453125 -7.703125 3.96875 -8.421875 5.015625 -8.421875 C 5.203125 -8.421875 6.359375 -8.453125 6.828125 -7.921875 C 6.328125 -7.890625 6.078125 -7.15625 6.078125 -6.890625 C 6.078125 -6.375 6.578125 -5.859375 7.125 -5.859375 C 7.65625 -5.859375 8.171875 -6.3125 8.171875 -6.90625 C 8.171875 -8.265625 6.53125 -9.1875 5 -9.1875 C 2.515625 -9.1875 0.53125 -6.890625 0.53125 -4.421875 C 0.53125 -1.875 2.65625 0.21875 4.953125 0.21875 C 7.625 0.21875 8.390625 -2.296875 8.390625 -2.484375 Z M 8.390625 -2.484375 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph1-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="rl-glyph1-1">
<path style="stroke:none;" d="M 9.34375 -5.4375 C 9.34375 -7.515625 7.953125 -8.8125 6.15625 -8.8125 C 3.484375 -8.8125 0.8125 -5.96875 0.8125 -3.140625 C 0.8125 -1.171875 2.15625 0.21875 4 0.21875 C 6.65625 0.21875 9.34375 -2.53125 9.34375 -5.4375 Z M 4.03125 -0.21875 C 3.171875 -0.21875 2.296875 -0.84375 2.296875 -2.390625 C 2.296875 -3.359375 2.8125 -5.515625 3.453125 -6.53125 C 4.4375 -8.0625 5.578125 -8.359375 6.140625 -8.359375 C 7.296875 -8.359375 7.890625 -7.40625 7.890625 -6.21875 C 7.890625 -5.4375 7.484375 -3.34375 6.734375 -2.046875 C 6.03125 -0.890625 4.9375 -0.21875 4.03125 -0.21875 Z M 4.03125 -0.21875 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph1-2">
<path style="stroke:none;" d="M 7.4375 -7.53125 C 7.078125 -8.265625 6.5 -8.8125 5.59375 -8.8125 C 3.265625 -8.8125 0.796875 -5.875 0.796875 -2.96875 C 0.796875 -1.09375 1.890625 0.21875 3.453125 0.21875 C 3.84375 0.21875 4.84375 0.140625 6.03125 -1.28125 C 6.203125 -0.4375 6.890625 0.21875 7.84375 0.21875 C 8.546875 0.21875 9 -0.234375 9.328125 -0.875 C 9.65625 -1.59375 9.921875 -2.8125 9.921875 -2.84375 C 9.921875 -3.046875 9.734375 -3.046875 9.6875 -3.046875 C 9.484375 -3.046875 9.46875 -2.96875 9.40625 -2.6875 C 9.0625 -1.390625 8.703125 -0.21875 7.890625 -0.21875 C 7.34375 -0.21875 7.296875 -0.734375 7.296875 -1.140625 C 7.296875 -1.578125 7.328125 -1.734375 7.546875 -2.609375 C 7.765625 -3.453125 7.8125 -3.640625 7.984375 -4.40625 L 8.703125 -7.1875 C 8.84375 -7.75 8.84375 -7.796875 8.84375 -7.875 C 8.84375 -8.203125 8.609375 -8.40625 8.265625 -8.40625 C 7.796875 -8.40625 7.484375 -7.96875 7.4375 -7.53125 Z M 6.140625 -2.375 C 6.03125 -2.015625 6.03125 -1.96875 5.734375 -1.640625 C 4.859375 -0.53125 4.046875 -0.21875 3.484375 -0.21875 C 2.484375 -0.21875 2.21875 -1.3125 2.21875 -2.09375 C 2.21875 -3.09375 2.84375 -5.53125 3.3125 -6.453125 C 3.921875 -7.625 4.828125 -8.359375 5.625 -8.359375 C 6.90625 -8.359375 7.1875 -6.734375 7.1875 -6.609375 C 7.1875 -6.5 7.15625 -6.375 7.125 -6.28125 Z M 6.140625 -2.375 "/>
</symbol>
<symbol overflow="visible" id="rl-glyph2-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="rl-glyph2-1">
<path style="stroke:none;" d="M 3.4375 -5.515625 L 4.859375 -5.515625 C 5.125 -5.515625 5.296875 -5.515625 5.296875 -5.8125 C 5.296875 -6.015625 5.125 -6.015625 4.890625 -6.015625 L 3.5625 -6.015625 L 4.078125 -8.078125 C 4.09375 -8.15625 4.109375 -8.21875 4.109375 -8.28125 C 4.109375 -8.53125 3.921875 -8.71875 3.640625 -8.71875 C 3.296875 -8.71875 3.078125 -8.484375 2.984375 -8.125 C 2.890625 -7.765625 3.0625 -8.4375 2.453125 -6.015625 L 1.03125 -6.015625 C 0.765625 -6.015625 0.59375 -6.015625 0.59375 -5.703125 C 0.59375 -5.515625 0.75 -5.515625 1 -5.515625 L 2.328125 -5.515625 L 1.5 -2.21875 C 1.421875 -1.875 1.296875 -1.375 1.296875 -1.1875 C 1.296875 -0.359375 2 0.140625 2.796875 0.140625 C 4.34375 0.140625 5.21875 -1.8125 5.21875 -2 C 5.21875 -2.171875 5.03125 -2.171875 5 -2.171875 C 4.828125 -2.171875 4.8125 -2.15625 4.703125 -1.90625 C 4.3125 -1.03125 3.59375 -0.25 2.828125 -0.25 C 2.546875 -0.25 2.34375 -0.4375 2.34375 -0.9375 C 2.34375 -1.078125 2.40625 -1.375 2.421875 -1.5 Z M 3.4375 -5.515625 "/>
</symbol>
</g>
<clipPath id="rl-clip1">
  <path d="M 193 58 L 369.699219 58 L 369.699219 127 L 193 127 Z M 193 58 "/>
</clipPath>
</defs>
<g id="rl-surface1">
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 81.053969 28.347562 L -81.055406 28.347562 C -83.254625 28.347562 -85.039781 26.562406 -85.039781 24.363187 L -85.039781 -24.363375 C -85.039781 -26.562594 -83.254625 -28.34775 -81.055406 -28.34775 L 81.053969 -28.34775 C 83.257094 -28.34775 85.04225 -26.562594 85.04225 -24.363375 L 85.04225 24.363187 C 85.04225 26.562406 83.257094 28.347562 81.053969 28.347562 Z M 81.053969 28.347562 " transform="matrix(1,0,0,-1,85.239,92.871)"/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-1" x="28.88" y="99.536"/>
  <use xlink:href="#rl-glyph0-2" x="42.449116" y="99.536"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-3" x="52.969664" y="99.536"/>
  <use xlink:href="#rl-glyph0-4" x="63.490211" y="99.536"/>
  <use xlink:href="#rl-glyph0-5" x="69.029439" y="99.536"/>
  <use xlink:href="#rl-glyph0-6" x="76.840149" y="99.536"/>
  <use xlink:href="#rl-glyph0-2" x="86.802789" y="99.536"/>
  <use xlink:href="#rl-glyph0-7" x="97.881245" y="99.536"/>
  <use xlink:href="#rl-glyph0-8" x="114.479003" y="99.536"/>
  <use xlink:href="#rl-glyph0-2" x="123.325827" y="99.536"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-9" x="133.846375" y="99.536"/>
</g>
<g clip-path="url(#rl-clip1)" clip-rule="nonzero">
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 280.280531 28.347562 L 118.171156 28.347562 C 115.968031 28.347562 114.182875 26.562406 114.182875 24.363187 L 114.182875 -24.363375 C 114.182875 -26.562594 115.968031 -28.34775 118.171156 -28.34775 L 280.280531 -28.34775 C 282.47975 -28.34775 284.264906 -26.562594 284.264906 -24.363375 L 284.264906 24.363187 C 284.264906 26.562406 282.47975 28.347562 280.280531 28.347562 Z M 280.280531 28.347562 " transform="matrix(1,0,0,-1,85.239,92.871)"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-10" x="258.448" y="97.952"/>
  <use xlink:href="#rl-glyph0-11" x="273.39196" y="97.952"/>
  <use xlink:href="#rl-glyph0-8" x="283.3546" y="97.952"/>
  <use xlink:href="#rl-glyph0-2" x="292.201424" y="97.952"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-9" x="302.721972" y="97.952"/>
</g>
<path style="fill:none;stroke-width:1.59404;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 28.714125 28.746 C 67.815687 67.843656 131.409438 67.843656 165.690688 33.562406 " transform="matrix(1,0,0,-1,85.239,92.871)"/>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 255.75 64.125 C 254.589844 62.429688 253.070313 59.308594 252.402344 56.765625 L 248.386719 60.78125 C 250.929688 61.449219 254.050781 62.964844 255.75 64.125 "/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-12" x="113.051" y="21.585"/>
  <use xlink:href="#rl-glyph0-13" x="128.552868" y="21.585"/>
  <use xlink:href="#rl-glyph0-14" x="139.631324" y="21.585"/>
  <use xlink:href="#rl-glyph0-8" x="147.481884" y="21.585"/>
  <use xlink:href="#rl-glyph0-5" x="156.328708" y="21.585"/>
  <use xlink:href="#rl-glyph0-3" x="164.139418" y="21.585"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-15" x="173.54415" y="21.585"/>
  <use xlink:href="#rl-glyph0-9" x="183.50679" y="21.585"/>
  <use xlink:href="#rl-glyph0-4" x="191.257724" y="21.585"/>
  <use xlink:href="#rl-glyph0-6" x="196.796952" y="21.585"/>
  <use xlink:href="#rl-glyph0-2" x="206.759592" y="21.585"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-16" x="224.473166" y="21.585"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph1-1" x="232.223" y="21.585"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph2-1" x="241.881" y="24.575"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-17" x="248.897" y="21.585"/>
</g>
<path style="fill:none;stroke-width:1.59404;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 170.511 -28.746188 C 131.409438 -67.843844 67.815687 -67.843844 33.534437 -33.562594 " transform="matrix(1,0,0,-1,85.239,92.871)"/>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 113.953125 121.617188 C 115.113281 123.3125 116.632813 126.433594 117.300781 128.976563 L 121.316406 124.960938 C 118.773438 124.292969 115.648438 122.777344 113.953125 121.617188 "/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-10" x="136.225" y="174.12"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-18" x="150.611052" y="174.12"/>
  <use xlink:href="#rl-glyph0-9" x="159.457876" y="174.12"/>
  <use xlink:href="#rl-glyph0-4" x="167.20881" y="174.12"/>
  <use xlink:href="#rl-glyph0-6" x="172.748038" y="174.12"/>
  <use xlink:href="#rl-glyph0-2" x="182.710678" y="174.12"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-16" x="200.424252" y="174.12"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph1-2" x="208.175" y="174.12"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph2-1" x="218.707" y="177.108"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#rl-glyph0-17" x="225.723" y="174.12"/>
</g>
</g>
</svg>
"""

# ╔═╡ 2d58ccfd-9b20-459a-a636-a47496df1b18
md"""
### Temporal Difference Learning
Temporal difference (TD) learning algorithms are model-free reinforcement learning algorithms that learn the value function by sampling from the environment.

The `TabularTDLearning.jl` package includes $Q$-learning, $\rm S{\small ARSA}$, and $\rm S{\small ARSA(\lambda)}$.
"""

# ╔═╡ f0725051-8770-47d5-b8a7-65b1ab0955dd
md"""
### Q-Learning
The $Q$-learning algorithm$^3$ is model-free meaning it does not rely on the transition model $T$ or the reward model $R$, but uses samples of reward $r$ and the next state $s^\prime$.

$$Q(s,a) \leftarrow Q(s,a) + \alpha\overbrace{\biggl(\underbrace{r + \gamma \max_{a^\prime} Q(s^\prime, a^\prime)}_{\rm TD\ target} - Q(s,a)\biggr)}^{\rm temporal\ difference}
$$

In more detail, the $Q(s,a)$ equation means:
$$
\begin{align}
\rm new\ value \leftarrow \rm old\ value &+\ \\ \rm learning\ rate \ \cdot &\biggl(\rm reward\ + discount \cdot \bigl(future\ value\ estimate\bigr) - \rm old\ value \biggr)
\end{align}$$
"""

# ╔═╡ 10994f20-831d-4c63-ba4d-fe7e43347d5d
@bind γ_q Slider(0:0.05:1, default=0.95, show_value=true)

# ╔═╡ 9d1994d2-7ea9-4828-98fc-27bf5d17bab9
q_mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ_q, # custom discount for visualization of Q-learning policy
    initialstate = 𝒮,
    isterminal   = termination);

# ╔═╡ acbaca80-cafb-49c3-adea-4fd507c2c142
md"""
#### Q-learning Solver
"""

# ╔═╡ 2db772d9-16a1-4bdb-9205-611a1921831f
md"""
#### Q-learning Policy
"""

# ╔═╡ 9cf0f694-d4a3-48fd-9cb0-1c8ac0361d5c
@bind n_episodes_q Slider(1:500, default=500)

# ╔═╡ 61e6ad96-ff2a-4dd9-9698-48c33bd43f26
q_learning_solver = QLearningSolver(n_episodes=n_episodes_q,
                                    learning_rate=0.8,
                                    exploration_policy=EpsGreedyPolicy(q_mdp, 0.5),
	                                verbose=false);

# ╔═╡ ee7c1fe5-2991-4b2a-981b-1c72106d5855
q_learning_policy = solve(q_learning_solver, q_mdp);

# ╔═╡ bf23cf92-103f-4181-b9e6-97efe0249d0a
md"""
### SARSA
The $\rm S{\small ARSA}$ algorithm$^4$ is a modification of $Q$-learning that uses $(s, a, r, s^\prime, a^\prime)$. It uses the actual next action $a^\prime$ to update the $Q$-values **instead** of maximizing over all actions.

$$Q(s,a) \leftarrow Q(s,a) + \alpha\overbrace{\biggl(\underbrace{r + \gamma Q(s^\prime, a^\prime)}_{\rm TD\ target} - Q(s,a)\biggr)}^{\rm temporal\ difference}
$$

In more detail, the $Q(s,a)$ equation means:
$$
\begin{align}
\rm new\ value \leftarrow \rm old\ value &+\ \\ \rm learning\ rate \ \cdot &\biggl(\rm reward\ + discount \cdot \bigl(next\ value\bigr) - \rm old\ value \biggr)
\end{align}$$
"""

# ╔═╡ 1c1c765e-3a36-42f2-b1cb-8683b265ecad
@bind γ_sarsa Slider(0:0.05:1, default=0.95, show_value=true)

# ╔═╡ 3bcf0923-8ac2-4f82-97ca-d0996658a046
sarsa_mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ_sarsa, # custom discount for visualization of SARSA policy
    initialstate = 𝒮,
    isterminal   = termination);

# ╔═╡ e8620cb9-21de-4e5d-805a-0571eeceef7d
md"""
#### SARSA Solver
"""

# ╔═╡ c0929ea6-4b20-4e34-bec3-f0fc5935e406
md"""
#### SARSA Policy
"""

# ╔═╡ 5bfbceb4-7006-47c0-a965-13500caef00d
@bind n_episodes_sarsa Slider(0:10:1000, default=500)

# ╔═╡ f73f735c-6e8a-4ad4-b404-9772ce557eb1
sarsa_solver = SARSASolver(n_episodes=n_episodes_sarsa,
                           learning_rate=0.8,
                           exploration_policy=EpsGreedyPolicy(sarsa_mdp, 0.5),
	                       verbose=false);

# ╔═╡ 133eaaec-7113-4ec6-bac0-555f6efa1cb3
sarsa_policy = solve(sarsa_solver, sarsa_mdp);

# ╔═╡ 799026ed-92f0-439a-a7e3-bd362eb18b99
md"""
## Online Solvers
Online methods create a *planner* instead of a *policy* because we do not do any actual work when calling `solve` but instead plan out the solution.
Online methods solve for the best action given the current state instead of solving for all states.
"""

# ╔═╡ 3484668f-9cdb-4ac9-b683-8054f0ea9d7e
md"""
### Monte Carlo Tree Search
*Monte Carlo tree search* (MCTS)$^5$ is an anytime online algorithm that uses simulated rollouts of a random policy to estimate the value of each state-action node in a tree.

The four main stages of the algorithm are shown in the diagram below.
"""

# ╔═╡ 105a8fb9-008c-4ae1-83e8-8894209ada0e
html"""
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="512.428pt" height="184.382pt" viewBox="0 0 512.428 184.382" version="1.1">
<defs>
<g>
<symbol overflow="visible" id="glyph0-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="glyph0-1">
<path style="stroke:none;" d="M 5.515625 -2 C 5.515625 -2.796875 4.890625 -3.921875 3.78125 -4.1875 L 2.4375 -4.515625 C 1.5625 -4.71875 1.5 -5.25 1.5 -5.609375 C 1.5 -6.28125 1.921875 -6.75 2.765625 -6.75 C 4.140625 -6.75 4.515625 -5.90625 4.65625 -4.890625 C 4.671875 -4.75 4.828125 -4.546875 4.9375 -4.546875 C 5.078125 -4.546875 5.21875 -4.75 5.21875 -4.9375 L 5.21875 -6.953125 C 5.21875 -7.125 5.078125 -7.359375 4.96875 -7.359375 C 4.890625 -7.359375 4.78125 -7.296875 4.703125 -7.171875 L 4.421875 -6.71875 C 4.28125 -6.859375 3.71875 -7.359375 2.765625 -7.359375 C 1.609375 -7.359375 0.5625 -6.3125 0.5625 -5.25 C 0.5625 -4.609375 0.953125 -4.0625 1.078125 -3.90625 C 1.546875 -3.421875 1.953125 -3.296875 2.78125 -3.09375 C 2.9375 -3.0625 3.109375 -3.03125 3.265625 -2.984375 C 3.859375 -2.84375 3.953125 -2.84375 4.25 -2.515625 C 4.3125 -2.4375 4.5625 -2.21875 4.5625 -1.671875 C 4.5625 -0.96875 4.203125 -0.421875 3.296875 -0.421875 C 2.875 -0.421875 2.296875 -0.453125 1.75 -0.875 C 1.109375 -1.390625 1.125 -2 1.109375 -2.328125 C 1.109375 -2.40625 0.890625 -2.59375 0.84375 -2.59375 C 0.71875 -2.59375 0.5625 -2.359375 0.5625 -2.171875 L 0.5625 -0.171875 C 0.5625 -0.015625 0.71875 0.21875 0.828125 0.21875 C 0.890625 0.21875 1.015625 0.140625 1.078125 0.046875 L 1.359375 -0.40625 C 1.515625 -0.265625 2.265625 0.21875 3.296875 0.21875 C 4.546875 0.21875 5.515625 -0.953125 5.515625 -2 Z M 5.515625 -2 "/>
</symbol>
<symbol overflow="visible" id="glyph0-2">
<path style="stroke:none;" d="M 5.28125 -2.265625 L 4.890625 -2.265625 C 4.6875 -0.890625 4.609375 -0.578125 3.34375 -0.578125 L 2.265625 -0.578125 C 1.90625 -0.578125 2.015625 -0.4375 2.015625 -0.84375 L 2.015625 -2.515625 L 2.5625 -2.515625 C 3.296875 -2.515625 3.265625 -2.484375 3.265625 -1.65625 L 3.8125 -1.65625 L 3.8125 -3.96875 L 3.265625 -3.96875 C 3.265625 -3.125 3.296875 -3.09375 2.5625 -3.09375 L 2.015625 -3.09375 L 2.015625 -4.578125 C 2.015625 -4.96875 1.90625 -4.84375 2.265625 -4.84375 L 3.28125 -4.84375 C 4.4375 -4.84375 4.53125 -4.65625 4.671875 -3.40625 L 5.21875 -3.40625 L 5 -5.421875 L 0.265625 -5.421875 L 0.265625 -4.84375 C 1.09375 -4.84375 0.984375 -4.9375 0.984375 -4.53125 L 0.984375 -0.875 C 0.984375 -0.46875 1.09375 -0.578125 0.265625 -0.578125 L 0.265625 0 L 5.109375 0 L 5.4375 -2.265625 Z M 5.28125 -2.265625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-3">
<path style="stroke:none;" d="M 4.71875 -2.265625 L 4.328125 -2.265625 C 4.234375 -1.296875 4.28125 -0.578125 2.890625 -0.578125 L 2.265625 -0.578125 C 1.90625 -0.578125 2.015625 -0.4375 2.015625 -0.84375 L 2.015625 -4.53125 C 2.015625 -5.03125 2.015625 -4.875 2.90625 -4.875 L 2.90625 -5.46875 L 1.53125 -5.421875 L 0.265625 -5.46875 L 0.265625 -4.875 C 1.09375 -4.875 0.984375 -4.96875 0.984375 -4.5625 L 0.984375 -0.875 C 0.984375 -0.46875 1.09375 -0.578125 0.265625 -0.578125 L 0.265625 0 L 4.65625 0 L 4.875 -2.265625 Z M 4.71875 -2.265625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-4">
<path style="stroke:none;" d="M 5.53125 -1.921875 C 5.53125 -2.015625 5.390625 -2.25 5.25 -2.25 C 5.140625 -2.25 4.984375 -2.03125 4.96875 -1.921875 C 4.90625 -0.875 4.203125 -0.4375 3.375 -0.4375 C 2.875 -0.4375 1.515625 -0.53125 1.515625 -2.71875 C 1.515625 -4.9375 2.90625 -5.015625 3.359375 -5.015625 C 4.09375 -5.015625 4.78125 -4.65625 4.96875 -3.390625 C 4.984375 -3.3125 5.15625 -3.09375 5.25 -3.09375 C 5.390625 -3.09375 5.53125 -3.3125 5.53125 -3.5 L 5.53125 -5.1875 C 5.53125 -5.34375 5.390625 -5.59375 5.28125 -5.59375 C 5.21875 -5.59375 5.109375 -5.515625 5.03125 -5.421875 L 4.71875 -5.046875 C 4.640625 -5.140625 4.046875 -5.59375 3.296875 -5.59375 C 1.796875 -5.59375 0.34375 -4.234375 0.34375 -2.71875 C 0.34375 -1.203125 1.796875 0.140625 3.296875 0.140625 C 4.59375 0.140625 5.53125 -1.0625 5.53125 -1.921875 Z M 5.53125 -1.921875 "/>
</symbol>
<symbol overflow="visible" id="glyph0-5">
<path style="stroke:none;" d="M 5.6875 -3.53125 L 5.515625 -5.390625 L 0.359375 -5.390625 L 0.1875 -3.375 L 0.734375 -3.375 C 0.84375 -4.796875 0.890625 -4.8125 2 -4.8125 C 2.5625 -4.8125 2.4375 -4.96875 2.4375 -4.546875 L 2.4375 -0.921875 C 2.4375 -0.515625 2.546875 -0.578125 1.78125 -0.578125 L 1.421875 -0.578125 L 1.421875 0.015625 C 1.96875 -0.03125 2.546875 -0.03125 2.953125 -0.03125 C 3.34375 -0.03125 3.9375 -0.03125 4.46875 0.015625 L 4.46875 -0.578125 L 4.109375 -0.578125 C 3.34375 -0.578125 3.453125 -0.515625 3.453125 -0.921875 L 3.453125 -4.546875 C 3.453125 -4.96875 3.34375 -4.8125 3.890625 -4.8125 C 4.96875 -4.8125 5.03125 -4.8125 5.140625 -3.375 L 5.6875 -3.375 Z M 5.6875 -3.53125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-6">
<path style="stroke:none;" d="M 2.765625 -0.15625 L 2.765625 -0.578125 C 1.90625 -0.578125 2.015625 -0.453125 2.015625 -0.890625 L 2.015625 -4.5625 C 2.015625 -4.984375 1.90625 -4.875 2.765625 -4.875 L 2.765625 -5.46875 L 1.5 -5.421875 L 0.234375 -5.46875 L 0.234375 -4.875 C 1.09375 -4.875 0.984375 -4.984375 0.984375 -4.5625 L 0.984375 -0.890625 C 0.984375 -0.453125 1.09375 -0.578125 0.234375 -0.578125 L 0.234375 0.015625 L 1.5 -0.03125 L 2.765625 0.015625 Z M 2.765625 -0.15625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-7">
<path style="stroke:none;" d="M 5.984375 -2.703125 C 5.984375 -4.1875 4.640625 -5.59375 3.171875 -5.59375 C 1.6875 -5.59375 0.34375 -4.1875 0.34375 -2.703125 C 0.34375 -1.203125 1.71875 0.140625 3.171875 0.140625 C 4.625 0.140625 5.984375 -1.203125 5.984375 -2.703125 Z M 4.8125 -2.8125 C 4.8125 -0.84375 3.875 -0.453125 3.171875 -0.453125 C 2.46875 -0.453125 1.515625 -0.84375 1.515625 -2.8125 C 1.515625 -4.6875 2.5 -5.015625 3.171875 -5.015625 C 3.84375 -5.015625 4.8125 -4.671875 4.8125 -2.8125 Z M 4.8125 -2.8125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-8">
<path style="stroke:none;" d="M 5.828125 -5.03125 L 5.828125 -5.46875 L 4.828125 -5.421875 C 4.671875 -5.421875 4.21875 -5.4375 3.8125 -5.46875 L 3.8125 -4.875 C 4.6875 -4.875 4.53125 -4.5 4.53125 -4.28125 L 4.53125 -1.3125 L 4.78125 -1.4375 L 2.03125 -5.28125 C 1.9375 -5.40625 1.8125 -5.453125 1.625 -5.453125 L 0.265625 -5.453125 L 0.265625 -4.875 C 0.671875 -4.875 0.953125 -4.875 0.984375 -4.859375 L 0.984375 -1.171875 C 0.984375 -0.9375 1.140625 -0.578125 0.265625 -0.578125 L 0.265625 0.015625 L 1.28125 -0.03125 C 1.4375 -0.03125 1.890625 -0.015625 2.296875 0.015625 L 2.296875 -0.578125 C 1.421875 -0.578125 1.578125 -0.9375 1.578125 -1.171875 L 1.578125 -4.734375 L 1.3125 -4.625 L 1.421875 -4.5 L 4.546875 -0.15625 C 4.625 -0.046875 4.765625 0 4.828125 0 C 4.96875 0 5.109375 -0.234375 5.109375 -0.421875 L 5.109375 -4.28125 C 5.109375 -4.5 4.96875 -4.875 5.828125 -4.875 Z M 5.828125 -5.03125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-9">
<path style="stroke:none;" d="M 6.953125 -2.90625 L 6.578125 -2.90625 C 6.296875 -1.203125 6.21875 -0.640625 4.453125 -0.640625 L 2.953125 -0.640625 C 2.484375 -0.640625 2.609375 -0.53125 2.609375 -0.875 L 2.609375 -3.375 L 3.4375 -3.375 C 4.421875 -3.375 4.40625 -3.234375 4.40625 -2.203125 L 4.96875 -2.203125 L 4.96875 -5.171875 L 4.40625 -5.171875 C 4.40625 -4.140625 4.421875 -4 3.4375 -4 L 2.609375 -4 L 2.609375 -6.234375 C 2.609375 -6.5625 2.484375 -6.46875 2.953125 -6.46875 L 4.375 -6.46875 C 5.9375 -6.46875 6.09375 -6.125 6.28125 -4.53125 L 6.84375 -4.53125 L 6.515625 -7.109375 L 0.390625 -7.109375 L 0.390625 -6.46875 L 0.78125 -6.46875 C 1.5625 -6.46875 1.421875 -6.515625 1.421875 -6.15625 L 1.421875 -0.9375 C 1.421875 -0.578125 1.5625 -0.640625 0.78125 -0.640625 L 0.390625 -0.640625 L 0.390625 0 L 6.65625 0 L 7.125 -2.90625 Z M 6.953125 -2.90625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-10">
<path style="stroke:none;" d="M 5.96875 -0.15625 L 5.96875 -0.578125 C 5.53125 -0.578125 5.40625 -0.53125 5.21875 -0.6875 C 5.078125 -0.8125 3.5 -3.015625 3.484375 -3.046875 L 4.40625 -4.3125 C 4.75 -4.75 5.046875 -4.859375 5.640625 -4.875 L 5.640625 -5.46875 C 5.25 -5.4375 4.9375 -5.421875 4.6875 -5.421875 L 3.609375 -5.46875 L 3.609375 -4.890625 C 3.921875 -4.859375 3.90625 -4.9375 3.90625 -4.6875 C 3.90625 -4.515625 3.875 -4.515625 3.8125 -4.4375 L 3.125 -3.515625 L 2.28125 -4.671875 C 2.1875 -4.78125 2.25 -4.71875 2.25 -4.75 C 2.25 -4.921875 2.234375 -4.859375 2.578125 -4.875 L 2.578125 -5.46875 L 1.359375 -5.421875 C 1.09375 -5.421875 0.703125 -5.4375 0.234375 -5.46875 L 0.234375 -4.875 C 0.984375 -4.875 0.953125 -4.859375 1.21875 -4.46875 L 2.515625 -2.703125 L 1.453125 -1.265625 C 1.171875 -0.890625 0.953125 -0.59375 0.125 -0.578125 L 0.125 0.015625 C 0.53125 -0.015625 0.84375 -0.03125 1.09375 -0.03125 C 1.421875 -0.03125 1.875 -0.015625 2.15625 0.015625 L 2.15625 -0.5625 C 1.8125 -0.59375 1.859375 -0.5625 1.859375 -0.75 C 1.859375 -0.921875 1.890625 -0.921875 2.046875 -1.140625 C 2.359375 -1.5625 2.65625 -1.96875 2.859375 -2.234375 L 3.953125 -0.75 C 3.984375 -0.703125 3.96875 -0.75 3.96875 -0.671875 C 3.96875 -0.609375 4.078125 -0.59375 3.640625 -0.5625 L 3.640625 0.015625 L 4.875 -0.03125 C 5.109375 -0.03125 5.515625 -0.015625 5.96875 0.015625 Z M 5.96875 -0.15625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-11">
<path style="stroke:none;" d="M 5.203125 -3.875 C 5.203125 -4.59375 4.296875 -5.453125 3.15625 -5.453125 L 0.296875 -5.453125 L 0.296875 -4.875 C 1.109375 -4.875 1 -4.96875 1 -4.5625 L 1 -0.875 C 1 -0.46875 1.109375 -0.578125 0.296875 -0.578125 L 0.296875 0.015625 L 1.53125 -0.03125 L 2.75 0.015625 L 2.75 -0.578125 C 1.9375 -0.578125 2.03125 -0.46875 2.03125 -0.875 L 2.03125 -2.328125 L 3.21875 -2.328125 C 4.21875 -2.328125 5.203125 -3.125 5.203125 -3.875 Z M 4.03125 -3.875 C 4.03125 -3.484375 4.203125 -2.890625 2.9375 -2.890625 L 2 -2.890625 L 2 -4.609375 C 2 -5 1.90625 -4.875 2.25 -4.875 L 2.9375 -4.875 C 4.203125 -4.875 4.03125 -4.28125 4.03125 -3.875 Z M 4.03125 -3.875 "/>
</symbol>
<symbol overflow="visible" id="glyph0-12">
<path style="stroke:none;" d="M 5.921875 -0.15625 L 5.921875 -0.578125 C 5.21875 -0.578125 5.28125 -0.453125 5.140625 -0.8125 L 3.421875 -5.234375 C 3.375 -5.390625 3.171875 -5.625 3.0625 -5.625 C 2.921875 -5.625 2.734375 -5.390625 2.6875 -5.265625 L 1.09375 -1.140625 C 0.984375 -0.875 0.96875 -0.59375 0.171875 -0.578125 L 0.171875 0.015625 C 0.5625 -0.015625 0.84375 -0.03125 1.0625 -0.03125 L 2.046875 0.015625 L 2.046875 -0.578125 C 1.46875 -0.59375 1.59375 -0.765625 1.59375 -0.859375 C 1.59375 -0.9375 1.59375 -0.953125 1.859375 -1.6875 L 3.765625 -1.6875 L 3.90625 -1.28125 C 3.984375 -1.09375 4.109375 -0.75 4.109375 -0.6875 C 4.109375 -0.40625 3.921875 -0.578125 3.59375 -0.578125 L 3.59375 0.015625 L 4.8125 -0.03125 C 5.140625 -0.03125 5.53125 -0.015625 5.921875 0.015625 Z M 3.75 -2.265625 L 2.09375 -2.265625 L 2.96875 -4.5 L 2.65625 -4.5 L 3.53125 -2.265625 Z M 3.75 -2.265625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-13">
<path style="stroke:none;" d="M 4.203125 -1.578125 C 4.203125 -2.203125 3.765625 -2.5 3.765625 -2.5 L 3.765625 -2.734375 C 3.40625 -3.109375 3.140625 -3.1875 2.296875 -3.390625 C 2.078125 -3.4375 1.734375 -3.515625 1.671875 -3.53125 C 1.171875 -3.734375 1.171875 -3.953125 1.171875 -4.25 C 1.171875 -4.71875 1.421875 -5.046875 2.078125 -5.046875 C 3.34375 -5.046875 3.375 -4.03125 3.40625 -3.71875 C 3.421875 -3.65625 3.59375 -3.421875 3.703125 -3.421875 C 3.828125 -3.421875 3.984375 -3.640625 3.984375 -3.828125 L 3.984375 -5.1875 C 3.984375 -5.34375 3.828125 -5.59375 3.734375 -5.59375 C 3.671875 -5.59375 3.546875 -5.53125 3.40625 -5.3125 C 3.34375 -5.265625 3.25 -5.109375 3.296875 -5.1875 C 3.046875 -5.421875 2.484375 -5.59375 2.078125 -5.59375 C 1.171875 -5.59375 0.34375 -4.75 0.34375 -3.96875 C 0.34375 -3.40625 0.71875 -2.96875 0.84375 -2.84375 C 1.140625 -2.546875 1.453125 -2.46875 2.359375 -2.25 C 2.90625 -2.125 2.890625 -2.15625 3.125 -1.9375 C 3.171875 -1.90625 3.390625 -1.75 3.390625 -1.3125 C 3.390625 -0.8125 3.15625 -0.4375 2.46875 -0.4375 C 2.21875 -0.4375 1.78125 -0.421875 1.34375 -0.78125 C 0.890625 -1.125 0.921875 -1.5 0.90625 -1.765625 C 0.890625 -1.84375 0.671875 -2.03125 0.640625 -2.03125 C 0.515625 -2.03125 0.34375 -1.78125 0.34375 -1.609375 L 0.34375 -0.265625 C 0.34375 -0.09375 0.515625 0.140625 0.609375 0.140625 C 0.671875 0.140625 0.78125 0.078125 0.921875 -0.140625 C 0.984375 -0.1875 1.078125 -0.34375 1.03125 -0.28125 C 1.203125 -0.140625 1.828125 0.140625 2.46875 0.140625 C 3.4375 0.140625 4.203125 -0.78125 4.203125 -1.578125 Z M 4.203125 -1.578125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-14">
<path style="stroke:none;" d="M 7.125 -0.15625 L 7.125 -0.578125 C 6.3125 -0.578125 6.40625 -0.46875 6.40625 -0.875 L 6.40625 -4.5625 C 6.40625 -4.96875 6.3125 -4.875 7.125 -4.875 L 7.125 -5.453125 L 5.765625 -5.453125 C 5.546875 -5.453125 5.359375 -5.28125 5.28125 -5.09375 L 3.5625 -0.90625 L 3.875 -0.90625 L 2.15625 -5.109375 C 2.09375 -5.28125 1.890625 -5.453125 1.671875 -5.453125 L 0.3125 -5.453125 L 0.3125 -4.875 C 1.140625 -4.875 1.03125 -4.96875 1.03125 -4.5625 L 1.03125 -1.171875 C 1.03125 -0.9375 1.1875 -0.578125 0.3125 -0.578125 L 0.3125 0.015625 L 1.328125 -0.03125 C 1.46875 -0.03125 1.9375 -0.015625 2.328125 0.015625 L 2.328125 -0.578125 C 1.46875 -0.578125 1.609375 -0.9375 1.609375 -1.171875 L 1.609375 -4.84375 L 1.421875 -4.671875 L 3.1875 -0.34375 C 3.21875 -0.25 3.421875 0 3.53125 0 C 3.625 0 3.828125 -0.25 3.875 -0.34375 L 5.65625 -4.734375 L 5.46875 -4.921875 L 5.46875 -0.875 C 5.46875 -0.46875 5.578125 -0.578125 4.75 -0.578125 L 4.75 0.015625 L 5.9375 -0.03125 C 6.296875 -0.03125 6.65625 -0.015625 7.125 0.015625 Z M 7.125 -0.15625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-15">
<path style="stroke:none;" d="M 5.828125 -5.03125 L 5.828125 -5.46875 C 5.4375 -5.4375 4.90625 -5.421875 4.84375 -5.421875 C 4.5625 -5.421875 4.28125 -5.4375 3.84375 -5.46875 L 3.84375 -4.875 C 4.71875 -4.875 4.5625 -4.5 4.5625 -4.28125 L 4.5625 -1.9375 C 4.5625 -0.78125 3.859375 -0.4375 3.1875 -0.4375 C 2.875 -0.4375 2.015625 -0.390625 2.015625 -1.890625 L 2.015625 -4.5625 C 2.015625 -4.96875 1.90625 -4.875 2.734375 -4.875 L 2.734375 -5.46875 L 1.5 -5.421875 L 0.265625 -5.46875 L 0.265625 -4.875 C 1.109375 -4.875 0.984375 -4.96875 0.984375 -4.546875 L 0.984375 -2.375 C 0.984375 -2.140625 0.984375 -1.765625 1 -1.640625 C 1.09375 -0.828125 2.046875 0.140625 3.15625 0.140625 C 4.3125 0.140625 5.109375 -1.03125 5.109375 -1.765625 L 5.109375 -4.421875 C 5.125 -4.8125 5.234375 -4.875 5.828125 -4.875 Z M 5.828125 -5.03125 "/>
</symbol>
<symbol overflow="visible" id="glyph0-16">
<path style="stroke:none;" d="M 4.9375 -7.4375 C 4.9375 -7.546875 4.6875 -7.8125 4.578125 -7.8125 C 4.453125 -7.8125 4.25 -7.546875 4.21875 -7.4375 L 0.609375 1.953125 C 0.5625 2.078125 0.8125 2.25 0.8125 2.25 L 0.5625 2.140625 C 0.5625 2.25 0.8125 2.5 0.921875 2.5 C 1.046875 2.5 1.234375 2.25 1.28125 2.140625 L 4.890625 -7.25 C 4.9375 -7.375 4.890625 -7.328125 4.890625 -7.328125 Z M 4.9375 -7.4375 "/>
</symbol>
<symbol overflow="visible" id="glyph0-17">
<path style="stroke:none;" d="M 7.9375 -1.03125 C 7.9375 -1.09375 7.796875 -1.375 7.671875 -1.375 C 7.5625 -1.375 7.390625 -1.109375 7.390625 -1.046875 C 7.3125 -0.265625 7.0625 -0.328125 6.84375 -0.328125 C 6.3125 -0.328125 6.40625 -0.609375 6.21875 -1.765625 C 6.109375 -2.46875 6.015625 -2.828125 5.703125 -3.15625 C 5.5 -3.375 5.078125 -3.609375 4.6875 -3.71875 L 4.6875 -3.390625 C 5.875 -3.671875 6.671875 -4.5 6.671875 -5.171875 C 6.671875 -6.140625 5.296875 -7.140625 3.796875 -7.140625 L 0.40625 -7.140625 L 0.40625 -6.5 L 0.8125 -6.5 C 1.578125 -6.5 1.4375 -6.546875 1.4375 -6.1875 L 1.4375 -0.9375 C 1.4375 -0.578125 1.578125 -0.640625 0.8125 -0.640625 L 0.40625 -0.640625 L 0.40625 0.015625 C 0.921875 -0.03125 1.640625 -0.03125 2.03125 -0.03125 C 2.40625 -0.03125 3.125 -0.03125 3.625 0.015625 L 3.625 -0.640625 L 3.234375 -0.640625 C 2.46875 -0.640625 2.609375 -0.578125 2.609375 -0.9375 L 2.609375 -3.296875 L 3.703125 -3.296875 C 4.234375 -3.296875 4.453125 -3.15625 4.609375 -2.984375 C 5 -2.625 4.9375 -2.484375 4.9375 -1.78125 C 4.9375 -1.09375 5 -0.625 5.421875 -0.234375 C 5.796875 0.078125 6.453125 0.21875 6.8125 0.21875 C 7.609375 0.21875 7.9375 -0.75 7.9375 -1.03125 Z M 5.328125 -5.171875 C 5.328125 -4.109375 4.78125 -3.84375 3.65625 -3.84375 L 2.609375 -3.84375 L 2.609375 -6.265625 C 2.609375 -6.53125 2.46875 -6.453125 2.71875 -6.484375 C 2.796875 -6.5 3.109375 -6.5 3.3125 -6.5 C 4.171875 -6.5 5.328125 -6.65625 5.328125 -5.171875 Z M 5.328125 -5.171875 "/>
</symbol>
<symbol overflow="visible" id="glyph0-18">
<path style="stroke:none;" d="M 7.109375 -1.984375 C 7.109375 -2.828125 6.15625 -3.765625 4.921875 -3.890625 L 4.921875 -3.5625 C 5.890625 -3.734375 6.8125 -4.515625 6.8125 -5.28125 C 6.8125 -6.171875 5.6875 -7.140625 4.34375 -7.140625 L 0.421875 -7.140625 L 0.421875 -6.5 L 0.8125 -6.5 C 1.578125 -6.5 1.453125 -6.546875 1.453125 -6.1875 L 1.453125 -0.9375 C 1.453125 -0.578125 1.578125 -0.640625 0.8125 -0.640625 L 0.421875 -0.640625 L 0.421875 0 L 4.625 0 C 6 0 7.109375 -1.046875 7.109375 -1.984375 Z M 5.546875 -5.28125 C 5.546875 -4.578125 5.109375 -3.984375 3.984375 -3.984375 L 2.578125 -3.984375 L 2.578125 -6.265625 C 2.578125 -6.59375 2.453125 -6.5 2.921875 -6.5 L 4.28125 -6.5 C 5.296875 -6.5 5.546875 -5.84375 5.546875 -5.28125 Z M 5.796875 -2 C 5.796875 -1.25 5.359375 -0.640625 4.296875 -0.640625 L 2.921875 -0.640625 C 2.453125 -0.640625 2.578125 -0.53125 2.578125 -0.875 L 2.578125 -3.421875 L 4.4375 -3.421875 C 5.46875 -3.421875 5.796875 -2.703125 5.796875 -2 Z M 5.796875 -2 "/>
</symbol>
<symbol overflow="visible" id="glyph0-19">
<path style="stroke:none;" d="M 6.078125 -0.15625 L 6.078125 -0.578125 C 5.609375 -0.578125 5.5 -0.5625 5.125 -1.0625 L 3.515625 -3.265625 C 5.28125 -4.890625 5.390625 -4.84375 5.96875 -4.890625 L 5.96875 -5.46875 C 5.59375 -5.4375 5.4375 -5.421875 5.1875 -5.421875 C 4.921875 -5.421875 4.40625 -5.4375 4.109375 -5.46875 L 4.109375 -4.890625 C 4.375 -4.859375 4.28125 -4.953125 4.28125 -4.84375 C 4.28125 -4.671875 4.15625 -4.578125 4 -4.4375 L 2.015625 -2.703125 L 2.015625 -4.5625 C 2.015625 -4.96875 1.90625 -4.875 2.734375 -4.875 L 2.734375 -5.46875 L 1.5 -5.421875 L 0.265625 -5.46875 L 0.265625 -4.875 C 1.09375 -4.875 0.984375 -4.96875 0.984375 -4.5625 L 0.984375 -0.875 C 0.984375 -0.46875 1.09375 -0.578125 0.265625 -0.578125 L 0.265625 0.015625 L 1.5 -0.03125 L 2.734375 0.015625 L 2.734375 -0.578125 C 1.90625 -0.578125 2.015625 -0.46875 2.015625 -0.875 L 2.015625 -1.9375 L 2.796875 -2.625 L 4.03125 -0.96875 C 4.140625 -0.84375 4.203125 -0.75 4.203125 -0.625 C 4.203125 -0.421875 4.125 -0.578125 3.890625 -0.578125 L 3.890625 0.015625 L 5.078125 -0.03125 C 5.359375 -0.03125 5.65625 -0.015625 6.078125 0.015625 Z M 6.078125 -0.15625 "/>
</symbol>
<symbol overflow="visible" id="glyph0-20">
<path style="stroke:none;" d="M 6.0625 -0.84375 C 6.0625 -0.890625 5.90625 -1.15625 5.78125 -1.15625 C 5.671875 -1.15625 5.5 -0.890625 5.5 -0.859375 C 5.4375 -0.25 5.234375 -0.40625 5.15625 -0.40625 C 4.96875 -0.40625 4.96875 -0.484375 4.921875 -0.59375 C 4.828125 -0.75 4.828125 -0.921875 4.6875 -1.5625 C 4.609375 -1.90625 4.359375 -2.578125 3.625 -2.875 L 3.625 -2.546875 C 4.3125 -2.71875 5.078125 -3.328125 5.078125 -3.9375 C 5.078125 -4.640625 4.046875 -5.453125 2.84375 -5.453125 L 0.296875 -5.453125 L 0.296875 -4.875 C 1.109375 -4.875 1 -4.96875 1 -4.5625 L 1 -0.875 C 1 -0.46875 1.109375 -0.578125 0.296875 -0.578125 L 0.296875 0.015625 L 1.5 -0.03125 L 2.71875 0.015625 L 2.71875 -0.578125 C 1.90625 -0.578125 2 -0.46875 2 -0.875 L 2 -2.453125 L 2.75 -2.453125 C 3 -2.453125 3.15625 -2.5 3.453125 -2.1875 C 3.671875 -1.96875 3.625 -1.828125 3.625 -1.390625 C 3.625 -0.84375 3.671875 -0.546875 3.984375 -0.234375 C 4.171875 -0.0625 4.671875 0.140625 5.125 0.140625 C 5.765625 0.140625 6.0625 -0.640625 6.0625 -0.84375 Z M 3.921875 -3.9375 C 3.921875 -3.546875 4.03125 -3 2.734375 -3 L 2 -3 L 2 -4.609375 C 2 -4.859375 1.96875 -4.84375 2 -4.859375 C 2.046875 -4.875 2.3125 -4.875 2.5 -4.875 C 3.234375 -4.875 3.921875 -5 3.921875 -3.9375 Z M 3.921875 -3.9375 "/>
</symbol>
<symbol overflow="visible" id="glyph0-21">
<path style="stroke:none;" d="M 6.078125 -1.96875 L 6.078125 -2.40625 C 5.609375 -2.375 5.203125 -2.359375 4.9375 -2.359375 C 4.578125 -2.359375 4 -2.359375 3.5 -2.40625 L 3.5 -1.8125 L 3.859375 -1.8125 C 4.609375 -1.8125 4.5 -1.859375 4.5 -1.46875 L 4.5 -1.171875 C 4.5 -0.9375 4.546875 -0.8125 4.140625 -0.59375 C 3.921875 -0.453125 3.703125 -0.4375 3.40625 -0.4375 C 2.796875 -0.4375 1.515625 -0.59375 1.515625 -2.71875 C 1.515625 -4.90625 2.875 -5.015625 3.359375 -5.015625 C 4.09375 -5.015625 4.765625 -4.65625 4.953125 -3.390625 C 4.96875 -3.3125 5.140625 -3.09375 5.25 -3.09375 C 5.375 -3.09375 5.515625 -3.3125 5.515625 -3.5 L 5.515625 -5.1875 C 5.515625 -5.34375 5.375 -5.59375 5.28125 -5.59375 C 5.21875 -5.59375 5.09375 -5.515625 5.015625 -5.421875 L 4.71875 -5.046875 C 4.46875 -5.328125 3.84375 -5.59375 3.296875 -5.59375 C 1.796875 -5.59375 0.34375 -4.25 0.34375 -2.71875 C 0.34375 -1.234375 1.75 0.140625 3.3125 0.140625 C 3.65625 0.140625 4.5 0.046875 4.78125 -0.34375 C 4.765625 -0.375 5.171875 -0.015625 5.28125 -0.015625 C 5.375 -0.015625 5.515625 -0.25 5.515625 -0.40625 L 5.515625 -1.53125 C 5.515625 -1.90625 5.40625 -1.8125 6.078125 -1.8125 Z M 6.078125 -1.96875 "/>
</symbol>
<symbol overflow="visible" id="glyph1-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="glyph1-1">
<path style="stroke:none;" d="M 4.359375 -0.0625 C 5.90625 -0.640625 7.375 -2.421875 7.375 -4.34375 C 7.375 -5.953125 6.3125 -7.03125 4.828125 -7.03125 C 2.6875 -7.03125 0.484375 -4.765625 0.484375 -2.4375 C 0.484375 -0.78125 1.609375 0.21875 3.046875 0.21875 C 3.296875 0.21875 3.625 0.171875 4.015625 0.0625 C 3.984375 0.6875 3.984375 0.703125 3.984375 0.84375 C 3.984375 1.15625 3.984375 1.9375 4.8125 1.9375 C 5.984375 1.9375 6.46875 0.109375 6.46875 0 C 6.46875 -0.0625 6.40625 -0.09375 6.359375 -0.09375 C 6.28125 -0.09375 6.265625 -0.046875 6.234375 0.015625 C 6 0.71875 5.421875 0.96875 5.078125 0.96875 C 4.609375 0.96875 4.46875 0.703125 4.359375 -0.0625 Z M 2.484375 -0.140625 C 1.703125 -0.453125 1.359375 -1.21875 1.359375 -2.125 C 1.359375 -2.8125 1.625 -4.234375 2.375 -5.296875 C 3.109375 -6.3125 4.046875 -6.78125 4.78125 -6.78125 C 5.765625 -6.78125 6.5 -6 6.5 -4.671875 C 6.5 -3.671875 5.984375 -1.328125 4.3125 -0.40625 C 4.265625 -0.75 4.171875 -1.46875 3.4375 -1.46875 C 2.90625 -1.46875 2.421875 -0.984375 2.421875 -0.453125 C 2.421875 -0.265625 2.484375 -0.15625 2.484375 -0.140625 Z M 3.09375 -0.03125 C 2.953125 -0.03125 2.640625 -0.03125 2.640625 -0.453125 C 2.640625 -0.859375 3.015625 -1.25 3.4375 -1.25 C 3.859375 -1.25 4.046875 -1.015625 4.046875 -0.40625 C 4.046875 -0.265625 4.03125 -0.25 3.9375 -0.203125 C 3.671875 -0.09375 3.375 -0.03125 3.09375 -0.03125 Z M 3.09375 -0.03125 "/>
</symbol>
</g>
<clipPath id="clip1">
  <path d="M 279 109 L 330 109 L 330 184.382813 L 279 184.382813 Z M 279 109 "/>
</clipPath>
<clipPath id="clip2">
  <path d="M 489 76 L 512.429688 76 L 512.429688 106 L 489 106 Z M 489 76 "/>
</clipPath>
</defs>
<g id="surface1">
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 8.503469 -0.0019375 C 8.503469 4.697281 4.694875 8.505875 -0.0004375 8.505875 C -4.69575 8.505875 -8.504344 4.697281 -8.504344 -0.0019375 C -8.504344 -4.69725 -4.69575 -8.505844 -0.0004375 -8.505844 C 4.694875 -8.505844 8.503469 -4.69725 8.503469 -0.0019375 Z M 8.503469 -0.0019375 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-1" x="34.955" y="11.906"/>
  <use xlink:href="#glyph0-2" x="41.042173" y="11.906"/>
  <use xlink:href="#glyph0-3" x="46.601326" y="11.906"/>
  <use xlink:href="#glyph0-2" x="51.71216" y="11.906"/>
  <use xlink:href="#glyph0-4" x="57.271314" y="11.906"/>
  <use xlink:href="#glyph0-5" x="63.159234" y="11.906"/>
  <use xlink:href="#glyph0-6" x="69.047154" y="11.906"/>
  <use xlink:href="#glyph0-7" x="72.055871" y="11.906"/>
  <use xlink:href="#glyph0-8" x="78.39211" y="11.906"/>
</g>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -42.519969 -40.904281 L -25.512156 -40.904281 L -25.512156 -23.896469 L -42.519969 -23.896469 Z M -42.519969 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.79701;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -6.516063 -6.208969 L -21.637156 -20.611312 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 34.714844 50.035156 C 35.890625 49.269531 38.054688 48.285156 39.804688 47.871094 L 37.125 45.058594 C 36.625 46.785156 35.535156 48.894531 34.714844 50.035156 "/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -42.519969 -64.798812 C -42.519969 -60.1035 -46.328563 -56.294906 -51.023875 -56.294906 C -55.719188 -56.294906 -59.527781 -60.1035 -59.527781 -64.798812 C -59.527781 -69.498031 -55.719188 -73.302719 -51.023875 -73.302719 C -46.328563 -73.302719 -42.519969 -69.498031 -42.519969 -64.798812 Z M -42.519969 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -38.738719 -41.404281 L -46.980906 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -8.504344 -64.798812 C -8.504344 -60.1035 -12.312938 -56.294906 -17.00825 -56.294906 C -21.703563 -56.294906 -25.512156 -60.1035 -25.512156 -64.798812 C -25.512156 -69.498031 -21.703563 -73.302719 -17.00825 -73.302719 C -12.312938 -73.302719 -8.504344 -69.498031 -8.504344 -64.798812 Z M -8.504344 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:1.19553;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -29.293406 -41.404281 L -23.859813 -51.748031 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 38.535156 83.039063 C 38.027344 81.386719 37.589844 78.496094 37.6875 76.28125 L 33.453125 78.503906 C 35.332031 79.679688 37.460938 81.683594 38.535156 83.039063 "/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -8.504344 -40.904281 L 8.503469 -40.904281 L 8.503469 -23.896469 L -8.504344 -23.896469 Z M -8.504344 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M -0.0004375 -9.001937 L -0.0004375 -23.69725 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 25.511281 -40.904281 L 42.519094 -40.904281 L 42.519094 -23.896469 L 25.511281 -23.896469 Z M 25.511281 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 6.519094 -6.208969 L 25.312062 -24.111312 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 25.511281 -64.798812 C 25.511281 -60.1035 21.706594 -56.294906 17.007375 -56.294906 C 12.312062 -56.294906 8.503469 -60.1035 8.503469 -64.798812 C 8.503469 -69.498031 12.312062 -73.302719 17.007375 -73.302719 C 21.706594 -73.302719 25.511281 -69.498031 25.511281 -64.798812 Z M 25.511281 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 29.448781 -41.1035 L 21.05425 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 59.526906 -64.798812 C 59.526906 -60.1035 55.722219 -56.294906 51.023 -56.294906 C 46.327687 -56.294906 42.519094 -60.1035 42.519094 -64.798812 C 42.519094 -69.498031 46.327687 -73.302719 51.023 -73.302719 C 55.722219 -73.302719 59.526906 -69.498031 59.526906 -64.798812 Z M 59.526906 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 38.581594 -41.1035 L 46.980031 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 139.597219 -0.0019375 C 139.597219 4.697281 135.788625 8.505875 131.093313 8.505875 C 126.394094 8.505875 122.589406 4.697281 122.589406 -0.0019375 C 122.589406 -4.69725 126.394094 -8.505844 131.093313 -8.505844 C 135.788625 -8.505844 139.597219 -4.69725 139.597219 -0.0019375 Z M 139.597219 -0.0019375 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-9" x="165.508" y="12.12"/>
  <use xlink:href="#glyph0-10" x="172.900279" y="12.12"/>
  <use xlink:href="#glyph0-11" x="179.007377" y="12.12"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-12" x="183.899033" y="12.12"/>
  <use xlink:href="#glyph0-8" x="190.006132" y="12.12"/>
  <use xlink:href="#glyph0-13" x="196.11323" y="12.12"/>
  <use xlink:href="#glyph0-6" x="200.676119" y="12.12"/>
  <use xlink:href="#glyph0-7" x="203.684836" y="12.12"/>
  <use xlink:href="#glyph0-8" x="210.021076" y="12.12"/>
</g>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 88.573781 -40.904281 L 105.581594 -40.904281 L 105.581594 -23.896469 L 88.573781 -23.896469 Z M 88.573781 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 124.792531 -6.001937 L 105.780813 -24.111312 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 88.573781 -64.798812 C 88.573781 -60.1035 84.765187 -56.294906 80.069875 -56.294906 C 75.370656 -56.294906 71.565969 -60.1035 71.565969 -64.798812 C 71.565969 -69.498031 75.370656 -73.302719 80.069875 -73.302719 C 84.765187 -73.302719 88.573781 -69.498031 88.573781 -64.798812 Z M 88.573781 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 92.511281 -41.1035 L 84.112844 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.79701;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 122.589406 -64.798812 C 122.589406 -60.1035 118.780813 -56.294906 114.0855 -56.294906 C 109.386281 -56.294906 105.581594 -60.1035 105.581594 -64.798812 C 105.581594 -69.498031 109.386281 -73.302719 114.0855 -73.302719 C 118.780813 -73.302719 122.589406 -69.498031 122.589406 -64.798812 Z M 122.589406 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 101.644094 -41.1035 L 109.948781 -56.919906 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 105.581594 -105.705062 L 122.589406 -105.705062 L 122.589406 -88.69725 L 105.581594 -88.69725 Z M 105.581594 -105.705062 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:1.39478;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 114.0855 -73.701156 L 114.0855 -88.19725 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 122.589406 -40.904281 L 139.597219 -40.904281 L 139.597219 -23.896469 L 122.589406 -23.896469 Z M 122.589406 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 131.093313 -8.705062 L 131.093313 -23.69725 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 156.605031 -40.904281 L 173.612844 -40.904281 L 173.612844 -23.896469 L 156.605031 -23.896469 Z M 156.605031 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 137.394094 -6.001937 L 156.405813 -24.111312 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 156.605031 -64.798812 C 156.605031 -60.1035 152.796438 -56.294906 148.101125 -56.294906 C 143.405813 -56.294906 139.597219 -60.1035 139.597219 -64.798812 C 139.597219 -69.498031 143.405813 -73.302719 148.101125 -73.302719 C 152.796438 -73.302719 156.605031 -69.498031 156.605031 -64.798812 Z M 156.605031 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 160.542531 -41.1035 L 152.144094 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 190.620656 -64.798812 C 190.620656 -60.1035 186.812063 -56.294906 182.11675 -56.294906 C 177.421438 -56.294906 173.612844 -60.1035 173.612844 -64.798812 C 173.612844 -69.498031 177.421438 -73.302719 182.11675 -73.302719 C 186.812063 -73.302719 190.620656 -69.498031 190.620656 -64.798812 Z M 190.620656 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 169.675344 -41.1035 L 178.073781 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 270.390188 -0.0019375 C 270.390188 4.697281 266.581594 8.505875 261.886281 8.505875 C 257.190969 8.505875 253.382375 4.697281 253.382375 -0.0019375 C 253.382375 -4.69725 257.190969 -8.505844 261.886281 -8.505844 C 266.581594 -8.505844 270.390188 -4.69725 270.390188 -0.0019375 Z M 270.390188 -0.0019375 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-1" x="270.302" y="11.294"/>
  <use xlink:href="#glyph0-6" x="276.389173" y="11.294"/>
  <use xlink:href="#glyph0-14" x="279.39789" y="11.294"/>
  <use xlink:href="#glyph0-15" x="286.839982" y="11.294"/>
  <use xlink:href="#glyph0-3" x="292.947081" y="11.294"/>
  <use xlink:href="#glyph0-12" x="298.057915" y="11.294"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-5" x="303.497516" y="11.294"/>
  <use xlink:href="#glyph0-6" x="309.385437" y="11.294"/>
  <use xlink:href="#glyph0-7" x="312.394154" y="11.294"/>
  <use xlink:href="#glyph0-8" x="318.730393" y="11.294"/>
  <use xlink:href="#glyph0-16" x="324.837491" y="11.294"/>
  <use xlink:href="#glyph0-17" x="330.346831" y="11.294"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-7" x="338.028027" y="11.294"/>
  <use xlink:href="#glyph0-3" x="344.364266" y="11.294"/>
  <use xlink:href="#glyph0-3" x="349.4751" y="11.294"/>
  <use xlink:href="#glyph0-7" x="354.585934" y="11.294"/>
  <use xlink:href="#glyph0-15" x="360.922173" y="11.294"/>
  <use xlink:href="#glyph0-5" x="367.029272" y="11.294"/>
</g>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 219.36675 -40.904281 L 236.374563 -40.904281 L 236.374563 -23.896469 L 219.36675 -23.896469 Z M 219.36675 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 255.5855 -6.001937 L 236.573781 -24.111312 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 219.36675 -64.798812 C 219.36675 -60.1035 215.558156 -56.294906 210.862844 -56.294906 C 206.167531 -56.294906 202.358938 -60.1035 202.358938 -64.798812 C 202.358938 -69.498031 206.167531 -73.302719 210.862844 -73.302719 C 215.558156 -73.302719 219.36675 -69.498031 219.36675 -64.798812 Z M 219.36675 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 223.30425 -41.1035 L 214.905813 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 253.382375 -64.798812 C 253.382375 -60.1035 249.573781 -56.294906 244.878469 -56.294906 C 240.183156 -56.294906 236.374563 -60.1035 236.374563 -64.798812 C 236.374563 -69.498031 240.183156 -73.302719 244.878469 -73.302719 C 249.573781 -73.302719 253.382375 -69.498031 253.382375 -64.798812 Z M 253.382375 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 232.437063 -41.1035 L 240.8355 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 236.374563 -105.705062 L 253.382375 -105.705062 L 253.382375 -88.69725 L 236.374563 -88.69725 Z M 236.374563 -105.705062 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 244.878469 -73.501937 L 244.878469 -88.19725 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph1-1" x="300.664" y="179.124"/>
</g>
<g clip-path="url(#clip1)" clip-rule="nonzero">
<path style="fill:none;stroke-width:1.59404;stroke-linecap:round;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-dasharray:0,3.1881;stroke-miterlimit:10;" d="M 244.878469 -106.201156 C 244.878469 -107.69725 247.714406 -108.443344 247.714406 -109.939437 C 247.714406 -111.021469 246.331594 -111.955062 244.878469 -112.927719 C 243.425344 -113.900375 242.042531 -114.833969 242.042531 -115.916 C 242.042531 -116.998031 243.425344 -117.931625 244.878469 -118.904281 C 246.331594 -119.880844 247.714406 -120.810531 247.714406 -121.892562 C 247.714406 -122.974594 246.331594 -123.908187 244.878469 -124.88475 C 243.425344 -125.857406 242.042531 -126.791 242.042531 -127.873031 C 242.042531 -129.365219 244.878469 -130.115219 244.878469 -131.607406 L 244.878469 -135.775375 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
</g>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 304.605469 168.796875 C 304.984375 166.777344 306.121094 163.5 307.445313 161.226563 L 301.765625 161.226563 C 303.089844 163.5 304.226563 166.777344 304.605469 168.796875 "/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 253.382375 -40.904281 L 270.390188 -40.904281 L 270.390188 -23.896469 L 253.382375 -23.896469 Z M 253.382375 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 261.886281 -8.705062 L 261.886281 -23.69725 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 287.398 -40.904281 L 304.405813 -40.904281 L 304.405813 -23.896469 L 287.398 -23.896469 Z M 287.398 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 268.187063 -6.001937 L 287.198781 -24.111312 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 287.398 -64.798812 C 287.398 -60.1035 283.593313 -56.294906 278.894094 -56.294906 C 274.198781 -56.294906 270.390188 -60.1035 270.390188 -64.798812 C 270.390188 -69.498031 274.198781 -73.302719 278.894094 -73.302719 C 283.593313 -73.302719 287.398 -69.498031 287.398 -64.798812 Z M 287.398 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 291.3355 -41.1035 L 282.940969 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 321.413625 -64.798812 C 321.413625 -60.1035 317.608938 -56.294906 312.909719 -56.294906 C 308.214406 -56.294906 304.405813 -60.1035 304.405813 -64.798812 C 304.405813 -69.498031 308.214406 -73.302719 312.909719 -73.302719 C 317.608938 -73.302719 321.413625 -69.498031 321.413625 -64.798812 Z M 321.413625 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 300.468313 -41.1035 L 308.86675 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 401.483938 -0.0019375 C 401.483938 4.697281 397.675344 8.505875 392.980031 8.505875 C 388.280813 8.505875 384.476125 4.697281 384.476125 -0.0019375 C 384.476125 -4.69725 388.280813 -8.505844 392.980031 -8.505844 C 397.675344 -8.505844 401.483938 -4.69725 401.483938 -0.0019375 Z M 401.483938 -0.0019375 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-18" x="408.995" y="11.837"/>
  <use xlink:href="#glyph0-12" x="416.676195" y="11.837"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-4" x="422.564116" y="11.837"/>
  <use xlink:href="#glyph0-19" x="428.452036" y="11.837"/>
  <use xlink:href="#glyph0-11" x="434.788275" y="11.837"/>
  <use xlink:href="#glyph0-20" x="440.347428" y="11.837"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-7" x="446.125759" y="11.837"/>
  <use xlink:href="#glyph0-11" x="452.461998" y="11.837"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-12" x="457.353655" y="11.837"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-21" x="463.241575" y="11.837"/>
  <use xlink:href="#glyph0-12" x="469.627627" y="11.837"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-5" x="475.067228" y="11.837"/>
  <use xlink:href="#glyph0-6" x="480.955149" y="11.837"/>
  <use xlink:href="#glyph0-7" x="483.963866" y="11.837"/>
  <use xlink:href="#glyph0-8" x="490.300105" y="11.837"/>
</g>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 350.4605 -40.904281 L 367.468313 -40.904281 L 367.468313 -23.896469 L 350.4605 -23.896469 Z M 350.4605 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.79701;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 383.0855 -9.423812 L 367.964406 -23.826156 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 446.1875 32.417969 C 445.007813 33.183594 442.847656 34.167969 441.097656 34.585938 L 443.777344 37.398438 C 444.277344 35.667969 445.367188 33.558594 446.1875 32.417969 "/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 350.4605 -64.798812 C 350.4605 -60.1035 346.651906 -56.294906 341.956594 -56.294906 C 337.257375 -56.294906 333.452688 -60.1035 333.452688 -64.798812 C 333.452688 -69.498031 337.257375 -73.302719 341.956594 -73.302719 C 346.651906 -73.302719 350.4605 -69.498031 350.4605 -64.798812 Z M 350.4605 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 354.237844 -41.404281 L 345.999563 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 384.476125 -64.798812 C 384.476125 -60.1035 380.667531 -56.294906 375.972219 -56.294906 C 371.273 -56.294906 367.468313 -60.1035 367.468313 -64.798812 C 367.468313 -69.498031 371.273 -73.302719 375.972219 -73.302719 C 380.667531 -73.302719 384.476125 -69.498031 384.476125 -64.798812 Z M 384.476125 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:1.19553;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 366.355031 -46.482406 L 371.788625 -56.830062 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 423.414063 67.613281 C 423.921875 69.265625 424.359375 72.15625 424.261719 74.367188 L 428.496094 72.144531 C 426.617188 70.96875 424.488281 68.96875 423.414063 67.613281 "/>
<path style="fill:none;stroke-width:0.99628;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 367.468313 -105.705062 L 384.476125 -105.705062 L 384.476125 -88.69725 L 367.468313 -88.69725 Z M 367.468313 -105.705062 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph1-1" x="431.755" y="125.844"/>
</g>
<path style="fill:none;stroke-width:1.39478;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 375.972219 -80.080062 L 375.972219 -88.19725 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style=" stroke:none;fill-rule:nonzero;fill:rgb(0%,0%,0%);fill-opacity:1;" d="M 435.699219 100.011719 C 435.351563 101.871094 434.304688 104.894531 433.082031 106.984375 L 438.3125 106.984375 C 437.09375 104.894531 436.046875 101.871094 435.699219 100.011719 "/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 384.476125 -40.904281 L 401.483938 -40.904281 L 401.483938 -23.896469 L 384.476125 -23.896469 Z M 384.476125 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 392.980031 -9.001937 L 392.980031 -23.69725 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 418.49175 -40.904281 L 435.499563 -40.904281 L 435.499563 -23.896469 L 418.49175 -23.896469 Z M 418.49175 -40.904281 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 399.495656 -6.208969 L 418.292531 -24.111312 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 418.49175 -64.798812 C 418.49175 -60.1035 414.683156 -56.294906 409.987844 -56.294906 C 405.292531 -56.294906 401.483938 -60.1035 401.483938 -64.798812 C 401.483938 -69.498031 405.292531 -73.302719 409.987844 -73.302719 C 414.683156 -73.302719 418.49175 -69.498031 418.49175 -64.798812 Z M 418.49175 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 422.42925 -41.1035 L 414.030813 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
<g clip-path="url(#clip2)" clip-rule="nonzero">
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 452.507375 -64.798812 C 452.507375 -60.1035 448.698781 -56.294906 444.003469 -56.294906 C 439.308156 -56.294906 435.499563 -60.1035 435.499563 -64.798812 C 435.499563 -69.498031 439.308156 -73.302719 444.003469 -73.302719 C 448.698781 -73.302719 452.507375 -69.498031 452.507375 -64.798812 Z M 452.507375 -64.798812 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
</g>
<path style="fill:none;stroke-width:0.3985;stroke-linecap:butt;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:10;" d="M 431.562063 -41.1035 L 439.9605 -57.095687 " transform="matrix(1,0,0,-1,59.727,26.209)"/>
</g>
</svg>
"""

# ╔═╡ 548124bb-0229-40f3-ba57-e436f37612ec
@bind γ_mcts Slider(0:0.05:1, default=0.95, show_value=true)

# ╔═╡ 60644fcd-dec8-4e0e-bbee-2e29443e61c8
mcts_mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ_mcts, # custom discount for visualization of MCTS policy
    initialstate = 𝒮,
    isterminal   = termination);

# ╔═╡ 7af62c0f-2154-4e0a-86bb-17dd3a84f362
md"First we create the MCTS solver with appropriate input parameters."

# ╔═╡ d69bb3f3-f805-40f8-8f19-6210246ff86c
mcts_solver = MCTSSolver(n_iterations=50,
	                     depth=20,
	                     exploration_constant=5.0,
                         enable_tree_vis=true);

# ╔═╡ 6467f4c6-a59f-4601-8b8e-5627e3386d0e
md"Then we solve the MDP to create a planner (again, we use \"planner\" instead of \"policy\" to indicate that the planner is executed online)."

# ╔═╡ e5143473-6471-43e4-ac20-2907968f35d3
mcts_planner = solve(mcts_solver, mcts_mdp);

# ╔═╡ 6b9b6f24-f4d4-435a-856a-469e5a20f602
md"We can sample a random initial state $s_0$, or pick a starting state."

# ╔═╡ 756cf1c4-ddfb-4173-8287-5eac419093ff
s₀ = State(8,2) # rand(initialstate(mcts_mdp))

# ╔═╡ 6befa3ba-35e8-4b03-a28b-b30ceb9a66d3
md"Similar to the `action` function which returns an action, MCTS also has an `action_info` function which—as the name suggests—returns the action and some information. We want to retrieve the `tree` from the `info` object."

# ╔═╡ 54e28d34-af2f-49df-95f4-d95ab9f6b3ea
aₘ, info = action_info(mcts_planner, s₀);

# ╔═╡ d156dbbd-a63d-48a8-8c15-664df10416f5
md"""
#### Online Tree Visualization
To visualize the online search tree, we can use `D3Trees.jl`.
"""

# ╔═╡ d4cee740-ab00-4078-b2b8-671cd155620a
md"We can control the labels for the state nodes—here we show the $(x,y)$ state and the reward."

# ╔═╡ 76d59916-c550-4d6e-b5c9-8989d67ad871
MCTS.node_tag(s::State) = "($(s.x), $(s.y))\nr = $(reward(mcts_mdp, s))"

# ╔═╡ dc03ec16-f319-4a98-a2ea-66ea5489151c
md"Click to expand the levels of the search tree. The root note is the initial state, and the layers alternate between states and actions."

# ╔═╡ 2af5a9a9-a612-44a6-8be0-1d6cc43fc200
tree = D3Tree(info[:tree], s₀, init_expand=1)

# ╔═╡ 6af23b44-528a-4027-9ba9-e1e91781d83b
md"""
## Simulations

Say we want to simulate many different episodes of the policies we created. `POMDPSimulators` provides this functionality.
"""

# ╔═╡ d613b978-98fa-44aa-ad1d-c51e50e2d12a
md"""
The `simulate` function performs the simulations given one of the simulators listed below.
```julia
simulate(sim::Simulator, m::MDP, p::Policy, s0=rand(initialstate(m)))
```
"""

# ╔═╡ 299dcfef-91d1-48c1-a163-830f340e31df
md"""
There are several simulators we can run, depending on the type of information we are looking for (see the [`POMDPSimulators`](https://juliapomdp.github.io/POMDPSimulators.jl/latest/which/) documentation for more details):
"""

# ╔═╡ 9f651d75-42c6-4a3e-ab88-6b6ed68a36ea
md"""
##### Fast rollout simulations and get the discounted reward [[docs](https://juliapomdp.github.io/POMDPSimulators.jl/latest/rollout/#Rollout)]:
```julia
simulator = RolloutSimulator()
```
"""

# ╔═╡ 15bce1e5-9078-438a-9659-d72cf3440b4e
md"""
##### Closely examime the histories (states, actions, rewards, etc.) [[docs](https://juliapomdp.github.io/POMDPSimulators.jl/latest/history_recorder/#History-Recorder)]:
```julia
simulator = HistoryRecorder()
```
"""

# ╔═╡ 273de0d6-aa6b-49c2-94e7-047879402ef9
md"""
##### Visualize a simulation [[docs](https://juliapomdp.github.io/POMDPSimulators.jl/latest/display/#Display)]:
```julia
simulator = DisplaySimulator()
```
"""

# ╔═╡ 144beb1b-3aec-4daa-a4f4-eb8274b1a009
md"""
##### Evaulate performance on many parallel Monte Carlo simulations [[docs](https://juliapomdp.github.io/POMDPSimulators.jl/latest/parallel/#Parallel)]:
```julia
run_parallel(queue::Vector{Sim})
```
"""

# ╔═╡ fdf3325d-a949-4f39-9067-77bb13ab8ba7
md"""
##### Interact with an environment from the perspective of the policy [[docs](https://juliapomdp.github.io/POMDPSimulators.jl/latest/sim/#sim-function)]:
```julia
sim(mdp, max_steps=100)
```
"""

# ╔═╡ 351a4e76-b13b-4719-9c86-097e0b10e930
md"""
##### Step through each individual simulation step [[docs](https://juliapomdp.github.io/POMDPSimulators.jl/latest/stepthrough/#Stepping-through)]:
```julia
stepthrough(mdp, policy, "s,a,r", max_steps=100)
```
"""

# ╔═╡ b26cc58b-c5ce-4660-889c-e6eb09577dd2
md"""
Here we use the `stepthrough` function to iterate over a simulation and inspect the state `s`, action `a`, and reward `r` at each time step.
"""

# ╔═╡ 06bf7a9e-2fc9-4c58-a1f2-b3bba9fce96f
md"""
Now lets collect a full simulation using `stepthrough` wrapped in `collect`.
"""

# ╔═╡ 908bd8f5-a523-43d9-ae83-7934ca93a226
md"""
### Animated GIF (Single Episode)
"""

# ╔═╡ 58c8c26f-21c9-4ef5-a0e0-712c9bd51150
md"Create GIF of single simulated episode? $(@bind create_episode_gif CheckBox())"

# ╔═╡ 4f13867a-a91e-4251-8d43-746baa6d12a6
isfile("gifs/gridworld_episode.gif") && LocalResource("./gifs/gridworld_episode.gif")

# ╔═╡ a0178751-85c0-4424-bfd0-4df86434855b
md"""
#### Rollout Simulator
We can "rollout" the policy for each algorithm above (e.g., _value iteration_, _Q-learning_, _SARSA_) and collect statistics on the quality of each policy through simulation.
"""

# ╔═╡ 0fdb2e3a-838c-435f-a152-f27ba2413a3b
rollsim = RolloutSimulator();

# ╔═╡ 680c3c05-2011-411e-87f1-7b27544cf8ea
md"""
Now we can simulate a single _episode_ (which picks a random initial state and follows the given policy until we hit a termination state). The return value of the `RolloutSimulator` is the _discounted reward_ (or _discounted return_) that the agent collected (meaning—in the Grid World problem—the final reward is discounted based on the chosen $\gamma$ where:

$$\begin{equation}
\text{discounted reward} = \sum_{t=1}^{\text{horizon}} \gamma^{t-1} r_t
\end{equation}$$

So if our first step ($t=1$) moves into a reward state (which are also terminal states by our Grid World definition), we would directly return that reward value (because $\gamma^0=1$). But if the agent "meanders" over time and then eventually lands in a reward state, our return value will be noticably discounted.
"""

# ╔═╡ 52ef9f86-2172-4600-9ddd-83f24fdc3944
md"""
Now we'll run $N_\text{sim}$ number of simulations—each representing a single episode—and collect the reward to compute some statistics.
"""

# ╔═╡ 8be067e6-e12d-4262-b3d7-368e17db8e93
md"""
Run large simulation: $(@bind show_simulation CheckBox())
"""

# ╔═╡ a4e1b417-60c9-475e-aa16-b2112f7b47cf
md"
## Custom Reward Color Scheme
We use the `ColorSchemes` package to define a custom red-to-green color scheme for cost-to-rewards."

# ╔═╡ 87710f14-a7ec-4983-b105-8090c4dd1463
cmap = ColorScheme([Colors.RGB(180/255, 0.0, 0.0), Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "threetone, red, white, and green")

# ╔═╡ 25d170e9-1c26-4058-96fc-04ab47964b51
# plot the U values (maximum Q over the actions)
# x = row, y = column, z = U-value
function plot_grid_world(mdp::MDP,
        policy::Policy=NothingPolicy(),
        iter=0,
        discount=NaN;
        outline=true,
        show_policy=true,
        extra_title=isnan(discount) ? "" : " (iter=$iter, γ=$discount)",
        show_rewards=false,
        outline_state::Union{State, Nothing}=nothing)
    
    gr()
	
    if policy isa NothingPolicy
        # override when the policy is empty
        show_policy = false
    end
    
    if iter == 0
        # solver has not been run yet, so we just plot the raw rewards
        # overwrite policy at time=0 to be emp
        U = get_rewards(mdp, policy)
    else
        # otherwise, use the Value Function to get the values (i.e., utility)
        U = values(mdp, policy)
    end

    # reshape to grid
    (xmax, ymax) = params.size
    Uxy = reshape(U, xmax, ymax)


    # plot values (i.e the U matrix)
    fig = heatmap(Uxy',
                  legend=:none,
                  aspect_ratio=:equal,
                  framestyle=:box,
                  tickdirection=:out,
                  color=cmap.colors)
    xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    
    if show_rewards
        for s in filter(s->reward(mdp, s) != 0, states(mdp))
            r = reward(mdp, s)
            annotate!([(s.x, s.y, (r, :white, :center, 12, "Computer Modern"))])
        end
    end
    
    for x in 1:xmax, y in 1:ymax
        # display policy on the plot as arrows
        if show_policy
            grid = policy_grid(policy, xmax, ymax)
            annotate!([(x, y, (grid[x,y], :center, 12, "Computer Modern"))])
        end
        if outline
            rect = rectangle(1, 1, x - 0.5, y - 0.5)
            plot!(rect, fillalpha=0, linecolor=:gray)
        end
    end

    if !isnothing(outline_state)
        terminal_states = filter(s->reward(mdp, s) != 0, states(mdp))
        color = (outline_state in terminal_states) ? "yellow" : "blue"
        rect = rectangle(1, 1, outline_state.x - 0.5, outline_state.y - 0.5)
        plot!(rect, fillalpha=0, linecolor=color)
    end

    title!("Grid World Policy Plot$extra_title")

    return fig
end

# ╔═╡ c85a316a-acd3-4aa3-8b65-28a332950240
render = plot_grid_world

# ╔═╡ 5af462ea-781b-4509-98de-c68c29ec81fd
mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = 𝒮,
    isterminal   = termination,
	render       = render);

# ╔═╡ 8d683391-75eb-4f5d-8849-7dd77720b5bf
render(mdp)

# ╔═╡ e67994b8-d519-4c24-9d61-5cbef1629baa
render(mdp; show_rewards=true)

# ╔═╡ ba95fdd5-d643-4de2-8235-90e2b5651410
policy = solve(solver, mdp)

# ╔═╡ 4ef71be8-4768-4f58-bf31-78a895452617
render(mdp, policy, 30, γ; outline=false)

# ╔═╡ 32668e09-b12b-43c8-862f-b6f1a77557ec
one_based_policy!(policy) # change default initial policy from all zeros to all ones

# ╔═╡ 3b883115-7d45-4439-8d04-48df4684452f
aᵣ = action(policy, sᵣ)

# ╔═╡ 8e3c5337-951e-495c-a0e7-b050660d0192
value(policy, sᵣ)

# ╔═╡ 7ee1a2f0-0210-4e89-98e5-73f18fb178b1
begin
	_U = map(a->(a, value(policy, sᵣ, a), action(policy, sᵣ) == a), actions(mdp))
	Markdown.parse("""
\$s = ($(sᵣ.x), $(sᵣ.y))\\qquad a = \\text{$(action(policy, sᵣ))}\$

Action \$a \\in \\mathcal{A}\$   | Value from \$Q(s,a)\$  |  Selected from \$\\pi(s)\$
:------------------------------- | :------------------- | :----------
UP                               | $(_U[1][2])          | $(_U[1][3] ? true : "")
DOWN                             | $(_U[2][2])          | $(_U[2][3] ? true : "")
LEFT                             | $(_U[3][2])          | $(_U[3][3] ? true : "")
RIGHT                            | $(_U[4][2])          | $(_U[4][3] ? true : "")
""")
end

# ╔═╡ 9eba07a6-2753-42c5-8581-3fb7489c066a
distr = transition(mdp, sᵣ, aᵣ)

# ╔═╡ 6b77edd5-79d1-4b0a-b6e2-f8b83af52db7
U(π, s) = maximum(a->Q(π, s, a), actions(mdp)) # utility

# ╔═╡ e06f597d-dc01-4dea-b63a-5f61d49170e0
U(policy, sᵣ) == value(policy, sᵣ)

# ╔═╡ 50ce080c-235f-425a-925d-65f1c29e7d60
with_terminal() do
	for (s,a,r) in stepthrough(mdp, policy, "s,a,r", max_steps=100)
		@info "In state ($(s.x), $(s.y)), taking action $a, receiving reward $r"
	end
end		

# ╔═╡ bc474e19-7d3e-481c-b36c-e231cd72db94
steps = collect(stepthrough(mdp, policy, "s,a,r", max_steps=100))

# ╔═╡ 02f590b6-b034-4ac4-afd4-76bde37e68b6
md"Simulation time step: $(@bind t Slider(1:length(steps), default=1))"

# ╔═╡ 87c715e7-29aa-4111-a0d1-1c5049341142
simulate(rollsim, mdp, policy)

# ╔═╡ 7124f670-d5ba-4e9b-86cf-48f2be0bc2a2
if show_simulation
	using Statistics
	mean_std(X) = (μ=mean(X), σ=std(X), r=X)
	N_sim = 10_000

	stats_vi = mean_std([simulate(rollsim, mdp, policy) for _ in 1:N_sim])
	stats_ql = mean_std([simulate(rollsim, mdp, q_learning_policy) for _ in 1:N_sim])
	stats_sarsa = mean_std([simulate(rollsim, mdp, sarsa_policy) for _ in 1:N_sim])

	results = (value_iteration=stats_vi, q_learning=stats_ql, sarsa=stats_sarsa)
end

# ╔═╡ 92bb38e3-00e2-4d5b-a511-4f0580777aa5
if show_simulation
    using RollingFunctions
    using LaTeXStrings
    window = 500
    
    rolling_mean_vi = rolling(mean, results.value_iteration.r, window)
    rolling_mean_ql = rolling(mean, results.q_learning.r, window)
    rolling_mean_sarsa = rolling(mean, results.sarsa.r, window)
    rolling_error_vi = log.(rolling(std, results.value_iteration.r, window)/3)
    rolling_error_ql = log.(rolling(std, results.q_learning.r, window)/3)
    rolling_error_sarsa = log.(rolling(std, results.sarsa.r, window)/3)
    num_simulations = rolling(minimum, 1:N_sim, window)

    fig = plot(num_simulations, rolling_mean_vi,
		ribbon=rolling_error_vi, fillalpha=0.2,
        color="blue", label="Value iteration", legend=(0.8, 0.65))
    plot!(num_simulations, rolling_mean_ql,
		ribbon=rolling_error_ql, fillalpha=0.2,
        color="red", label="Q-learning")
    plot!(num_simulations, rolling_mean_sarsa,
		ribbon=rolling_error_sarsa, fillalpha=0.2,
        color="black", label="SARSA")

    xlabel!("number of simulations")
    ylabel!("mean reward")
    title!("Rolling Mean")
    xticks!(0:2000:N_sim, latexstring.(0:2000:N_sim))
    yticks!(1:6, latexstring.(1:6))
    fig
end

# ╔═╡ c5fbf696-3e5c-4b59-be4d-9a43f30d6211
begin
	grid_plot = render(mdp, policy, 30; outline=false, outline_state=sᵣ)
	distr_plot = plot_transition_probability(distr)
	plot(grid_plot, distr_plot, layout=2)
end

# ╔═╡ 4bb93999-e6b5-4590-8605-9bfe83778890
viz = render(vi_mdp, vi_policy, vi_iterations, γ_vi)

# ╔═╡ edc2ec3c-95d9-4079-9efe-a39ea2053e15
savefig(viz, "gridworld.svg")

# ╔═╡ 49cd3c8b-d633-40e7-99f4-57a948b53c92
function create_value_iteration_gif()
	frames = Frames(MIME("image/png"), fps=2)
	push!(frames, render(vi_mdp, NothingPolicy(), 0, γ_vi; outline=true))
	last_frame = nothing
	for iter in 0:21
		local_solver = ValueIterationSolver(max_iterations=iter)
		local_policy = solve(local_solver, vi_mdp)
		one_based_policy!(local_policy)
		last_frame = render(vi_mdp, local_policy, iter, γ_vi; outline=false)
		push!(frames, last_frame)
	end
	[push!(frames, last_frame) for _ in 1:10] # duplicate last frame
	!isdir("gifs") && mkdir("gifs") # create "gifs" directory
	write("gifs/gridworld_vi.gif", frames)
	LocalResource("./gifs/gridworld_vi.gif")
end

# ╔═╡ dfa19121-b929-467a-a4ab-3f6563f200cf
create_gif ? create_value_iteration_gif() : LocalResource("./gifs/gridworld_vi.gif")

# ╔═╡ ea27b654-d225-4708-bf3f-fab306f51c72
function create_discount_gif()
	frames_γ = Frames(MIME("image/png"), fps=2)
	last_frame_γ = nothing
	for γ_iter in 0:0.05:1
		local_solver = ValueIterationSolver(max_iterations=30)

		# local MDP to play around with γ
		vi_γ_mdp = QuickMDP(GridWorld,
			states       = 𝒮,
			actions      = 𝒜,
			transition   = T,
			reward       = R,
			discount     = γ_iter,
			initialstate = 𝒮,
			isterminal   = termination);

		local_π = solve(local_solver, vi_γ_mdp)
		one_based_policy!(local_π)
		last_frame_γ = render(vi_γ_mdp, local_π, 30, γ_iter; outline=false)
		push!(frames_γ, last_frame_γ)
	end
	[push!(frames_γ, last_frame_γ) for _ in 1:10] # duplicate last frame
	!isdir("gifs") && mkdir("gifs") # create "gifs" directory
	write("gifs/gridworld_vi_γ.gif", frames_γ)
	LocalResource("./gifs/gridworld_vi_γ.gif")
end

# ╔═╡ b0a444df-0b41-430b-924f-83075944368a
create_gif ? create_discount_gif() : LocalResource("./gifs/gridworld_vi_γ.gif")

# ╔═╡ ee5ae9e6-b837-4521-bac7-cf069765d278
render(mdp, q_learning_policy, n_episodes_q, γ)

# ╔═╡ fb0a10c6-ca0b-480c-93d6-09aac5cc96ed
render(mdp, sarsa_policy, n_episodes_sarsa, γ)

# ╔═╡ caad8138-251b-4644-9217-3e3bba49e357
render(mdp, policy; outline_state=steps[t].s, outline=false)

# ╔═╡ 50121862-2d65-4f52-aa2f-69d23f8968e8
function create_simulated_episode_gif()
	sim_frames = Frames(MIME("image/png"), fps=2)
	for i in 1:length(steps)
		frame_i = render(mdp, policy;
			outline_state=steps[i].s, outline=false)
		push!(sim_frames, frame_i)
	end
	!isdir("gifs") && mkdir("gifs") # create "gifs" directory
	write("gifs/gridworld_episode.gif", sim_frames)
end

# ╔═╡ 0b6b147f-289c-466b-b859-e0b75f2d7823
create_episode_gif && create_simulated_episode_gif();

# ╔═╡ 92cdb96c-e636-4308-bf49-6d34246d3256
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

# ╔═╡ 228fb0a2-fa38-401d-9793-3e487f79d34d
md"""
## Latexify
Shout-out to Niklas Korsbo for the [`Latexify`](https://github.com/korsbo/Latexify.jl) package where we can print Julia expressions as LaTeX formated equations. Definitely not required for anything POMDPs related—just nice to have.
"""

# ╔═╡ e2d85e1d-3368-43b1-93db-d24023805b9f
@latexify Q(s,a) = Q(s,a) + α(r + γQ(s′, a′) - Q(s,a))

# ╔═╡ 9cc9137f-4585-43c9-b60d-2cd80d7766ff
md"""
## References
1. Maxim Egorov, Zachary N. Sunberg, Edward Balaban, Tim A. Wheeler, Jayesh K. Gupta, and Mykel J. Kochenderfer, "POMDPs.jl: A Framework for Sequential Decision Making under Uncertainty", *Journal of Machine Learning Research*, vol. 18, no. 26, pp. 1–5, 2017. [http://jmlr.org/papers/v18/16-300.html](http://jmlr.org/papers/v18/16-300.html)

2. Mykel J. Kochenderfer, Tim A. Wheeler, and Kyle H. Wray, "Algorithms for Decision Making", *MIT Press*, 2022. [https://algorithmsbook.com](https://algorithmsbook.com)

3. Christopher J. C. H. Watkins, "Learning from Delayed Rewards", *PhD Thesis*, University of Cambridge, Department of Engineering, 1989.

4. Gavin A. Rummery and Mahesan Niranjan, "On-Line Q-Learning Using Connectionist Systems", University of Cambridge, 1997, vol. 37.

5. Rémi Coulom, "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search", *International Conference on Computers and Games*, 2006.
"""

# ╔═╡ 9c139d5b-b8c9-4e00-ba5b-ee66fad73658
TableOfContents(title="Markov Decision Processes", depth=4)

# ╔═╡ aeec3109-1ebc-4edf-9ac5-eda3a30467cc
md"""
---
"""

# ╔═╡ 99f2318b-b1d2-4afa-960b-eed2971abede
html"""
<script>
var section = 0;
var subsection = 0;
var headers = document.querySelectorAll('h2, h3');
for (var i=0; i < headers.length; i++) {
    var header = headers[i];
    var text = header.innerText;
    var original = header.getAttribute("text-original");
    if (original === null) {
        // Save original header text
        header.setAttribute("text-original", text);
    } else {
        // Replace with original text before adding section number
        text = header.getAttribute("text-original");
    }
    var numbering = "";
    switch (header.tagName) {
        case 'H2':
            section += 1;
            numbering = section + ".";
            subsection = 0;
            break;
        case 'H3':
            subsection += 1;
            numbering = section + "." + subsection;
            break;
    }
    header.innerText = numbering + " " + text;
};
</script>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
D3Trees = "e3df1716-f71e-5df9-9e2d-98e193103c45"
DiscreteValueIteration = "4b033969-44f6-5439-a48b-c11fa3648068"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
MCTS = "e12ccd36-dcad-5f33-8774-9175229e7b33"
POMDPModelTools = "08074719-1b2a-587c-a292-00f91cc44415"
POMDPPolicies = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
POMDPSimulators = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Reel = "71555da5-176e-5e73-a222-aebc6c6e4f2f"
RollingFunctions = "b0e4dd01-7b14-53d8-9b45-175a3e362653"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TabularTDLearning = "df77095f-98f0-51e5-a706-d0ff5a5ad33f"

[compat]
ColorSchemes = "~3.23.0"
Colors = "~0.12.10"
D3Trees = "~0.3.3"
DiscreteValueIteration = "~0.4.6"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
MCTS = "~0.5.3"
POMDPModelTools = "~0.3.13"
POMDPPolicies = "~0.4.3"
POMDPSimulators = "~0.3.14"
POMDPs = "~0.9.5"
Parameters = "~0.12.3"
Plots = "~1.39.0"
PlutoUI = "~0.7.52"
QuickPOMDPs = "~0.2.14"
Reel = "~1.3.2"
RollingFunctions = "~0.8.0"
TabularTDLearning = "~0.4.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "0e5a7420558159018d075a02f39374ab5e4c0979"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.AccurateArithmetic]]
deps = ["LinearAlgebra", "Random", "VectorizationBase"]
git-tree-sha1 = "07af26e8d08c211ef85918f3e25d4c0990d20d70"
uuid = "22286c92-06ac-501d-9306-4abd417d9753"
version = "0.3.8"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonRLInterface]]
deps = ["Tricks"]
git-tree-sha1 = "6c7d1ebb157fdf0f696698ef01946fe93c9efff4"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.3"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.D3Trees]]
deps = ["AbstractTrees", "HTTP", "JSON", "Random", "Sockets"]
git-tree-sha1 = "cace6d05f71aeefe7ffd6f955a0725271f2b6cd5"
uuid = "e3df1716-f71e-5df9-9e2d-98e193103c45"
version = "0.3.3"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiscreteValueIteration]]
deps = ["POMDPLinter", "POMDPTools", "POMDPs", "Printf", "SparseArrays"]
git-tree-sha1 = "62d78a713948c4a95df289ca0eb8639697e1d2eb"
uuid = "4b033969-44f6-5439-a48b-c11fa3648068"
version = "0.4.6"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "aa9925a229d45fe3018715238956766fa21804d1"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.2.6"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "50351f83f95282cf903e968d7c6e8d44a5f83d0b"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d73afa4a2bb9de56077242d98cf763074ab9a970"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.9"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1596bab77f4f073a14c62424283e7ebff3072eca"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+1"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "bb198ff907228523f3dee1070ceee63b9359b6ab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "424a5a6ce7c5d97cca7bcc4eac551b97294c54af"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.9"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.KahanSummation]]
git-tree-sha1 = "6292e7878fe190651e74148edb11356dbbc2e194"
uuid = "8e2b3108-d4c1-50be-a7a2-16352aec75c3"
version = "0.3.1"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "0d097476b6c381ab7906460ef1ef1638fbce1d91"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.2"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "c88a4afe1703d731b1c4fdf4e3c7e77e3b176ea2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.165"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MCTS]]
deps = ["Colors", "D3Trees", "POMDPLinter", "POMDPTools", "POMDPs", "Printf", "ProgressMeter", "Random"]
git-tree-sha1 = "70df88ffe6f23e630a84543432052620be12b7d0"
uuid = "e12ccd36-dcad-5f33-8774-9175229e7b33"
version = "0.5.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MarchingCubes]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "c8e29e2bacb98c9b6f10445227a8b0402f2f173a"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[deps.POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Random", "Reexport", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "36d32d62e036ae8ebb9b8efe9e8658f902815700"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.13"

[[deps.POMDPPolicies]]
deps = ["Distributions", "LinearAlgebra", "POMDPTools", "POMDPs", "Parameters", "Random", "Reexport", "SparseArrays", "StatsBase"]
git-tree-sha1 = "bd72fbfea89a64946963518aa53097e2a1233c59"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.3"

[[deps.POMDPSimulators]]
deps = ["DataFrames", "Distributed", "NamedTupleTools", "POMDPLinter", "POMDPTools", "POMDPs", "ProgressMeter", "Random", "Reexport"]
git-tree-sha1 = "3735b7a48bd892f153ab7327cb71e447e8f18e14"
uuid = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
version = "0.3.14"

[[deps.POMDPTools]]
deps = ["CommonRLInterface", "DataFrames", "Distributed", "Distributions", "LinearAlgebra", "NamedTupleTools", "POMDPLinter", "POMDPs", "Parameters", "ProgressMeter", "Random", "Reexport", "SparseArrays", "Statistics", "StatsBase", "Tricks", "UnicodePlots"]
git-tree-sha1 = "961b4857eb1869517478416037bf33885ed5f6a0"
uuid = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
version = "0.1.5"

[[deps.POMDPs]]
deps = ["Distributions", "Graphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "9ab2df9294d0b23def1e5274a0ebf691adc8f782"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.5"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "d4c9ebdc6528a4aaf7cfcf43b482e927267b400d"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.6"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "ee094908d720185ddbdc58dbe0c1cbe35453ec7a"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "364898e8f13f7eaaceec55fd3d08680498c0aa6e"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.4.2+3"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "eeab25344bf9901146c0200a7ca64ea479f8bf5c"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.0"

[[deps.QuickPOMDPs]]
deps = ["NamedTupleTools", "POMDPTools", "POMDPs", "Random", "Tricks", "UUIDs"]
git-tree-sha1 = "5a198408a330c4bff6e085831e542f75f4e5e38f"
uuid = "8af83fb2-a731-493c-9049-9e19dbce6165"
version = "0.2.14"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reel]]
deps = ["FFMPEG"]
git-tree-sha1 = "0f600c38899603d9667111176eb6b5b33c80781e"
uuid = "71555da5-176e-5e73-a222-aebc6c6e4f2f"
version = "1.3.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RollingFunctions]]
deps = ["AccurateArithmetic", "FastBroadcast", "KahanSummation", "LinearAlgebra", "LoopVectorization", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "4a54152985fea23b0b0e99a77566a87137221a0a"
uuid = "b0e4dd01-7b14-53d8-9b45-175a3e362653"
version = "0.8.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "51621cca8651d9e334a659443a74ce50a3b6dfab"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.3"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "f02eb61eb5c97b48c153861c72fbbfdddc607e06"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.4.17"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a1f34829d5ac0ef499f6d84428bd6b4c71f02ead"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.0"

[[deps.TabularTDLearning]]
deps = ["DataStructures", "POMDPLinter", "POMDPTools", "POMDPs", "Random"]
git-tree-sha1 = "76b0324059490155499671bed5a20c5a76c07577"
uuid = "df77095f-98f0-51e5-a706-d0ff5a5ad33f"
version = "0.4.4"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeTypeAbstraction", "LazyModules", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "ae67ab0505b9453655f7d5ea65183a1cd1b3cfa0"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.12.4"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "04a51d15436a572301b5abbb9d099713327e9fc4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.4+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─7ce1bec4-f238-407e-aefb-c633ee2fadd5
# ╟─db0265cd-ebe0-4bf2-9e70-c0f978b91ff6
# ╟─faf0ca6e-693d-4217-9e3c-e9e60339d416
# ╟─a84c5d59-ebc6-465e-8d92-87618b5712f0
# ╟─8d30248e-b3c3-4f37-8296-392898790283
# ╟─5e9b28a4-50b9-4c44-8d3c-48eb72bda3a5
# ╠═8477c27b-ea40-4c59-b1ca-da8b641eb884
# ╠═ae71c096-311f-4da5-8a6c-3829242af1f9
# ╠═d91bcc7c-0941-4d6c-b695-026c33f67329
# ╠═232d7d72-94f8-4dfe-abdf-2b6e712847f7
# ╠═e1850dbb-30ea-4e1e-94f7-10582f89fb5d
# ╠═43a7c406-3d59-4b80-8d34-7b6119e9c936
# ╠═25d170e9-1c26-4058-96fc-04ab47964b51
# ╟─52b96024-9f52-4f07-926b-2297ed7dd166
# ╟─d9755f26-3f30-48ba-91d7-266c0204237d
# ╟─e2a84ebf-a259-43c1-b512-f6c6b6e02d14
# ╠═c092511d-c2e7-4b8c-8104-b4b10893cb02
# ╠═13dbf845-14a7-4c98-a1db-b3a83c9ce37c
# ╠═8d683391-75eb-4f5d-8849-7dd77720b5bf
# ╟─31ae33aa-5f25-4cd8-8e63-8e77c2233208
# ╠═b83aceeb-4360-43ab-9396-ac57a9416791
# ╟─07846f69-2f7a-4e12-9f4b-6fed8659e9ed
# ╠═4a14aee4-12f1-4d55-9532-9b88e4c465f8
# ╟─99acb099-742c-4d13-abd8-c588217e4466
# ╟─581376af-21eb-4cc8-91af-7b671ebf4e71
# ╠═c1d07fca-1fbd-4450-96b1-c829d7ad8306
# ╟─dcfc1975-04e8-4d8e-ab46-d1e0846c071e
# ╠═bcc5e8a3-1e3a-40cf-a306-13599a4952ac
# ╟─d66edb3a-7ccc-4e75-8d42-8d5b1ff5afbb
# ╠═bc541507-61db-4084-9712-1c57d139e17f
# ╟─b2856919-5529-431b-8025-0b7f3f3081b0
# ╠═1303be2a-d18c-44b0-afb9-06a6b4ce5c08
# ╟─268e2bb2-e6e2-4198-ad83-a93fcfa65b80
# ╠═27e554ff-9861-4a41-ad65-9d5ae7727e45
# ╟─148d8e67-33a4-4065-911e-9ee0c33d8822
# ╠═49901c66-db64-48a2-b122-84d5f6b769db
# ╟─51796bfc-ee3c-4cab-9d58-359608fd4106
# ╠═f7814a66-23c8-4782-ba06-755397af87db
# ╠═e67994b8-d519-4c24-9d61-5cbef1629baa
# ╟─e5286fa6-1a48-4020-ab03-c24a175c8c04
# ╠═87a6c45e-6f3e-428e-8301-3b0c4166a84b
# ╠═4ef71be8-4768-4f58-bf31-78a895452617
# ╟─fd5b8960-933a-4ca0-9a7e-5003821ccfe3
# ╠═6970821b-2b87-4e66-b737-512e83627998
# ╟─7c2c2733-eb28-4d85-9074-99e64074e414
# ╠═49b140ad-641f-436c-9492-cf3efbadd8d2
# ╠═5af462ea-781b-4509-98de-c68c29ec81fd
# ╟─a3f76a77-bb6d-4a6b-8e5a-170dcc867c07
# ╟─9f5dc78e-8183-4a03-9282-1aebf1af218c
# ╠═142f0646-541e-453b-a3b1-4b8fadf709cc
# ╟─c67f7fc6-7af8-4e4f-a341-133c70f879bc
# ╠═48c966cb-6b79-42a6-8ff0-2fe3261f3981
# ╟─12501ad4-b42d-4fc4-b54b-30f4b929c0ab
# ╠═ba95fdd5-d643-4de2-8235-90e2b5651410
# ╟─32668e09-b12b-43c8-862f-b6f1a77557ec
# ╟─90b507bd-8cab-4c30-816e-a4b264e903a6
# ╟─7ee1a2f0-0210-4e89-98e5-73f18fb178b1
# ╟─73182581-fdf4-4252-b64e-34f39e1f96da
# ╟─c5fbf696-3e5c-4b59-be4d-9a43f30d6211
# ╟─786b27eb-129f-4538-beca-7e8b69fd40e4
# ╠═9cb6e19b-25f4-44b5-8155-d55ad3ba617c
# ╟─da9926ae-4e49-4ff3-abc2-d8249bddb0f2
# ╠═9eba07a6-2753-42c5-8581-3fb7489c066a
# ╟─b942fd56-13c3-4729-a701-63f103b13638
# ╟─4de6845e-a555-4147-86e8-d623e399c22a
# ╠═3b883115-7d45-4439-8d04-48df4684452f
# ╟─887f90ce-98eb-4262-894b-e14a0a53fa50
# ╠═8e3c5337-951e-495c-a0e7-b050660d0192
# ╟─a4e4d65f-a734-404d-8478-029b0017651c
# ╠═73cde70f-17f9-4ccd-ae4e-0cf050c2915e
# ╠═6b77edd5-79d1-4b0a-b6e2-f8b83af52db7
# ╠═e06f597d-dc01-4dea-b63a-5f61d49170e0
# ╟─cd4d33c8-bc24-4327-a65e-ed2d46af766b
# ╠═c85a316a-acd3-4aa3-8b65-28a332950240
# ╟─d3b0aeb2-11c9-4d4e-aa53-bb831ccd74a2
# ╠═4d698cad-a570-4608-b6dd-20de5d7dbe33
# ╠═5024e6c5-39e2-4bd6-acca-05267cf8639e
# ╠═1540a649-b238-498e-a8fb-5a29461194b5
# ╠═6d024b5b-faa3-4075-babd-c6b260cef55e
# ╠═4bb93999-e6b5-4590-8605-9bfe83778890
# ╠═a8817d6e-2302-4f39-8b93-66d550ca09ef
# ╠═3590b4b0-5a05-4052-aa58-f48d3912ce77
# ╟─dd5e9a13-4297-4717-a150-e1908faea2ca
# ╠═edc2ec3c-95d9-4079-9efe-a39ea2053e15
# ╟─b4dd0437-b945-4b4e-a504-1ac0fca54a75
# ╠═80866699-58e9-4c32-a440-c5433c56a0ad
# ╠═38af3571-9b0a-4b19-b33a-573101b597a0
# ╠═49cd3c8b-d633-40e7-99f4-57a948b53c92
# ╠═dfa19121-b929-467a-a4ab-3f6563f200cf
# ╟─ea27b654-d225-4708-bf3f-fab306f51c72
# ╠═b0a444df-0b41-430b-924f-83075944368a
# ╟─a2b7e745-8b15-42c6-89ca-e97aef1c9a0f
# ╟─a94529cf-1ab3-4b28-887a-04be9d103869
# ╟─2d58ccfd-9b20-459a-a636-a47496df1b18
# ╠═50f78e55-0433-438e-aaee-46c34eba8ba5
# ╟─f0725051-8770-47d5-b8a7-65b1ab0955dd
# ╠═10994f20-831d-4c63-ba4d-fe7e43347d5d
# ╠═9d1994d2-7ea9-4828-98fc-27bf5d17bab9
# ╟─acbaca80-cafb-49c3-adea-4fd507c2c142
# ╠═61e6ad96-ff2a-4dd9-9698-48c33bd43f26
# ╟─2db772d9-16a1-4bdb-9205-611a1921831f
# ╠═ee7c1fe5-2991-4b2a-981b-1c72106d5855
# ╠═ee5ae9e6-b837-4521-bac7-cf069765d278
# ╠═9cf0f694-d4a3-48fd-9cb0-1c8ac0361d5c
# ╟─bf23cf92-103f-4181-b9e6-97efe0249d0a
# ╠═1c1c765e-3a36-42f2-b1cb-8683b265ecad
# ╠═3bcf0923-8ac2-4f82-97ca-d0996658a046
# ╟─e8620cb9-21de-4e5d-805a-0571eeceef7d
# ╠═f73f735c-6e8a-4ad4-b404-9772ce557eb1
# ╟─c0929ea6-4b20-4e34-bec3-f0fc5935e406
# ╠═133eaaec-7113-4ec6-bac0-555f6efa1cb3
# ╠═fb0a10c6-ca0b-480c-93d6-09aac5cc96ed
# ╠═5bfbceb4-7006-47c0-a965-13500caef00d
# ╟─799026ed-92f0-439a-a7e3-bd362eb18b99
# ╟─3484668f-9cdb-4ac9-b683-8054f0ea9d7e
# ╠═79ad2ccc-615e-4b02-9f9d-cf29aaabe7fc
# ╟─105a8fb9-008c-4ae1-83e8-8894209ada0e
# ╠═548124bb-0229-40f3-ba57-e436f37612ec
# ╠═60644fcd-dec8-4e0e-bbee-2e29443e61c8
# ╟─7af62c0f-2154-4e0a-86bb-17dd3a84f362
# ╠═d69bb3f3-f805-40f8-8f19-6210246ff86c
# ╟─6467f4c6-a59f-4601-8b8e-5627e3386d0e
# ╠═e5143473-6471-43e4-ac20-2907968f35d3
# ╟─6b9b6f24-f4d4-435a-856a-469e5a20f602
# ╠═756cf1c4-ddfb-4173-8287-5eac419093ff
# ╟─6befa3ba-35e8-4b03-a28b-b30ceb9a66d3
# ╠═54e28d34-af2f-49df-95f4-d95ab9f6b3ea
# ╟─d156dbbd-a63d-48a8-8c15-664df10416f5
# ╠═73126c4f-108e-4e09-8e37-82b4dbb4ffb5
# ╟─d4cee740-ab00-4078-b2b8-671cd155620a
# ╠═76d59916-c550-4d6e-b5c9-8989d67ad871
# ╟─dc03ec16-f319-4a98-a2ea-66ea5489151c
# ╠═2af5a9a9-a612-44a6-8be0-1d6cc43fc200
# ╟─6af23b44-528a-4027-9ba9-e1e91781d83b
# ╠═6e6d983c-7cb3-4f85-ac0c-f1daf8ee3fee
# ╟─d613b978-98fa-44aa-ad1d-c51e50e2d12a
# ╟─299dcfef-91d1-48c1-a163-830f340e31df
# ╟─9f651d75-42c6-4a3e-ab88-6b6ed68a36ea
# ╟─15bce1e5-9078-438a-9659-d72cf3440b4e
# ╟─273de0d6-aa6b-49c2-94e7-047879402ef9
# ╟─144beb1b-3aec-4daa-a4f4-eb8274b1a009
# ╟─fdf3325d-a949-4f39-9067-77bb13ab8ba7
# ╟─351a4e76-b13b-4719-9c86-097e0b10e930
# ╟─b26cc58b-c5ce-4660-889c-e6eb09577dd2
# ╠═50ce080c-235f-425a-925d-65f1c29e7d60
# ╟─06bf7a9e-2fc9-4c58-a1f2-b3bba9fce96f
# ╠═bc474e19-7d3e-481c-b36c-e231cd72db94
# ╠═caad8138-251b-4644-9217-3e3bba49e357
# ╠═02f590b6-b034-4ac4-afd4-76bde37e68b6
# ╟─908bd8f5-a523-43d9-ae83-7934ca93a226
# ╟─58c8c26f-21c9-4ef5-a0e0-712c9bd51150
# ╠═50121862-2d65-4f52-aa2f-69d23f8968e8
# ╠═0b6b147f-289c-466b-b859-e0b75f2d7823
# ╠═4f13867a-a91e-4251-8d43-746baa6d12a6
# ╟─a0178751-85c0-4424-bfd0-4df86434855b
# ╠═0fdb2e3a-838c-435f-a152-f27ba2413a3b
# ╟─680c3c05-2011-411e-87f1-7b27544cf8ea
# ╠═87c715e7-29aa-4111-a0d1-1c5049341142
# ╟─52ef9f86-2172-4600-9ddd-83f24fdc3944
# ╠═8be067e6-e12d-4262-b3d7-368e17db8e93
# ╠═7124f670-d5ba-4e9b-86cf-48f2be0bc2a2
# ╠═92bb38e3-00e2-4d5b-a511-4f0580777aa5
# ╟─a4e1b417-60c9-475e-aa16-b2112f7b47cf
# ╠═720fe4d3-5801-49b4-a7b9-4923be051220
# ╠═87710f14-a7ec-4983-b105-8090c4dd1463
# ╠═92cdb96c-e636-4308-bf49-6d34246d3256
# ╟─228fb0a2-fa38-401d-9793-3e487f79d34d
# ╠═012cb4ad-7e30-477b-9d30-0bc020f12606
# ╠═e2d85e1d-3368-43b1-93db-d24023805b9f
# ╟─9cc9137f-4585-43c9-b60d-2cd80d7766ff
# ╠═9c139d5b-b8c9-4e00-ba5b-ee66fad73658
# ╟─aeec3109-1ebc-4edf-9ac5-eda3a30467cc
# ╟─99f2318b-b1d2-4afa-960b-eed2971abede
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
