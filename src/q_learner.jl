using MDPs
import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment, VectorSpace
using Flux
using Random
using UnPack
using DataStructures
using StatsBase

export DQNLearner

mutable struct DQNLearner{T<:AbstractFloat} <: AbstractHook
    π::DQNPolicy{T}
    ρ::Float32
    min_explore_steps::Int
    train_interval::Int
    batch_size::Int

    s::Union{Vector{T}, Nothing}
    buff::CircularBuffer{Tuple{Vector{T}, Int, Float64, Vector{T}, Bool}}
    qmodel′
    optim::Adam

    function DQNLearner(π::DQNPolicy{T}, α; polyak=0.995, min_explore_steps=10000, train_interval=1, batch_size=32, buffer_size=1000000) where T <: AbstractFloat
        buff = CircularBuffer{Tuple{Vector{T}, Int, Float64, Vector{T}, Bool}}(buffer_size)
        new{T}(π, polyak, min_explore_steps, train_interval, batch_size, nothing, buff, deepcopy(π.qmodel), Adam(α))
    end
end

function prestep(dqn::DQNLearner; env::AbstractMDP, kwargs...)
    dqn.s = copy(state(env))
end

function poststep(dqn::DQNLearner{T}; env::AbstractMDP{Vector{T}, Int}, steps::Int, rng::AbstractRNG, kwargs...) where T <: AbstractFloat
    @unpack π, ρ, batch_size, s, qmodel′ = dqn

    a, r, s′, d, γ = action(env), reward(env), copy(state(env)), in_absorbing_state(env), discount_factor(env)
    push!(dqn.buff, (s, a, r, s′, d))

    if steps >= dqn.min_explore_steps && steps % dqn.train_interval == 0
        replay_batch = rand(rng, dqn.buff, batch_size)
        𝐬, 𝐚, 𝐫, 𝐬′, 𝐝 = map(i -> reduce((𝐱, y) -> cat(𝐱, y; dims=ndims(y) + 1), map(experience -> experience[i], replay_batch)), 1:5)
        𝐬, 𝐫, 𝐬′, 𝐝 = tof32.((𝐬, 𝐫, 𝐬′, 𝐝))
        𝐚_𝐬 = map(j -> CartesianIndex(𝐚[j], j), 1:batch_size)
        
        θ = Flux.params(π.qmodel)
        ℓ, ∇θℓ = Flux.Zygote.withgradient(θ) do
            𝐪̂ = π.qmodel(𝐬)
            𝐪 = Flux.Zygote.ignore() do
                𝛑′ = π(𝐬′, :)
                𝐪′ = qmodel′(𝐬′)
                𝐯′ = sum(𝛑′ .* 𝐪′, dims=1)[1, :]
                𝛅 = zeros(Float32, size(𝐪̂)) # TD error
                𝛅[𝐚_𝐬] = 𝐫 + (1 .- 𝐝) * Float32(γ) .* 𝐯′ - @view 𝐪̂[𝐚_𝐬]
                𝐪̂ + 𝛅
            end
            Flux.mse(𝐪̂, 𝐪)
        end

        Flux.update!(dqn.optim, θ, ∇θℓ)

        θ′ = Flux.params(qmodel′)
        Flux.loadparams!(qmodel′, ρ .* θ′ .+ (1 - ρ) .* θ)

        if steps % 1000 == 0
            v̄ = mean(sum(π(𝐬, :) .* π.qmodel(𝐬), dims=1))
            @info "learning stats" steps ℓ v̄
        end
    end
    nothing
end
