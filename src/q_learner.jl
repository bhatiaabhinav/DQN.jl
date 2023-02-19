using MDPs
import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment, VectorSpace
using Flux
using Random
using UnPack
using DataStructures
using StatsBase
using Flux: Optimiser, ClipNorm, ClipValue

export DQNLearner

Base.@kwdef mutable struct DQNLearner{T<:AbstractFloat} <: AbstractHook
    π::DQNPolicy{T}
    η::Float32 = 0.0001
    γ::Float32 = 0.99
    ρ::Float32 = 0.999
    min_explore_steps::Int = 10000
    train_interval::Int = 1
    gradsteps::Int = 1
    batch_size::Int = 32
    buffer_size::Int = 1000000
    clipnorm = Inf
    clipval = Inf
    device = cpu

    s::Union{Vector{T}, Nothing} = nothing
    buff::CircularBuffer{Tuple{Vector{T}, Int, Float64, Vector{T}, Bool}} = CircularBuffer{Tuple{Vector{T}, Int, Float64, Vector{T}, Bool}}(buffer_size)
    π_gpu::DQNPolicy{T} = device(deepcopy(π))
    qmodel′ = device(deepcopy(π.qmodel))
    optim = clipnorm < Inf ? Optimiser(ClipNorm(clipnorm), Adam(η)) : (clipval < Inf ? Optimiser(ClipValue(clipval), Adam(η)) : Adam(η))
    stats::Dict{Symbol, Float32} = Dict{Symbol, Float32}()
end

function prestep(dqn::DQNLearner; env::AbstractMDP, kwargs...)
    dqn.s = copy(state(env))
end

function poststep(dqn::DQNLearner{T}; env::AbstractMDP{Vector{T}, Int}, steps::Int, rng::AbstractRNG, returns, kwargs...) where T <: AbstractFloat
    @unpack π_gpu, γ, ρ, batch_size, s, qmodel′, device = dqn

    a, r, s′, d = action(env), reward(env), copy(state(env)), in_absorbing_state(env)
    push!(dqn.buff, (s, a, r, s′, d))

    if steps >= dqn.min_explore_steps && steps % dqn.train_interval == 0
        for gradstep in 1:dqn.gradsteps
            replay_batch = rand(rng, dqn.buff, batch_size)
            𝐬, 𝐚, 𝐫, 𝐬′, 𝐝 = map(i -> reduce((𝐱, y) -> cat(𝐱, y; dims=ndims(y) + 1), map(experience -> experience[i], replay_batch)), 1:5)
            𝐬, 𝐫, 𝐬′, 𝐝 = (𝐬, 𝐫, 𝐬′, 𝐝) .|> tof32 .|> device
            𝐚_𝐬 = map(j -> CartesianIndex(𝐚[j], j), 1:batch_size) |> device
            
            θ = Flux.params(π_gpu.qmodel)
            ℓ, ∇θℓ = Flux.Zygote.withgradient(θ) do
                𝐪̂ = π_gpu.qmodel(𝐬)
                𝐪 = Flux.Zygote.ignore() do
                    𝛑′ = π_gpu(𝐬′, :)
                    𝐪′ = qmodel′(𝐬′)
                    𝐯′ = sum(𝛑′ .* 𝐪′, dims=1)[1, :]
                    𝛅 = zeros(Float32, size(𝐪̂)) |> device # TD error
                    # println(size(𝛅))
                    𝛅[𝐚_𝐬] = 𝐫 + (1 .- 𝐝) * γ .* 𝐯′ - @view 𝐪̂[𝐚_𝐬]
                    𝐪̂ + 𝛅
                end
                Flux.mse(𝐪̂, 𝐪)
            end

            Flux.update!(dqn.optim, θ, ∇θℓ)

            v̄ = mean(sum(π_gpu(𝐬, :) .* π_gpu.qmodel(𝐬), dims=1))
            dqn.stats[:v̄] = v̄
            dqn.stats[:ℓ] = ℓ
        end

        θ = Flux.params(π_gpu.qmodel)
        θ′ = Flux.params(qmodel′)
        for (param, param′) in zip(θ, θ′)
            copy!(param′, ρ * param′ + (1 - ρ) * param)
        end

        if device == gpu
            θ = Flux.params(π_gpu.qmodel)
            θcpu = Flux.params(dqn.π.qmodel)
            for (param, param_cpu) in zip(θ, θcpu)
                copy!(param_cpu, param)
            end
        end

        if steps % 1000 == 0
            episodes = length(returns)
            @debug "learning stats" steps episodes dqn.stats...
        end
    end
    nothing
end
