using MDPs
using Flux
using Flux.Zygote
using Random
using UnPack
using DataStructures
using StatsBase

export ContextualDQNPolicy

mutable struct ContextualDQNPolicy{T <: AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    π::DQNPolicy{T}
    crnn::GRUContextRNN
    context::Matrix{Float32}
    isnewtraj::Float32
    prev_a::Vector{Float32}
    prev_r::Float32
    function ContextualDQNPolicy(π::DQNPolicy{T}, crnn::GRUContextRNN) where {T}
        new{T}(π, crnn, zeros(Float32, size(get_rnn_state(crnn), 1), 1), 1f0, zeros(Float32, π.n), 0f0)
    end
end


function (p::ContextualDQNPolicy{T})(rng::AbstractRNG, o::Vector{T})::Int where {T}
    set_rnn_state!(p.crnn, p.context)
    # assuming the policy will not be called on a terminal state
    evidence = reshape(vcat(p.isnewtraj, p.prev_a, p.prev_r, tof32(o), 0f0), :, 1)
    c = p.crnn(evidence)
    o = reshape(o, :, 1) |> tof32
    s = vcat(c, o)
    a = p.π(rng, s)
    return a[1]
end

function (p::ContextualDQNPolicy{T})(o::Vector{T}, a::Int)::Float64 where {T}  # returns probability
    set_rnn_state!(p.crnn, p.context)
    evidence = reshape(vcat(p.isnewtraj, p.prev_a, p.prev_r, tof32(o), 0f0), :, 1)
    c = p.crnn(evidence)
    o = reshape(o, :, 1) |> tof32
    s = vcat(c, o)
    return p.π(s, :)[a, 1]
end


function MDPs.preepisode(p::ContextualDQNPolicy{T}; kwargs...) where {T}
    @debug "Resetting policy"
    p.isnewtraj = 1f0
    fill!(p.prev_a, 0f0)
    p.prev_r = 0
    fill!(p.context, 0f0)
    nothing
end

function MDPs.poststep(p::ContextualDQNPolicy{T}; env, kwargs...) where {T}
    p.isnewtraj = 0f0
    fill!(p.prev_a, 0f0)
    p.prev_a[action(env)] = 1f0
    p.prev_r = reward(env)
    copy!(p.context, get_rnn_state(p.crnn))
    @debug "Storing action, reward, grustate in policy struct"
    nothing
end

