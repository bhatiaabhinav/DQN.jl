using MDPs
using Random
using Flux

export DQNPolicy


struct DQNPolicy{T<:AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    qmodel
    ϵ::Float64
    n::Int
end

Flux.@functor DQNPolicy (qmodel, )

Flux.gpu(p::DQNPolicy{T}) where T = DQNPolicy{T}(Flux.gpu(p.qmodel), p.ϵ, p.n) 
Flux.cpu(p::DQNPolicy{T}) where T = DQNPolicy{T}(Flux.cpu(p.qmodel), p.ϵ, p.n)

function (p::DQNPolicy{T})(rng::AbstractRNG, 𝐬::Vector{T})::Int where T<:AbstractFloat
    return rand(rng) < p.ϵ ? rand(rng, 1:p.n) : (𝐬 |> tof32 |> p.qmodel |> argmax)
end

function (p::DQNPolicy{T})(𝐬::Vector{T}, a::Int)::Float64 where T <: AbstractFloat
    a_greedy::Int =  𝐬 |> tof32 |> p.qmodel |> argmax
    return a == a_greedy ? (1 - p.ϵ + p.ϵ / p.n) : p.ϵ / p.n
end




function (p::DQNPolicy)(rng::AbstractRNG, 𝐬::AbstractMatrix{<:AbstractFloat})::AbstractVector{Int}
    batch_size = size(𝐬, 2)
    𝐳 = rand(rng, batch_size)
    𝐚_random = rand(rng, 1:p.n, batch_size)
    𝐚_greedy = map(ci -> ci[1], argmax(p.qmodel(tof32(𝐬)), dims=1))[1, :]
    return (𝐳 .< p.ϵ) .* 𝐚_random + (𝐳 .>= p.ϵ) .* 𝐚_greedy
end

function (p::DQNPolicy)(𝐬::AbstractMatrix{<:AbstractFloat}, ::Colon)::AbstractMatrix{Float32}
    𝐪 = p.qmodel(tof32(𝐬))
    # println(typeof(𝐪))
    𝛑 = convert(typeof(𝐪), zeros(Float32, size(𝐪)))
    𝛑[argmax(𝐪, dims=1)] .= 1f0
    return Float32(p.ϵ) / p.n .+ (1f0 - Float32(p.ϵ)) * 𝛑
end




function tof32(𝐱::AbstractArray{<:Real, N})::AbstractArray{Float32, N} where N
    convert(AbstractArray{Float32, N}, 𝐱)
end
