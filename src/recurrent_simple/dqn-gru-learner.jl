import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment

export RecurrentDQNLearner

mutable struct RecurrentDQNLearner{T} <: AbstractHook
    π::ContextualDQNPolicy{T}
    γ::Float32
    ρ::Float32
    min_explore_steps::Int
    batch_size::Int
    horizon::Int
    tbptt_horizon::Int
    device

    buff::AbstractArray{Float32, 2} # sequence of evidence
    buff_head::Int
    traj_start_points::Set{Int}
    minibatch                               # preallocated memory for sampling a minibatch. Tuple 𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′
    𝐜::AbstractArray{Float32, 3}           # preallocated memory for recording context during a rollout

    policy::DQNPolicy{T}                    # train this policy and periodically copy weights to the original policy
    policy_crnn::GRUContextRNN              # train this context rnn and periodically copy weights to the original context rnn
    qmodel′                                 # target qmodel
    optim::Adam

    stats::Dict{Symbol, Float32}

    function RecurrentDQNLearner(π::ContextualDQNPolicy{T}, γ::Real, horizon::Int, η, aspace::MDPs.IntegerSpace, sspace; polyak=0.995, batch_size=32, min_explore_steps=horizon*batch_size, tbptt_horizon=horizon, buffer_size=10000000, buff_mem_MB_cap=Inf, device=Flux.cpu) where {T <: AbstractFloat}
        each_entry_size = 1 + length(aspace) + 1 + size(sspace, 1) + 1
        buffer_size = min(buffer_size, buff_mem_MB_cap * 2^20 / (4 * each_entry_size)) |> floor |> Int
        buff = zeros(Float32, each_entry_size, buffer_size)
        𝐞 = zeros(Float32, each_entry_size, horizon + 1, batch_size) |> device
        𝐨 = zeros(Float32, size(sspace, 1), horizon, batch_size) |> device
        𝐚 = zeros(Float32, size(aspace, 1), horizon, batch_size) |> device
        𝐫 = zeros(Float32, horizon, batch_size) |> device
        𝐨′ = zeros(Float32, size(sspace, 1), horizon, batch_size) |> device
        𝐝′ = zeros(Float32, horizon, batch_size) |> device
        𝐧′ = zeros(Float32, horizon, batch_size) |> device
        minibatch = (𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′)
        𝐜 = zeros(Float32, size(get_rnn_state(π.crnn), 1), horizon + 1, batch_size) |> device
        new{T}(π, γ, polyak, min_explore_steps, batch_size, horizon, tbptt_horizon, device, buff, 1, Set{Int}(), minibatch, 𝐜, device(deepcopy(π.π)), device(deepcopy(π.crnn)), device(deepcopy(π.π.qmodel)), Adam(η), Dict{Symbol, Float32}())
    end
end

function increment_buff_head!(dqn::RecurrentDQNLearner)
    cap = size(dqn.buff, 2)
    dqn.buff_head = ((dqn.buff_head + 1) - 1) % cap + 1
    nothing
end

function push_to_buff!(dqn::RecurrentDQNLearner, is_new_traj, prev_action::Int, prev_reward, cur_state, cur_state_terminal, aspace::MDPs.IntegerSpace)
    buff = dqn.buff
    m = dqn.buff_head
    n_actions = length(aspace)

    if Bool(buff[1, m])
        delete!(sac.traj_start_points, m)
    end


    buff[1, m] = is_new_traj
    buff[1+1:1+n_actions, m] .= 0f0
    buff[1+prev_action, m] = 1f0
    buff[1+n_actions+1, m] = prev_reward
    buff[1+n_actions+1+1:1+n_actions+1+length(cur_state), m] .= cur_state
    buff[end, m] = cur_state_terminal

    if Bool(is_new_traj)
        push!(dqn.traj_start_points, m)
    end

    increment_buff_head!(dqn)
    nothing
end

function sample_from_buff!(dqn::RecurrentDQNLearner, env)
    (𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′), seq_len, batch_size = dqn.minibatch, dqn.horizon + 1, dqn.batch_size
    cap = size(dqn.buff, 2)
    @assert seq_len < cap
    n_actions = length(action_space(env))

    function isvalidstartpoint(p_begin)
        """Ensures that the buffer head doesn't lie in between any sampled trajectory:"""
        p_end = ((p_begin + seq_len - 1) - 1) % cap + 1
        if p_end > p_begin
            return !(p_begin <= dqn.buff_head <= p_end)
        else
            return p_end < dqn.buff_head < p_begin
        end
    end
    
    for n in 1:batch_size
        start_index = rand(dqn.traj_start_points)
        while !isvalidstartpoint(start_index)
            start_index = rand(dqn.traj_start_points)
        end
        indices = ((start_index:(start_index .+ seq_len - 1)) .- 1) .% cap .+ 1
        𝐞[:, :, n] .= dqn.device(dqn.buff[:, indices])
    end

    # Note: "actions" are onehot
    prev_actions = @view 𝐞[1+1:1+n_actions, :, :]
    prev_rewards = @view 𝐞[1+n_actions+1, :, :]
    cur_obs = @view 𝐞[1+n_actions+1+1:1+n_actions+1+length(state(env)), :, :]
    
    obs = @view cur_obs[:, 1:end-1, :]
    actions = @view prev_actions[:, 2:end, :]
    rewards = @view prev_rewards[2:end, :]
    next_obs = @view cur_obs[:, 2:end, :]
    next_isterminals = @view 𝐞[end, 2:end, :]
    next_is_newtrajs = @view 𝐞[1, 2:end, :]

    copy!(𝐨, obs); copy!(𝐚, actions); copy!(𝐫, rewards); copy!(𝐨′, next_obs); copy!(𝐝′, next_isterminals); copy!(𝐧′, next_is_newtrajs)

    return 𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′
end


function preepisode(dqn::RecurrentDQNLearner; env, kwargs...)
    push_to_buff!(dqn, true, 1, 0f0, state(env), in_absorbing_state(env), action_space(env))
end

function poststep(dqn::RecurrentDQNLearner{T}; env::AbstractMDP{Vector{T}, Int}, steps::Int, returns, rng::AbstractRNG, kwargs...) where {T}
    @unpack policy, policy_crnn, γ, ρ, batch_size, horizon, tbptt_horizon, device, 𝐜, qmodel′ = dqn

    push_to_buff!(dqn, false, action(env), reward(env), state(env), in_absorbing_state(env), action_space(env))

    if steps >= dqn.min_explore_steps && (steps % 50 == 0)
        @debug "sampling trajectories"
        𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′  = sample_from_buff!(dqn, env)
        # note: 𝐚 is onehot!
        function dqn_update()
            θ = Flux.params(policy, policy_crnn)
            Flux.reset!(policy_crnn)
            fill!(𝐜, 0f0)
            for t in 1:(horizon+1)
                𝐜[:, t, :] .= @views policy_crnn(𝐞[:, t, :])
            end
            𝐜′ = @view 𝐜[:, 2:end, :]
            𝐬′ = reshape(vcat(𝐜′, 𝐨′), :, horizon * batch_size)
            𝛑′ = policy(𝐬′, :)
            𝐪̂′ = qmodel′(𝐬′)
            𝐯̂′ = sum(𝛑′ .* 𝐪̂′, dims=1)[:, ]
            𝐨 = reshape(𝐨, :, horizon * batch_size)
            𝐚 = argmax(reshape(𝐚, :, horizon * batch_size), dims=1) # CartesianIndices
            𝐫 = reshape(𝐫, horizon * batch_size)
            𝐝′ = reshape(𝐝′, horizon * batch_size)
            𝐧′ = reshape(𝐧′, horizon * batch_size)
            v̄ = 0f0
            Flux.reset!(policy_crnn)
            ℓ, ∇θℓ = Flux.Zygote.withgradient(θ) do
                _𝐜 = reduce(hcat, map(1:horizon) do t
                    @views reshape(policy_crnn(𝐞[:, t, :]), :, 1, batch_size)
                end)
                _𝐜 = reshape(_𝐜, :, horizon * batch_size)
                𝐬 = vcat(_𝐜, 𝐨)
                𝐪̂ = policy.qmodel(𝐬)
                v̄ += Zygote.@ignore mean(sum(policy(𝐬, :) .* 𝐪̂, dims=1))
                𝛅 = (𝐫 + γ * (1f0 .- 𝐝′) .* 𝐯̂′ - 𝐪̂[𝐚][1, :]) .* (1f0 .- 𝐧′)
                return mean(𝛅.^2)
            end
            Flux.update!(dqn.optim, θ, ∇θℓ)
            return ℓ, v̄
        end

        function target_network_update()
            θ = Flux.params(dqn.policy.qmodel)
            θ′ = Flux.params(dqn.qmodel′)
            for (param, param′) in zip(θ, θ′)
                copy!(param′, ρ * param′ + (1 - ρ) * param)
            end
        end

        function copy_back_policy_params()
            θ = Flux.params(policy, policy_crnn)
            θ′ = Flux.params(dqn.π.π, dqn.π.crnn)
            for (param, param′) in zip(θ, θ′)
                copy!(param′, param)
            end
        end

        @debug "dqn update"
        ℓ, v̄ = dqn_update()

        @debug "target network update"
        target_network_update()

        @debug "policy parameters back to the actor"
        copy_back_policy_params()

        episodes = length(returns)
        dqn.stats[:ℓ] = ℓ
        dqn.stats[:v̄] = v̄
        @debug "learning stats" steps episodes dqn.stats...
    end
    nothing
end
