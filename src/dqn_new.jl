


Base.@kwdef mutable struct DQNLearner
    policy::DQNPolicy
    lr::Float32 = 0.0003
    device = cpu
    clipnorm::Float32 = Inf
    adam_weight_decay::Float32 = 0f0
    adam_epsilon::Float32 = 1f-7

    optim = make_adam_optim(lr, (0.9, 0.999), adam_epsilon, clipnorm, adam_weight_decay)
    target_policy::DQNPolicy = deepcopy(policy)
    policy_gpu::DQNPolicy = device(deepcopy(policy))
    stats = Dict{Symbol, Any}()
end