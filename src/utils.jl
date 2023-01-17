function splitequal(H, TH)
    splits = []
    t = 0
    while t < H
        push!(splits, (t+1):(t+TH))
        t += TH
    end
    return splits
end