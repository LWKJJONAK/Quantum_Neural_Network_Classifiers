# for amplitude encoding
function acc_loss_rdm_cu(circuit::ChainBlock ,reg::AbstractArrayReg, y_batch::Matrix{Float64},batch_size::Int64, pos_)
    res = copy(reg) |> circuit
    q_ = zeros(batch_size,2);
    for i=1:batch_size
        rdm = density_matrix(viewbatch(res, i), (pos_,))
        q_[i,:] = probs(rdm)
    end
    
    pred = [x[2] for x in argmax(q_,dims=2)[:]]
    y_max = [x[2] for x in argmax(y_batch,dims=2)[:]]
    acc = sum(pred .== y_max)/batch_size
    loss = crossentropy(y_batch,q_)/batch_size
    acc, loss
end

# for block encoding
function acc_loss_rdm_cu(nbit::Int64, circuit::Vector, y_batch::Matrix{Float64},batch_size::Int64, pos_)
    q_ = zeros(batch_size,2);
    for i=1:batch_size
        res = zero_state(nbit) |> circuit[i]
        rdm = density_matrix(res, (pos_,))
        q_[i,:] = rdm |> probs
    end
    
    pred = [x[2] for x in argmax(q_,dims=2)[:]]
    y_max = [x[2] for x in argmax(y_batch,dims=2)[:]]
    acc = sum(pred .== y_max)/batch_size
    loss = crossentropy(y_batch,q_)/batch_size
    acc, loss
end

export crossentropy
function crossentropy(p, q)
    return -sum(p .* log.(q))
end
