# for amplitude encoding
function acc_loss_rdm_cu(circuit::ChainBlock,reg::ArrayReg,y_batch::Matrix{Float64},batch_size::Int64)
    rdm = density_matrix(copy(reg) |> circuit, (pos_,)).state
    q_ = zeros(batch_size,2);
    for i=1:batch_size
        q_[i,:] = diag(rdm[:,:,i]) |> real
    end
    
    pred = [x[2] for x in argmax(q_,dims=2)[:]]
    y_max = [x[2] for x in argmax(y_batch,dims=2)[:]]
    acc = sum(pred .== y_max)/batch_size
    loss = crossentropy(y_batch,q_)/batch_size
    acc, loss
end

# for block encoding
function acc_loss_rdm_cu(nbit::Int64,circuit::Vector,y_batch::Matrix{Float64},batch_size::Int64)
    rdm = batch([density_matrix(zero_state(nbit) |> circuit[i], (pos_)).state for i in 1:batch_size])[:,:,1,:]
    q_ = zeros(batch_size,2);
    for i=1:batch_size
        q_[i,:] = diag(rdm[:,:,i]) |> real
    end
    
    pred = [x[2] for x in argmax(q_,dims=2)[:]]
    y_max = [x[2] for x in argmax(y_batch,dims=2)[:]]
    acc = sum(pred .== y_max)/batch_size
    loss = crossentropy(y_batch,q_)/batch_size
    acc, loss
end