using Distributed
procs = addprocs(10)

@everywhere begin

    using Yao, BitBasis, CuYao, YaoExtensions, Yao.AD, Zygote
    using LinearAlgebra, Statistics, Random, StatsBase
    using NPZ, JLD2, FileIO, Printf, BenchmarkTools, MAT, PyPlot
    using Flux: batch, Flux
    using ArgParse
    using Distributions
    using Dates

    vars = matread("CODE/dataset/FashionMNIST.mat")
    x_train = vars["x_train"]
    y_train = vars["y_train"]
    x_test = vars["x_test"]
    y_test = vars["y_test"]
    nbit = 10;

    batch_size = 100
    lr = 0.003
    niters = 200;

    T = 2
    num_train = 1000
    num_test = 400
    x_train = real(x_train[:,1:num_train])*T
    y_train = y_train[1:num_train,:]
    x_test = real(x_test[:,1:num_test])*T
    y_test = y_test[1:num_test,:];

    pos_ = 5
    op0 = put(nbit, pos_=>0.5*(I2+Z))
    op1 = put(nbit, pos_=>0.5*(I2-Z));

    num_x = 270

    x_train_ = zeros(Float64,(num_x,num_train))
    x_train_[1:256,:] = x_train
    x_train = x_train_

    x_test_ = zeros(Float64,(num_x,num_test))
    x_test_[1:256,:] = x_test
    x_test = x_test_;

    # 按照实验要求，x_train[i]*2+pi
    x_train = x_train .+ pi
    x_test = x_test .+ pi;
    
    function acc_loss_rdm_cu(nbit,circuit,y_batch,batch_size)
        rdm = batch([density_matrix(focus!(zero_state(nbit) |> circuit[i], (pos_))).state for i in 1:batch_size])[:,:,1,:]
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
    
    function f(i_::Int64)
        t1 = time()

            # define a big block
        RX_layer(nbit::Int64) = chain(put(nbit, i => Rx(0)) for i in 1:nbit)
        RZ_layer(nbit::Int64) = chain(put(nbit, i => Rz(0)) for i in 1:nbit)

        E_layer1(nbit::Int64) = chain(chain(nbit,control(i,i+1=>X) for i in 1:2:5),chain(nbit,control(8,10=>X)))
        E_layer2(nbit::Int64) = chain(chain(nbit,control(i,i+1=>X) for i in 2:2:4),chain(nbit,control(6,8=>X)))
        E_layer3(nbit::Int64) = chain(nbit,control(i,i+1=>X) for i in 6:2:8)
        E_layer(nbit::Int64) = chain(nbit,E_layer1(nbit),E_layer2(nbit),E_layer3(nbit))

        big_block(nbit::Int64) = chain(RX_layer(nbit),RZ_layer(nbit),RX_layer(nbit),RX_layer(nbit),RZ_layer(nbit),RX_layer(nbit),E_layer(nbit))

        # locate the position of encoded x and trainable theta
        num_bigblock = 9
        num_params_bigblock = nparameters(big_block(nbit))
        num_x_bigblock = 30

        position_x = Int64[]
        position_theta = Int64[]
        p_x = Int64[i for i in 1:num_x_bigblock]
        p_theta = Int64[i for i in num_x_bigblock+1:num_params_bigblock]

        for ind in 1:num_bigblock
            x_ = (ind-1)*num_params_bigblock .+ p_x
            theta_ = (ind-1)*num_params_bigblock .+ p_theta
            for x in x_
                push!(position_x, x)
            end
            for t in theta_
                push!(position_theta, t)
            end
        end

        circuit = chain(big_block(nbit) for i in 1:num_bigblock)
        dispatch!(circuit, :random);

        # change x or trainable theta
        function dispatch_x(circuit, x)
            params_ = parameters(circuit)
            params_[position_x] = x
            dispatch!(circuit, params_)
        end
        function dispatch_theta(circuit, theta)
            params_ = parameters(circuit)
            params_[position_theta] = theta
            dispatch!(circuit, params_)
        end
        function dispatch_x_theta(circuit, x, theta)
            params_ = parameters(circuit)
            params_[position_theta] = theta
            params_[position_x] = x
            dispatch!(circuit, params_)
        end

        ini_params = parameters(circuit)
        train_cir = [chain(big_block(nbit) for i in 1:num_bigblock) for i in 1:num_train]
        test_cir = [chain(big_block(nbit) for i in 1:num_bigblock) for i in 1:num_test];
        for i in 1:num_train
            dispatch!(train_cir[i],ini_params)
            dispatch_x(train_cir[i], x_train[:,i])
        end
        for i in 1:num_test
            dispatch!(test_cir[i], ini_params)
            dispatch_x(test_cir[i], x_test[:,i])
        end

        optim = Flux.ADAM(lr)

        loss_train_history = Float64[]
        acc_train_history = Float64[]
        loss_test_history = Float64[]
        acc_test_history = Float64[];
        
        ini_theta = parameters(circuit)[position_theta];
        INI_setting = copy(ini_theta);

        ini_theta = copy(INI_setting);
        for i in train_cir
            dispatch_theta(i, ini_theta)
        end
        for i in test_cir
            dispatch_theta(i, ini_theta)
        end

        d = Normal(0,1)
        ε = 0.00;
        
        for k in 1:niters
            global lr
            global ini_param

            t1 = time()

            train_acc, train_loss = acc_loss_rdm_cu(nbit,train_cir,y_train,num_train)
            test_acc, test_loss = acc_loss_rdm_cu(nbit,test_cir,y_test,num_test)

            push!(loss_train_history, train_loss)
            push!(loss_test_history, test_loss)
            push!(acc_train_history, train_acc)
            push!(acc_test_history, test_acc)

            if train_acc >= 0.9999 && test_acc >= 0.9999
                break
            end

            batch_index = randperm(size(x_train)[2])[1:batch_size]
            batch_cir = train_cir[batch_index]
            y_batch = y_train[batch_index,:]

            # for data in each batch, each time we randomize both x and theta
            noise_batch_theta = ε*rand(d, (batch_size,size(parameters(circuit)[position_theta])[1]))
            for (i,j) in zip(batch_cir,1:batch_size)
                dispatch_theta(i,ini_theta+noise_batch_theta[j,:])
            end
            # x
            noise_batch_x = ε*rand(d, (batch_size,size(parameters(circuit)[position_x])[1]))
            for (i,j) in zip(batch_cir,1:batch_size)
                dispatch_x(i,parameters(i)[position_x]+noise_batch_x[j,:])
            end

            rdm = batch([density_matrix(focus!(zero_state(nbit) |> batch_cir[i], (pos_))).state for i in 1:batch_size])[:,:,1,:]
            q_ = zeros(batch_size,2);
            for i=1:batch_size
                q_[i,:] = diag(rdm[:,:,i]) |> real
            end

            Arr = Array{Float64}(zeros(batch_size,nparameters(batch_cir[1])))
        #     for i in 1:batch_size
        #         # clip the gradient, optional
        #         Arr[i,:] = g_clip(0.005, expect'(op0, zero_state(nbit)=>batch_cir[i])[2])
        #     end
            for i in 1:batch_size
                Arr[i,:] = expect'(op0, zero_state(nbit)=>batch_cir[i])[2]
            end

            C = [Arr, -Arr]

            grads = collect(mean([-sum([y_batch[i,j]*((1 ./ q_)[i,j])*batch(C)[i,:,j] for j in 1:2]) for i=1:batch_size]))
            noise_grad = 0.1*ε*rand(d, size(parameters(circuit)[position_theta]))
            updates = Flux.Optimise.update!(optim, ini_theta, grads[position_theta]+noise_grad);
            ini_theta = updates

            # recover the x
            for (i,j) in zip(batch_cir,1:batch_size)
                dispatch_x(i,parameters(i)[position_x]-noise_batch_x[j,:])
            end

            for i in train_cir
                dispatch_theta(i, updates)
            end
            for i in test_cir
                dispatch_theta(i, updates)
            end

            t2 = time()
        end

        t2 = time()
        T = (t2 - t1)/3600

        matwrite("CODE/circuit_encoding/benchmark/FashionMNIST_BE_20220208_"*string(i_)*".mat", Dict(
                "loss_train_history" => loss_train_history,
                "acc_train_history" => acc_train_history,
                "loss_test_history" => loss_test_history,
                "acc_test_history" => acc_test_history,
                "nbit" => nbit,
                "batch_size" => batch_size,
                "lr" => lr,
                "niters" => niters,
                "num_train" => num_train,
                "num_test" => num_test,
                "time_used_hour" => T
            )
        )
    end
    task(args) = f(args[1])
end
res = pmap(task, Array(1:10))





