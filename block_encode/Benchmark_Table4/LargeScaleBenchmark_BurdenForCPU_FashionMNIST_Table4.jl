using Distributed
procs = addprocs(45)

@everywhere begin
    using Yao, Zygote, YaoPlots, CuYao, YaoExtensions
    using LinearAlgebra, Statistics, Random, StatsBase, ArgParse, Distributions
    using Printf, BenchmarkTools, MAT
    using Flux: batch, Flux
end

@everywhere begin
    vars = matread("CODE/Github_ClassifierProject/dataset/FashionMNIST_1_2_wk.mat")
    
    num_train = 1000
    num_test = 200

    repeat_ = 100
    num_qubit = 10    # number of qubits
    batch_size = 50 # batch size
    lr = 0.005       # learning rate
    niters = 100;     # number of iterations
    optim = Flux.ADAM(lr); # Adam optimizer

    # index of qubit that will be measured
    pos_ = num_qubit;       
    op0 = put(num_qubit, pos_=>0.5*(I2+Z))
    op1 = put(num_qubit, pos_=>0.5*(I2-Z));
    
    include("../library/Layer.jl")
    include("../library/Loss_Accuracy.jl")

    function Ent_Layer_(nbit::Int64)
        Ent_CZ(nbit),Ent_CX(nbit),chain(nbit, Ent_CX(nbit), Ent_CX(nbit))
    end
    Parameterized_Layer(nbit::Int64) = Params_Layer(nbit)
end

@everywhere begin    
    function f(i_::Int64)
        Scal = [0.1,0.4,0.7,1.0,1.3,1.6,2.0,2.4,2.8,3.2,4.0,5.0,6.0,8.0,10.0]
        Scal_index = (i_-1)%15+1
        Ent_index = ((i_-Scal_index)/15+1) |> Int64

        depth = 9
        Ent_Layer(nbit::Int64) = Ent_Layer_(nbit)[Ent_index]
        Composite_Block(nbit::Int64) = chain(nbit, Parameterized_Layer(nbit::Int64), Ent_Layer(nbit::Int64))
        circuit = chain(Composite_Block(num_qubit) for _ in 1:depth)
        dispatch!(circuit, :random)
        ini_params = parameters(circuit)

        # set the scaling factor
        c = Scal[Scal_index]
        x_train = real(vars["x_train"][:,1:num_train])*c
        y_train = vars["y_train"][1:num_train,:]
        x_test = real(vars["x_test"][:,1:num_test])*c
        y_test = vars["y_test"][1:num_test,:];
        
        dim = 270
        x_train_ = zeros(Float64,(dim,num_train))
        x_train_[1:256,:] = x_train
        x_train = x_train_
        x_test_ = zeros(Float64,(dim,num_test))
        x_test_[1:256,:] = x_test
        x_test = x_test_
        
        train_cir = [chain(chain(num_qubit, Params_Layer(num_qubit), Ent_CX(num_qubit)) 
                for _ in 1:depth) for _ in 1:num_train]
        test_cir = [chain(chain(num_qubit, Params_Layer(num_qubit), Ent_CX(num_qubit)) 
                for _ in 1:depth) for _ in 1:num_test];
        
        for rep in 1:repeat_
            t1 = time()
            # assign random initial parameters to the circuit
            dispatch!(circuit, :random)
            ini_params = parameters(circuit)
            ini_theta = copy(ini_params);
            
            for i in 1:num_train
                dispatch!(train_cir[i], x_train[:,i]+ini_params)
            end
            for i in 1:num_test
                dispatch!(test_cir[i], x_test[:,i]+ini_params)
            end

            # record the training history
            loss_train_history = Float64[]
            acc_train_history = Float64[]
            loss_test_history = Float64[]
            acc_test_history = Float64[];
            
            for k in 1:niters
                # calculate the accuracy & loss for the training & test set
                train_acc, train_loss = acc_loss_rdm_cu(num_qubit,train_cir,y_train,num_train)
                test_acc, test_loss = acc_loss_rdm_cu(num_qubit,test_cir,y_test,num_test)
                push!(loss_train_history, train_loss)
                push!(loss_test_history, test_loss)
                push!(acc_train_history, train_acc)
                push!(acc_test_history, test_acc)
                if k % 10 == 0
                    @printf("\nStep=%d, loss=%.3f, acc=%.3f, test_loss=%.3f,test_acc=%.3f\n",k,train_loss,train_acc,test_loss,test_acc)
                end
                # at each training epoch, randomly choose a batch of samples from the training set
                batch_index = randperm(size(x_train)[2])[1:batch_size]
                batch_cir = train_cir[batch_index]
                y_batch = y_train[batch_index,:]
                rdm = batch([density_matrix(focus!(zero_state(num_qubit) |> batch_cir[i], (pos_))).state for i in 1:batch_size])[:,:,1,:]
                q_ = zeros(batch_size,2);
                for i=1:batch_size
                    q_[i,:] = diag(rdm[:,:,i]) |> real
                end

                Arr = Array{Float64}(zeros(batch_size,nparameters(batch_cir[1])))
                for i in 1:batch_size
                    Arr[i,:] = expect'(op0, zero_state(num_qubit)=>batch_cir[i])[2]
                end
                C = [Arr, -Arr]
                grads = collect(mean([-sum([y_batch[i,j]*((1 ./ q_)[i,j])*batch(C)[i,:,j] for j in 1:2]) for i=1:batch_size]))
                
                updates = Flux.Optimise.update!(optim, copy(ini_theta), grads);
                ini_theta = updates
                # update the parameters
                for i in 1:num_train
                    dispatch!(train_cir[i], x_train[:,i]+ini_theta)
                end
                for i in 1:num_test
                    dispatch!(test_cir[i], x_test[:,i]+ini_theta)
                end
            end

            t2 = time()
            T = (t2 - t1)/3600

            matwrite("CODE/Github_ClassifierProject/block_encode/BenchmarkData_Table4/FashionMNIST_20220429_Ent"*string(Ent_index)*"_Scal"*string(Scal_index)*"_"*string(rep)*".mat", Dict(
                    "loss_train_history" => loss_train_history,
                    "acc_train_history" => acc_train_history,
                    "loss_test_history" => loss_test_history,
                    "acc_test_history" => acc_test_history,
                    "nbit" => num_qubit,
                    "batch_size" => batch_size,
                    "lr" => lr,
                    "niters" => niters,
                    "num_train" => num_train,
                    "num_test" => num_test,
                    "time_used_hour" => T
                )
            )
        end
    end
    task(args) = f(args[1])
end
res = pmap(task, Array(1:45))