using Distributed
procs = addprocs(36)

@everywhere begin

    using Yao, Zygote, YaoPlots, CuYao, YaoExtensions
    using LinearAlgebra, Statistics, Random, StatsBase, ArgParse, Distributions
    using Printf, BenchmarkTools, MAT
    using Flux: batch, Flux

    vars = matread("CODE/Github_ClassifierProject/dataset/SPT_10qubit_wk.mat")
    x_train = vars["x_train"]
    y_train = vars["y_train"]
    x_test = vars["x_test"]
    y_test = vars["y_test"]
    
    num_train = 1000
    num_test = 200
    x_train = x_train[:,1:num_train]
    y_train = y_train[1:num_train,:]
    x_test = x_test[:,1:num_test]
    y_test = y_test[1:num_test,:];

    repeat_ = 100
    num_qubit = 10    # number of qubits
    batch_size = 100 # batch size
    lr = 0.005       # learning rate
    niters = 200;     # number of iterations
    optim = Flux.ADAM(lr); # Adam optimizer

    # index of qubit that will be measured
    pos_ = num_qubit;       
    op0 = put(num_qubit, pos_=>0.5*(I2+Z))
    op1 = put(num_qubit, pos_=>0.5*(I2-Z));
    
    x_train_yao = zero_state(num_qubit,nbatch=num_train)
    x_train_yao.state = x_train;
    cu_x_train_yao = copy(x_train_yao) |> cpu;

    x_test_yao = zero_state(num_qubit,nbatch=num_test)
    x_test_yao.state  = x_test;
    cu_x_test_yao = copy(x_test_yao) |> cpu;
    
    include("../library/Layer.jl")
    include("../library/Loss_Accuracy.jl")
    
    function Ent_Layer_(nbit::Int64)
        Ent_CZ(nbit),Ent_CX(nbit),chain(nbit, Ent_CX(nbit), Ent_CX(nbit))
    end
    Parameterized_Layer(nbit::Int64) = Params_Layer(nbit)
    
    function f(i_::Int64)
        
        D = [1,2,3,4,5,6,7,8,9,10,11,12]
        D_index = (i_-1)%12+1
        Ent_index = ((i_-D_index)/12+1) |> Int64
        depth = D[D_index]
        
        Ent_Layer(nbit::Int64) = Ent_Layer_(nbit)[Ent_index]
        Composite_Block(nbit::Int64) = chain(nbit, Parameterized_Layer(nbit::Int64), Ent_Layer(nbit::Int64))
        circuit = chain(Composite_Block(num_qubit) for _ in 1:depth)
        
        for rep in 1:repeat_
            t1 = time()
            # assign random initial parameters to the circuit
            dispatch!(circuit, :random)
            params = parameters(circuit);

            # record the training history
            loss_train_history = Float64[]
            acc_train_history = Float64[]
            loss_test_history = Float64[]
            acc_test_history = Float64[];

            for k in 0:niters
                global lr
                global ini_param

                # calculate the accuracy & loss for the training & test set
                train_acc, train_loss = acc_loss_rdm_cu(circuit,cu_x_train_yao,y_train,num_train)
                test_acc, test_loss = acc_loss_rdm_cu(circuit,cu_x_test_yao,y_test,num_test)
                push!(loss_train_history, train_loss)
                push!(loss_test_history, test_loss)
                push!(acc_train_history, train_acc)
                push!(acc_test_history, test_acc)
                if k % 1 == 0
                    #@printf("\nStep=%d, loss=%.3f, acc=%.3f, test_loss=%.3f,test_acc=%.3f\n",k,train_loss,train_acc,test_loss,test_acc)
                end

                # at each training epoch, randomly choose a batch of samples from the training set
                batch_index = randperm(size(x_train)[2])[1:batch_size]
                x_batch = x_train[:,batch_index]
                y_batch = y_train[batch_index,:];
                # prepare these samples into quantum states
                x_batch_1 = copy(x_batch)
                x_batch_yao = zero_state(num_qubit,nbatch=batch_size)
                x_batch_yao.state = x_batch_1;
                cu_x_batch_yao = copy(x_batch_yao) |> cpu;
                batc = [zero_state(num_qubit) for i in 1:batch_size]
                for i in 1:batch_size
                    batc[i].state = x_batch_1[:,i:i]
                end

                # for all samples in the batch, repeatly measure their qubits at position pos_ 
                # on the computational basis
                reg_ = focus!(copy(cu_x_batch_yao) |> circuit, (pos_)) |> cpu
                rdm = density_matrix(reg_).state;
                q_ = zeros(batch_size,2);
                for i=1:batch_size
                    q_[i,:] = diag(rdm[:,:,i]) |> real
                end

                # calculate the gradients w.r.t. the cross-entropy loss function
                Arr = Array{Float64}(zeros(batch_size,nparameters(circuit)))
                for i in 1:batch_size
                    Arr[i,:] = expect'(op0, copy(batc[i])=>circuit)[2]
                end
                C = [Arr, -Arr]
                grads = collect(mean([-sum([y_batch[i,j]*((1 ./ q_)[i,j])*batch(C)[i,:,j] for j in 1:2]) for i=1:batch_size]))

                # update the parameters
                updates = Flux.Optimise.update!(optim, params, grads);
                dispatch!(circuit, updates) 
            end

            t2 = time()
            T = (t2 - t1)/3600

            matwrite("CODE/Github_ClassifierProject/amplitude_encode/BenchmarkData_Table1/SPT10qubit_20220419_Ent"*string(Ent_index)*"_Dep"*string(D_index)*"_"*string(rep)*".mat", Dict(
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
res = pmap(task, Array(1:36))





