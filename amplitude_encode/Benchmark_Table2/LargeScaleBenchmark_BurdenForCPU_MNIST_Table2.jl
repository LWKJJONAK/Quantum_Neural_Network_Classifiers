using Distributed
procs = addprocs(30)

@everywhere begin
    using Yao, Zygote, YaoPlots, CuYao, YaoExtensions
    using LinearAlgebra, Statistics, Random, StatsBase, ArgParse, Distributions
    using Printf, BenchmarkTools, MAT
    using Flux: batch, Flux
    using SparseArrays
end

@everywhere begin
    vars = matread("CODE/Github_ClassifierProject/dataset/MNIST_1_9_wk.mat")
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
    num_qubit = 8    # number of qubits
    batch_size = 100 # batch size
    lr = 0.005       # learning rate
    niters = 400;     # number of iterations
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
    
    Parameterized_Layer(nbit::Int64) = Params_Layer(nbit)
    function spin_operator(num_spins::Int, sites::Vector{Int}, opts::Vector{Int})
        ops = [sparse([0 1+0im; 1+0im 0]), sparse([0 -1im; 1im 0]), 
            sparse([1+0im 0; 0 -1+0im]), sparse([1+0im 0; 0 1+0im]), 
            sparse([0 1+0im; 0 0]), sparse([0 0; 1+0im 0])]

        idx = [4 for i = 1:num_spins]
        idx[sites] = opts

        opt = sparse([1])
        for i in 1:num_spins
            opt = kron(opt, ops[idx[i]])
        end
        return opt
    end

    function opts(num_qubits::Int)
        tz = [spin_operator(num_qubits, [k], [3]) for k = 1:num_qubits]
        txx = [spin_operator(num_qubits, [k,k+1], [1,1]) for k = 1:num_qubits-1]
        tyy = [spin_operator(num_qubits, [k,k+1], [2,2]) for k = 1:num_qubits-1]

        return tz, txx, tyy
    end

    α = (√5 - 1) / 2
    φ = 0
    # φ = rand(Uniform(0,2π))
    hs = [cos(2π*α*k + φ) for k = 1:num_qubit]
    m = chain(num_qubit, [put(num_qubit-k+1=>X) for k = 2:2:num_qubit])
    tz, txx, tyy = opts(num_qubit);
    dm = 0.
    dw = 0.
    gm = 1.
    gw = 0.
    dss = (dm .+ dw * rand(Uniform(0,1), 1)) * hs'
    gss = gm .+ gw * rand(Uniform(-1,1), 1, num_qubit-1)
    h = sum(dss[1, :] .* tz) .+ sum(gss[1, :] .* (txx + tyy))
    
    D = [1,3,5]
    Time_Evo = [0.1,0.3,0.5,0.7,1.0,2.0,3.0,5.0,7.0,10.0]
    @const_gate a1 = exp(-im * Time_Evo[1] * h |> Array)
    @const_gate a2 = exp(-im * Time_Evo[2] * h |> Array)
    @const_gate a3 = exp(-im * Time_Evo[3] * h |> Array)
    @const_gate a4 = exp(-im * Time_Evo[4] * h |> Array)
    @const_gate a5 = exp(-im * Time_Evo[5] * h |> Array)
    @const_gate a6 = exp(-im * Time_Evo[6] * h |> Array)
    @const_gate a7 = exp(-im * Time_Evo[7] * h |> Array)
    @const_gate a8 = exp(-im * Time_Evo[8] * h |> Array)
    @const_gate a9 = exp(-im * Time_Evo[9] * h |> Array)
    @const_gate a10 = exp(-im * Time_Evo[10] * h |> Array)
    Ent_Layer_Analog = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
end

@everywhere begin
    function f(i_::Int64)
        D_index = (i_-1)%3 + 1
        Time_index = ((i_-D_index)/3+1) |> Int64
        depth = D[D_index]
        
        Ent_Layer(nbit::Int64) = Ent_Layer_Analog[Time_index]
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

            matwrite("CODE/Github_ClassifierProject/amplitude_encode/BenchmarkData_Table2/MNIST_20220424_Time"*string(Time_index)*"_Dep"*string(D_index)*"_"*string(rep)*".mat", Dict(
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
res = pmap(task, Array(1:30))





