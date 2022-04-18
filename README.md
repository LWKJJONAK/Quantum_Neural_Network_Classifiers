# Quantum Neural Network Classifiers

An implementation of quantum neural network (QNN) classifiers

![](library/framework.png)

## Download

```bash
$ git clone https://github.com/LWKJJONAK/Quantum_Neural_Network_Classifiers
```

The codes are mainly provided in jupyter notebook formats with intermediate results presented

## Environment

For the packages used in our simulations, there are multiple ways to install them.
For the first way, the list of packages we used can be found at [Project.toml](https://github.com/LWKJJONAK/Quantum_Neural_Network_Classifiers/blob/main/Project.toml) and the users can install them accordingly.
For example, if we want to install Yao.jl, simply type ] to enter the Package manager mode
and type "add Yao". Then the target package will be installed.

An alternative way for Linux users is shown as follows.
Julia's packages are installed at ~/.julia by default. So a convenient way to install the packages for our use is to download the two files ([Manifest.toml](https://github.com/LWKJJONAK/Quantum_Neural_Network_Classifiers/blob/main/Manifest.toml) and [Project.toml](https://github.com/LWKJJONAK/Quantum_Neural_Network_Classifiers/blob/main/Project.toml)) from this page, and put them at the folder .julia/environments/v1.7 (assuming we are using the version 1.7 of Julia). 
In the last step, typing ] to enter the Package manager mode,
typing "update" or "up" to update the packages and typing "build" will finish the installation.
In this way, we can install all the packages at once, while some of them may not be necessary.
For better coding experience, version 1.6 or higher of Julia is suggested.

## Contents

- [Amplitude-Encoding Based QNNs: Basic Building Blocks](amplitude_encode/amplitude_encoding_Sec_3.1.ipynb)
- [Amplitude-Encoding Based QNNs: An Example Code For The Whole Training Procedure](amplitude_encode/an_example_code_for_the_whole_training_procedure.ipynb)
- [Block-Encoding Based QNNs: An Example Code For The Whole Training Procedure](block_encode/block_encoding_Sec_4.1.ipynb)

## Built With

* [Yao](https://github.com/QuantumBFS/Yao.jl) - A framework for Quantum Algorithm Design

Detailed installation instructions and toturials of Julia and Yao.jl can be found at [julialang.org](https://julialang.org/) and [yaoquantum.org](https://yaoquantum.org/), respectively.

Examples of using Yao in other projects
- [TensorNetwork Inspired Circuits](https://github.com/GiggleLiu/QuantumPEPS.jl)
- [QuAlgorithmZoo](https://github.com/QuantumBFS/QuAlgorithmZoo.jl/tree/master/examples)
- [QuODE](https://github.com/QuantumBFS/QuDiffEq.jl)

## To Cite
```bibtex
@article{Li2022QuantumNeural,
    title={Quantum Neural Network Classifiers: Code, Brief Review, and Benchmarks},
    author={Li, Weikang and Lu, Zhide and Deng, Dong-Ling},
    eprint={arXiv:2205.00000},
    url={https://arxiv.org/abs/2205.00000}
}
```
We experimentally implement the block-encoding based QNNs for large-scale (256-dimensional) real-life images' classifications, see also the paper
```bibtex
@article{Ren2022Experimental,
    title={Experimental quantum adversarial learning with programmable superconducting qubits},
    author={W Ren*, W Li*, S Xu*, K Wang, W Jiang, F Jin, X Zhu, J Chen, Z Song, P Zhang, H Dong, X Zhang, 
    J Deng, Y Gao, C Zhang, Y Wu, B Zhang, Q Guo, H Li, Z Wang, J Biamonte, C Song, DL Deng, and H. Wang},
    eprint={arXiv:2204.01738},
    url={https://arxiv.org/abs/2204.01738}
}
```

## License

Released under [GPL-3.0 License](https://github.com/LWKJJONAK/Quantum_Neural_Network_Classifiers/blob/main/LICENSE)
