# Quantum Neural Network Classifiers

An implementation of quantum neural network (QNN) classifiers

## Download

```bash
$ git clone https://github.com/LWKJJONAK/Quantum_Neural_Network_Classifiers
```

The codes are mainly provided in jupyter notebook formats with intermediate results presented

## Contents

- [Amplitude-Encoding Based QNNs: Basic Building Blocks](amplitude_encode/amplitude_encoding_Sec_3.1.ipynb)
- [Amplitude-Encoding Based QNNs: An Example Code For The Whole Training Procedure](amplitude_encode/an_example_code_for_the_whole_training_procedure.ipynb)
- [Circuit-Encoding Based QNNs: An Example Code For The Whole Training Procedure](circuit_encode/circuit_encoding_Sec_4.1.ipynb)

## Built With

* [Yao](https://github.com/QuantumBFS/Yao.jl) - A framework for Quantum Algorithm Design

Examples of using Yao in other projects
- [TensorNetwork Inspired Circuits](https://github.com/GiggleLiu/QuantumPEPS.jl)
- [QuAlgorithmZoo](https://github.com/QuantumBFS/QuAlgorithmZoo.jl/tree/master/examples)
- [QuODE](https://github.com/QuantumBFS/QuDiffEq.jl)

## To Cite
```bibtex
@article{Li2022QuantumNeural,
    title={Quantum Neural Network Classifiers:  Code, Brief Review, and Benchmarks},
    author={Li, Weikang and Lu, Zhide and Deng, Dong-Ling},
    eprint={arXiv:2205.00000},
    url={https://arxiv.org/abs/2205.00000}
}
```
We also experimentally implement the circuit-encoding based QNNs for large-scale (256-dimensional) real-life image classification, see the paper below
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
