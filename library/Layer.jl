single_CNOT_layer(nbits) = vcat([2i=>(2i+1) for i in 1:Int64(floor((nbits-1)/2))],[(2i-1)=>2i for i in 1:Int64(floor((nbits)/2))])

CNOT_layer(nbits) = vcat([2i=>(2i+1) for i in 1:Int64(floor((nbits-1)/2))],[(2i-1)=>2i for i in 1:Int64(floor((nbits)/2))],[2i=>(2i+1) for i in 1:Int64(floor((nbits-1)/2))],[(2i-1)=>2i for i in 1:Int64(floor((nbits)/2))])

RX_layer(nbit::Int64) = chain(put(nbit, i => Rx(0)) for i in 1:nbit)
RZ_layer(nbit::Int64) = chain(put(nbit, i => Rz(0)) for i in 1:nbit)
E_layer(nbit::Int64) = (nbit%2 == 0) ? chain(chain(nbit,control(i,i+1=>X) for i in 1:2:nbit-1),chain(nbit,control(i,i+1=>X) for i in 2:2:nbit-2)) : chain(chain(nbit,control(i,i+1=>X) for i in 1:2:nbit-2),chain(nbit,control(i,i+1=>X) for i in 2:2:nbit-1))

Classifier_layer(nbit::Int64) = chain(RZ_layer(nbit),RX_layer(nbit),RZ_layer(nbit),E_layer(nbit),E_layer(nbit))
Classifier_X_layer(nbit::Int64) = chain(RX_layer(nbit),E_layer(nbit),E_layer(nbit))
Classifier_XZ_layer(nbit::Int64) = chain(RX_layer(nbit),RZ_layer(nbit),E_layer(nbit),E_layer(nbit))