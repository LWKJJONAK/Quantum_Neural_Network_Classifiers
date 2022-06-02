Ent_CX(nbit::Int64) = (nbit%2 == 0) ? 
    chain(chain(nbit,control(i,i+1=>X) for i in 1:2:nbit-1),
          chain(nbit,control(i,i+1=>X) for i in 2:2:nbit-2)) : 
    chain(chain(nbit,control(i,i+1=>X) for i in 1:2:nbit-2),
          chain(nbit,control(i,i+1=>X) for i in 2:2:nbit-1))

Ent_CZ(nbit::Int64) = (nbit%2 == 0) ? 
    chain(chain(nbit,control(i,i+1=>Z) for i in 1:2:nbit-1),
          chain(nbit,control(i,i+1=>Z) for i in 2:2:nbit-2)) : 
    chain(chain(nbit,control(i,i+1=>Z) for i in 1:2:nbit-2),
          chain(nbit,control(i,i+1=>Z) for i in 2:2:nbit-1))

RX_layer(nbit::Int64) = chain(put(nbit, i => Rx(0)) for i in 1:nbit)
RZ_layer(nbit::Int64) = chain(put(nbit, i => Rz(0)) for i in 1:nbit)
Params_Layer(nbit::Int64) = chain(RX_layer(nbit),RZ_layer(nbit),RX_layer(nbit))