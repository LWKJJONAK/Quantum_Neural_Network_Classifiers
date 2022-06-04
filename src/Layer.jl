ent_cx(nbit::Int64) = (nbit%2 == 0) ? 
    chain(chain(nbit,control(i,i+1=>X) for i in 1:2:nbit-1),
          chain(nbit,control(i,i+1=>X) for i in 2:2:nbit-2)) : 
    chain(chain(nbit,control(i,i+1=>X) for i in 1:2:nbit-2),
          chain(nbit,control(i,i+1=>X) for i in 2:2:nbit-1))

ent_cz(nbit::Int64) = (nbit%2 == 0) ? 
    chain(chain(nbit,control(i,i+1=>Z) for i in 1:2:nbit-1),
          chain(nbit,control(i,i+1=>Z) for i in 2:2:nbit-2)) : 
    chain(chain(nbit,control(i,i+1=>Z) for i in 1:2:nbit-2),
          chain(nbit,control(i,i+1=>Z) for i in 2:2:nbit-1))

rx_layer(nbit::Int64) = chain(put(nbit, i => Rx(0)) for i in 1:nbit)
rz_layer(nbit::Int64) = chain(put(nbit, i => Rz(0)) for i in 1:nbit)
params_layer(nbit::Int64) = chain(rx_layer(nbit),rz_layer(nbit),rx_layer(nbit))