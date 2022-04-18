# if you are using Yao#v0.7.4 and don't wish to change the version,
# overloading the density_matrix() as below will make the code work

import YaoAPI.density_matrix

function dens_m(M::DensityMatrix)
    M.state
end

struct density_matrix_
    state::Array{ComplexF64, 3}
end

function density_matrix(reg::BatchedArrayReg)
    temp = density_matrix.(reg)
    density_matrix_(cat(dens_m.(temp)...;dims = 3))
end