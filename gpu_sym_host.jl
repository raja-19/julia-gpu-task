using CUDA
using Test
using BenchmarkTools

k = 10
n = 3

arr1 = fill((fill(1.0f0, n), fill(1.0f0, n, n)), k)
arr2 = fill((fill(2.0f0, n), fill(2.0f0, n, n)), k)

function sym2flat(S, n)
  @inbounds [S[i, j] for i in 1:n for j in i:n]
end

function flat2sym(a, n)
  S = Array{Float32}(undef, n, n)
  for i in 1:n
    for j in i:n
      flatIdx = (i-1)*(2*n-i)÷2+j
      @inbounds S[i, j] = S[j, i] = a[flatIdx]
    end
  end
  S
end

arr1_h = mapreduce(tup->vcat(tup[1], sym2flat(tup[2], n)), vcat, arr1)
arr2_h = mapreduce(tup->vcat(tup[1], sym2flat(tup[2], n)), vcat, arr2)
arr1_d = CuArray(arr1_h)
arr2_d = CuArray(arr2_h)
arr3_d = arr1_d + arr2_d
arr3_h = Array(arr3_d)
arr3 = [(arr3_h[1:n], flat2sym(arr3_h[n+1:n*(n+3)÷2], n)) for i in 1:n*(n+3)÷2:length(arr3_h)]

expected = fill((fill(3.0f0, n), fill(3.0f0, n, n)), k)
println(@test arr3 == expected)

@btime CUDA.@sync $arr1_d + $arr2_d
