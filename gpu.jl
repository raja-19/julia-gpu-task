using CUDA
using Test
using BenchmarkTools

k = 10
n = 3

arr1 = fill((fill(1.0f0, n), fill(1.0f0, n, n)), k)
arr2 = fill((fill(2.0f0, n), fill(2.0f0, n, n)), k)

arr1_h = mapreduce(tup->hcat(tup[1], tup[2]), hcat, arr1)
arr2_h = mapreduce(tup->hcat(tup[1], tup[2]), hcat, arr2)
arr1_d = CuArray(arr1_h)
arr2_d = CuArray(arr2_h)
arr3_d = arr1_d + arr2_d
arr3_h = Array(arr3_d)
arr3 = [(arr3_h[:,i], arr3_h[:,i+1:i+n]) for i in 1:(n+1):size(arr3_h, 2)]

expected = fill((fill(3.0f0, n), fill(3.0f0, n, n)), k)
println(@test arr3 == expected)

@btime CUDA.@sync $arr1_d + $arr2_d
