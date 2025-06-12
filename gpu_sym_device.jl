using CUDA
using Test

k = 10
n = 3

arr1 = fill((fill(1.0f0, n), fill(1.0f0, n, n)), k)
arr2 = fill((fill(2.0f0, n), fill(2.0f0, n, n)), k)

function add!(arr1_d, arr2_d, arr3_d)
  n = size(arr3_d, 1)
  k = size(arr3_d, 2) ÷ (n + 1)
  start = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  stride = blockDim().x * gridDim().x
  tupsize = n * (n + 3) ÷ 2
  for idx in start:stride:k*n*(n+3)÷2
    tupidx = (idx - 1) ÷ tupsize + 1
    tupoffset = (idx - 1) % tupsize + 1
    if tupoffset <= n
      i = tupoffset
      j = (tupidx - 1) * (n + 1) + 1
      @inbounds arr3_d[i, j] = arr1_d[i, j] + arr2_d[i, j]
    else
      matoffset = tupoffset - n
      mati = Int(floor((2*n+3-sqrt((2*n+3)^2-8*(n+matoffset)))/2))
      matj = matoffset-(mati-1)*(2*n-mati) ÷ 2
      i, j = mati, (tupidx - 1) * (n + 1) + 1 + matj
      invi, invj = matj, (tupidx - 1) * (n + 1) + 1 + mati
      @inbounds arr3_d[invi, invj] = arr3_d[i, j] = arr1_d[i, j] + arr2_d[i, j]
    end
  end
  nothing
end

arr1_h = mapreduce(tup->hcat(tup[1], tup[2]), hcat, arr1)
arr2_h = mapreduce(tup->hcat(tup[1], tup[2]), hcat, arr2)
arr1_d = CuArray(arr1_h)
arr2_d = CuArray(arr2_h)
arr3_d = CuArray{Float32}(undef, n, k*(n+1))

nthreads = 1024
nblocks = ceil(Int, k*(n*(n+3)÷2)/nthreads)
CUDA.@sync @cuda threads=nthreads blocks=nblocks add!(arr1_d, arr2_d, arr3_d)
arr3_h = Array(arr3_d)
arr3 = [(arr3_h[:,i], arr3_h[:,i+1:i+n]) for i in 1:(n+1):size(arr3_h, 2)]

expected = fill((fill(3.0f0, n), fill(3.0f0, n, n)), k)
println(@test arr3 == expected)
