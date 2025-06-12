using CUDA
using BenchmarkTools
using Plots

function benchmark_wrapper(k, n)
  arr1_d = CUDA.fill(1.0f0, k*n*(n+3)÷2)
  arr2_d = CUDA.fill(2.0f0, k*n*(n+3)÷2)
  @belapsed CUDA.@sync $arr1_d + $arr2_d
end

ns = [10, 20, 50, 100]
ks = [100, 200, 500, 1000]
t = [10^6 * benchmark_wrapper(k, n) for k in ks for n in ns]

wireframe(ns, ks, t, xlabel="n", ylabel="k", zlabel="time, us")
savefig("t_of_nk.png")

tmesh = reshape(t, length(ns), length(ks))
klabels = labels=reshape(["k=$k" for k in ks], 1, :)
plot(ns, tmesh, marker=:diamond, xlabel="n", ylabel="time, us", labels=klabels)
savefig("t_of_n.png")

nlabels = labels=reshape(["n=$n" for n in ns], 1, :)
plot(ks, transpose(tmesh), marker=:diamond, xlabel="k", ylabel="time, us", labels=nlabels)
savefig("t_of_k.png")
