using Test

k = 10
n = 3

arr1 = fill((fill(1.0f0, n), fill(1.0f0, n, n)), k)
arr2 = fill((fill(2.0f0, n), fill(2.0f0, n, n)), k)
arr3 = map(.+, arr1, arr2)

expected = fill((fill(3.0f0, n), fill(3.0f0, n, n)), k)
println(@test arr3 == expected)
