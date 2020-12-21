### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 42338250-4387-11eb-1d21-a3b1adbf202c
begin
	using Pkg
    Pkg.activate(pwd())
end

# ╔═╡ 593f78f0-4387-11eb-3175-93d23999fd88
begin
	using CSV
	using Statistics
	using DataFrames
end

# ╔═╡ 9d6fccb0-4395-11eb-34d8-8de61157070c
using Colors

# ╔═╡ 1d3942f0-4387-11eb-05e8-85639ea3579a
md"""
# Data Mining
## Assignment 3 - Correlation Matrix
### Aadam - CS1945
"""

# ╔═╡ b3196bd0-43a3-11eb-3831-65d049f3abef
md"""
# Data Transformation
Reads `iris` data from [http://archive.ics.uci.edu/ml/datasets/Iris](http://archive.ics.uci.edu/ml/datasets/Iris), transforms it, and saves the transformed data into `data/iris_transformed.txt` file.
"""

# ╔═╡ bd551d90-4387-11eb-1e63-a5993648b680
function transform(csv_path)
	df = CSV.read(csv_path, DataFrame, datarow=1) # read from CSV
	insertcols!(df, 1, :row => axes(df, 1) ) # add column with row numbers
	
	# change categorical values to numeric
	df.Column5 = replace.(df.Column5, Ref("Iris-setosa" => 1))
	df.Column5 = replace.(df.Column5, Ref("Iris-versicolor" => 2))
	df.Column5 = replace.(df.Column5, Ref("Iris-virginica" => 3))
	df[!,:Column5] = parse.(Int,df[!, :Column5])
	
	# add data size info
	open("data/iris_transformed.txt", "w") do io
		println(io, size(df)[1])
		println(io, size(df)[2])
		println(io, "")
	end
	
	# append CSV data to file
	CSV.write("data/iris_transformed.txt", df, header=false, append=true)
	df
end

# ╔═╡ 4f544fa0-438c-11eb-1c71-f10ea0b7747c
df = transform("data/iris.data")[:, 2:end-1]

# ╔═╡ afbd4320-4390-11eb-1e7a-df636c495cba
md"""
# Calculate Correlation Matrix
The correlation matrix will be a NXN matrix (where N is number of records in your input dataset) containing Pearson’s correlation coefficient between each of the row in data matrix. Pearson’s correlation coefficient formula:

$\frac{{}\sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y})}
{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2(y_i - \overline{y})^2}}$
"""

# ╔═╡ eaa97310-438f-11eb-2ad5-d141a81d9791
function cor_col(x::AbstractVector, mx, y::AbstractVector, my)
    n = length(x)
    
	# Initialize the accumulators
	xx = zero(sqrt(abs2(one(x[1]))))
	yy = zero(sqrt(abs2(one(y[1]))))
	xy = zero(x[1] * y[1]')

	for i in eachindex(x, y)
		xi = x[i] - mx
		yi = y[i] - my
		xx += abs2(xi)
		yy += abs2(yi)
		xy += xi * yi'
	end
    return clamp(xy / max(xx, yy) / sqrt(min(xx, yy) / max(xx, yy)), -1, 1)
end

# ╔═╡ 578bf662-43a4-11eb-3d40-0b491ad520b2
function cor(X::AbstractMatrix)
	n = size(X)[1]
	cormat = zeros(n, n)
	
	for i in 1:n
		for j in i:n
			cormat[i, j] = cor_col(X[i, :], mean(X[i, :]), X[j, :], mean(X[j, :]))
			cormat[j, i] = cormat[i, j]
		end
	end
	cormat
end

# ╔═╡ 0eb51410-438d-11eb-0a4b-a59b660f63dd
cor_matrix = cor(Matrix(df))

# ╔═╡ 0ecb0d10-438d-11eb-2b61-a783a5176442
md"""
# Discretize
Calculate median/mean of each column of the correlation matrix and set all the values in that column that are above the calculated median/mean to 1 and rest to 0.
"""

# ╔═╡ 0ee10610-438d-11eb-193c-3d806714919a
cor_mean = mean(cor_matrix, dims=1)

# ╔═╡ 0ef59f80-438d-11eb-15f0-6986acb67f16
cor_median = median(cor_matrix, dims=1)

# ╔═╡ 0b236830-4395-11eb-129b-65553ddfc548
cor_mean_matrix = zeros(size(cor_matrix));

# ╔═╡ 16ab1130-4395-11eb-3d46-ad660d6c74c6
cor_median_matrix = zeros(size(cor_matrix));

# ╔═╡ 0f0d1f20-438d-11eb-27ef-b9a25e3b9436
for i in 1:size(cor_matrix)[1]
	cor_mean_matrix[:, i] = cor_matrix[:, i] .> cor_mean[i]
	cor_median_matrix[:, i] = cor_matrix[:, i] .> cor_median[i]
end

# ╔═╡ c0379bb0-4395-11eb-269b-19b8db40b0d5
md"""
# Visualize
Convert the discretized matrix into bitmap.
"""

# ╔═╡ 1257b542-4398-11eb-3aec-d9c38c46ccf3
md"""
### Median Discretized Correlation Matrix
"""

# ╔═╡ 9d8884d2-4395-11eb-294b-5d0b022f06e5
Gray.(cor_median_matrix)

# ╔═╡ 23e78ec2-4398-11eb-35d0-0b6de4d620de
md"""
### Mean Discretized Correlation Matrix
"""

# ╔═╡ f627fab0-4397-11eb-00e1-0f3bc6802852
Gray.(cor_mean_matrix)

# ╔═╡ Cell order:
# ╟─1d3942f0-4387-11eb-05e8-85639ea3579a
# ╟─42338250-4387-11eb-1d21-a3b1adbf202c
# ╠═593f78f0-4387-11eb-3175-93d23999fd88
# ╟─b3196bd0-43a3-11eb-3831-65d049f3abef
# ╠═bd551d90-4387-11eb-1e63-a5993648b680
# ╠═4f544fa0-438c-11eb-1c71-f10ea0b7747c
# ╟─afbd4320-4390-11eb-1e7a-df636c495cba
# ╠═eaa97310-438f-11eb-2ad5-d141a81d9791
# ╠═578bf662-43a4-11eb-3d40-0b491ad520b2
# ╠═0eb51410-438d-11eb-0a4b-a59b660f63dd
# ╟─0ecb0d10-438d-11eb-2b61-a783a5176442
# ╠═0ee10610-438d-11eb-193c-3d806714919a
# ╠═0ef59f80-438d-11eb-15f0-6986acb67f16
# ╠═0b236830-4395-11eb-129b-65553ddfc548
# ╠═16ab1130-4395-11eb-3d46-ad660d6c74c6
# ╠═0f0d1f20-438d-11eb-27ef-b9a25e3b9436
# ╟─c0379bb0-4395-11eb-269b-19b8db40b0d5
# ╠═9d6fccb0-4395-11eb-34d8-8de61157070c
# ╟─1257b542-4398-11eb-3aec-d9c38c46ccf3
# ╠═9d8884d2-4395-11eb-294b-5d0b022f06e5
# ╟─23e78ec2-4398-11eb-35d0-0b6de4d620de
# ╠═f627fab0-4397-11eb-00e1-0f3bc6802852
