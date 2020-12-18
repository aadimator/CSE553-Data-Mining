### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 85d5a830-4116-11eb-1479-630b53b50e94
begin
	using Pkg
    Pkg.activate(pwd())
end

# ╔═╡ 4862c2d0-4116-11eb-3f26-1bdefcb9ce3d
begin
	using UCIData
	using PlutoUI
	using DataFrames
	using Statistics
	using LinearAlgebra
	using Plots
	using Clustering
end

# ╔═╡ c73d5420-4126-11eb-19b3-bf6c04aef086
md"""
Assignment 2
"""

# ╔═╡ 360ab320-411d-11eb-2d11-c9fe7b3137d1
plotly()

# ╔═╡ 73936e9e-4116-11eb-0594-1d202bd073cd
function Kmeans(X, k; max_iters = 300, tol = 1e-5)
    # Reshape 2D array to a 1D array with length of all training examples
    # where each example is of size (n, ) ie the new array is just a list of example array
    X_array_list = collect(eachrow(X))
	print(X_array_list)

    # Save some info on the incoming data
    N = length(X_array_list)  # Length of all training examples
    n = length(X_array_list[1])  # Length of a single training example
    distances = zeros(N)  # Empty vector for all training examples. Useful later

    # Step 1: Random initialization
    reps_centroids = [zeros(n) for grp = 1:k]  # Initiate centroids for each
    labels = rand(1:k, N)  # Randomly assign labels (between 1 to k) to all training examples

    J_previous = Inf

    for iter = 1:max_iters

        # Step 2: Update the representative centroids for each group
        for j = 1:k
            # get group indices for each group
            group_idx = [i for i = 1:N if labels[i] == j]

            # use group indices to locate each group
            reps_centroids[j] = mean(X_array_list[group_idx]);
        end;

        # Step 3: Update the group labels
        for i = 1:N
            # compute the distance between each example and the updated representative centroid
            nearest_rep_distance = [norm(X_array_list[i] - reps_centroids[x]) for x = 1:k]

            # update distances and label arrays with value and index of closest neighbour
            # findmin returns the min value & index location
            distances[i], labels[i] = findmin(nearest_rep_distance)
        end;

        # Step 4: Compute the clustering cost
        J = (norm(distances)^ 2) / N

        # Show progress and terminate if J stopped decreasing.
        println("Iteration ", iter, ": Jclust = ", J, ".")

        # Final Step 5: Check for convergence
        if iter > 1 && abs(J - J_previous) < (tol * J)
            # TODO: Calculate the sum of squares

            # Terminate algorithm with the assumption that K-means has converged
            return labels, reps_centroids
    
        end

        J_previous = J
    end

end

# ╔═╡ 9accd320-4117-11eb-0fc4-15e8e85040a8
datasets = ["iris" => "iris", "glass-identification" => "glass"];

# ╔═╡ 8b43a9a0-4118-11eb-127d-d57dcea189e1
md"""
Dataset: $(@bind selected_dataset Select(datasets))

"""

# ╔═╡ 73c57b20-4116-11eb-3dba-01c9f7b91fa7
data = UCIData.dataset(selected_dataset);

# ╔═╡ 5cddc6a0-411c-11eb-2380-1b9442938a9b
features = collect(Matrix(data[:, 2:end-1])');

# ╔═╡ bec93fa0-4119-11eb-026e-0be14d6a5779
md"""
Vlaue of **k**: $(@bind k Slider(1:size(UCIData.dataset(selected_dataset))[1]; show_value=true))
"""

# ╔═╡ cc1f379e-4127-11eb-375c-d782bcac175d
# size(UCIData.dataset(selected_dataset))[1]

# ╔═╡ 737b2bb0-4116-11eb-124b-d981032a9638
result = kmeans(features, k);

# ╔═╡ ce4134a0-4124-11eb-2a7c-9761e81b2064
md"""
## Plot Clusters
Feature 1: $(@bind feature1 Select(string.(collect(1:size(features)[1]))))
Feature 2: $(@bind feature2 Select(string.(collect(1:size(features)[1])));)
"""

# ╔═╡ 73454ea0-4116-11eb-0938-f9a06f3dbaf8
scatter(features[parse(Int,feature1), :], features[parse(Int, feature2), :], 
        marker_z = result.assignments, 
        color =:gist_rainbow, legend = false)

# ╔═╡ 548ac220-4124-11eb-0da7-b96eb51b89d9


# ╔═╡ Cell order:
# ╠═c73d5420-4126-11eb-19b3-bf6c04aef086
# ╟─85d5a830-4116-11eb-1479-630b53b50e94
# ╠═4862c2d0-4116-11eb-3f26-1bdefcb9ce3d
# ╟─360ab320-411d-11eb-2d11-c9fe7b3137d1
# ╠═73936e9e-4116-11eb-0594-1d202bd073cd
# ╟─9accd320-4117-11eb-0fc4-15e8e85040a8
# ╟─8b43a9a0-4118-11eb-127d-d57dcea189e1
# ╠═73c57b20-4116-11eb-3dba-01c9f7b91fa7
# ╠═5cddc6a0-411c-11eb-2380-1b9442938a9b
# ╠═bec93fa0-4119-11eb-026e-0be14d6a5779
# ╠═cc1f379e-4127-11eb-375c-d782bcac175d
# ╠═737b2bb0-4116-11eb-124b-d981032a9638
# ╟─ce4134a0-4124-11eb-2a7c-9761e81b2064
# ╠═73454ea0-4116-11eb-0938-f9a06f3dbaf8
# ╠═548ac220-4124-11eb-0da7-b96eb51b89d9
