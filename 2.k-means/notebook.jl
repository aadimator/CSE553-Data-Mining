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
	using Distances
end

# ╔═╡ c73d5420-4126-11eb-19b3-bf6c04aef086
md"""
# Data Mining
## Assignment 2 - K-means Clustering
### Aadam & Obaidullah (CS1945 & CS1947)
"""

# ╔═╡ 360ab320-411d-11eb-2d11-c9fe7b3137d1
plotly()

# ╔═╡ 73936e9e-4116-11eb-0594-1d202bd073cd
function Kmeans(X, k; max_iters = 300, tol = 1e-5)
    # Reshape 2D array to a 1D array with length of all training examples
    X_array_list = collect(eachrow(X))

    # Save some info on the incoming data
    N = length(X_array_list)  # Length of all training examples
    n = length(X_array_list[1])  # Length of a single training example
    distances = zeros(N)  # Empty vector for all training examples.

    # Step 1: Random initialization
    reps_centroids = [zeros(n) for grp = 1:k]  # Initiate centroids for each
    labels = rand(1:k, N)  # Randomly assign labels to all training examples

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
            # compute the distance between each example and the updated centroid
            nearest_rep_distance = 
			[norm(X_array_list[i] - reps_centroids[x]) for x = 1:k]

            # update distances and label arrays 
            # findmin returns the min value & index location
            distances[i], labels[i] = findmin(nearest_rep_distance)
        end;

        # Step 4: Compute the clustering cost
        J = (norm(distances)^ 2) / N

        # Show progress and terminate if J stopped decreasing.
        println("Iteration ", iter, ": Jclust = ", J, ".")

        # Final Step 5: Check for convergence
        if iter > 1 && abs(J - J_previous) < (tol * J)
            # Terminate algorithm with the assumption that K-means has converged
            return labels, reps_centroids
        end

        J_previous = J
    end
	return labels, reps_centroids
end

# ╔═╡ aac40900-41cc-11eb-1103-97d9373bdd65
function calculate_average_distance(i, X, result)
	dist = 0
	center = result.centers[i]
	for instance in X[:, result.assignments .== i]
		dist += Euclidean()(instance, center)
	end
	return dist/result.counts[i]
end

# ╔═╡ 12615c80-41c7-11eb-2aed-c1eb96003ff1
function dbi(X, result)
	num_of_clusters = length(result.counts)
	
	#average distance of all points in a cluster to its center, for each cluster
    average_centroid_distances = [
		calculate_average_distance(i, X, result) for i in 1:num_of_clusters]
	
	sum_Dij = 0
	
	#calculate the sum Dij
    for i in 1:num_of_clusters     
        max_Dij = 0
        
        #calculate the max Dij
        for j in 1:num_of_clusters            
            if i != j
				
				#calculate di + dj
				numerator = 
				average_centroid_distances[i] + average_centroid_distances[j]

				#calculate dij
				denominator = 
				Euclidean()( result.centers[i], result.centers[j] )

				current_Dij = numerator/denominator

				if max_Dij < current_Dij
					max_Dij = current_Dij
				end
			end
                
		end
		sum_Dij += max_Dij
	end
    
    return sum_Dij / num_of_clusters
end

# ╔═╡ c7001000-42c6-11eb-23d1-314932816f0a
function big_delta(maskci, distances)
	maximum(distances[maskci, :][:, maskci])
end

# ╔═╡ c6ca5a00-42c6-11eb-386e-2f13a99d75b6
function delta(maskci, maskcj, distances)
	minimum(distances[maskci, :][:, maskcj])
end

# ╔═╡ c69d2f80-42c6-11eb-10cf-e70c07e40e97
function dunn(X, result)
	n = length(result.counts) # number of clusters
	
	distances = pairwise(Euclidean(), X, dims=2)
	
	deltas = ones(n, n) * Inf
	big_deltas = zeros(n)
	
	for k in 1:n
		for l in 1:n
			if k == l
				continue
			end
            deltas[k, l] = delta(
				result.assignments .== k, 
				result.assignments .== l, 
				distances)
		end
        
        big_deltas[k] = big_delta(result.assignments .== k, distances)
		
	end
	minimum(deltas)/maximum(big_deltas)
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
Vlaue of **k**: $(@bind k Slider(2:15; show_value=true))
"""
# size(UCIData.dataset(selected_dataset))[1]

# ╔═╡ 737b2bb0-4116-11eb-124b-d981032a9638
result = kmeans(features, k);

# ╔═╡ 725d2e10-42c2-11eb-3ee7-4ba089ebb440
md"""
### Davies Bouldin Index: $(round(dbi(features, result), digits=5))
### Dunn Index : $(round(dunn(features, result), digits=5))
"""

# ╔═╡ ce4134a0-4124-11eb-2a7c-9761e81b2064
md"""
## Plot Clusters
Feature $(@bind feature1 Select(string.(collect(1:size(features)[1])))) vs.
Feature $(@bind feature2 Select(string.(collect(1:size(features)[1])));)
"""

# ╔═╡ 73454ea0-4116-11eb-0938-f9a06f3dbaf8
begin
	theme(:dark)
	scatter(
		features[parse(Int,feature1), :], 
		features[parse(Int, feature2), :],
		marker_z = result.assignments, 
		markershape = :diamond,
		markersize = 3,
		xlabel="Feature $(feature1)", 
		ylabel="Feature $(feature2)",
		color = :seaborn_bright, 
		legend = false)
	scatter!(
		result.centers[parse(Int,feature1), :], 
		result.centers[parse(Int, feature2), :],
		markersize = 6,
		marker_z = result.centers',
		color = :temperaturemap
	)
end

# ╔═╡ 6cd3fe00-41c4-11eb-2e84-376a14af7398
md"""
### Cluster Centers
$(
df = DataFrame(hcat(result.centers', result.counts));
last_col_name = names(df)[end];
df[!,last_col_name] = convert.(Int,df[:,last_col_name]);
rename!(df, last_col_name => :NodeCount)
)
"""

# ╔═╡ Cell order:
# ╟─c73d5420-4126-11eb-19b3-bf6c04aef086
# ╟─85d5a830-4116-11eb-1479-630b53b50e94
# ╠═4862c2d0-4116-11eb-3f26-1bdefcb9ce3d
# ╟─360ab320-411d-11eb-2d11-c9fe7b3137d1
# ╠═73936e9e-4116-11eb-0594-1d202bd073cd
# ╠═aac40900-41cc-11eb-1103-97d9373bdd65
# ╠═12615c80-41c7-11eb-2aed-c1eb96003ff1
# ╠═c7001000-42c6-11eb-23d1-314932816f0a
# ╠═c6ca5a00-42c6-11eb-386e-2f13a99d75b6
# ╠═c69d2f80-42c6-11eb-10cf-e70c07e40e97
# ╟─9accd320-4117-11eb-0fc4-15e8e85040a8
# ╟─8b43a9a0-4118-11eb-127d-d57dcea189e1
# ╠═73c57b20-4116-11eb-3dba-01c9f7b91fa7
# ╠═5cddc6a0-411c-11eb-2380-1b9442938a9b
# ╟─bec93fa0-4119-11eb-026e-0be14d6a5779
# ╠═737b2bb0-4116-11eb-124b-d981032a9638
# ╠═725d2e10-42c2-11eb-3ee7-4ba089ebb440
# ╟─ce4134a0-4124-11eb-2a7c-9761e81b2064
# ╟─73454ea0-4116-11eb-0938-f9a06f3dbaf8
# ╟─6cd3fe00-41c4-11eb-2e84-376a14af7398
