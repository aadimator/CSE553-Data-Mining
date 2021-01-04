### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 8d2a2140-4d6b-11eb-1ed0-5dfeba2603af
begin
	using Pkg
    Pkg.activate(pwd())
end

# ╔═╡ b9452d10-4d6b-11eb-1074-83294cc14eec
begin
	using CSV
	using GraphIO
	using PlutoUI
	using GraphPlot
	using DataFrames
	using LightGraphs
	using SimpleWeightedGraphs
end

# ╔═╡ a4778db2-4d6b-11eb-0e97-c368743a47ea
md"""
# Data Mining
## Semester Project - Clustering by Density
### Aadam - CS1945
"""

# ╔═╡ b046df70-4e5a-11eb-350f-d1abb711d409
md"""
## Utility Functions
"""

# ╔═╡ 822e6af0-4d73-11eb-32e2-216f0814a3c0
# Convert string keys into integer mappings
function str_int(array)
	d = Dict{String, Int}()
	for (i, s) in enumerate(array)
		push!(d, s => i)
	end
	d
end

# ╔═╡ f0bd31ee-4d77-11eb-2834-85e4981d8680
# Return key of dictionary containing value 'i'
function get_key(i, d)
	for (k,v) in d 
		if v==i
			return k
		end
	end
end

# ╔═╡ f60d9610-4d75-11eb-002e-6f23a07f66ad
# Return a cleaned DataFrame from a CSV file
function get_df(path)
	df = CSV.read(path, DataFrame, datarow=1)
	d = str_int(union(df.Column1, df.Column2))
	df.sources = [d[i] for i in df.Column1]
	df.destinations = [d[i] for i in df.Column2]
	rename!(df, :Column3 => :weights)
	return df, d
end

# ╔═╡ 097a9b52-4e37-11eb-2afc-0102d9e2c69d
# Neighbors of a cluster are the nodes connected to any node of cluster but 
# not part of the cluster.
function get_neighbors(g, c, excluded)
	neighbors_set = Set()
	
	for n in c
		union!(neighbors_set, neighbors(g, n))
	end
	
	return collect(setdiff(setdiff(neighbors_set, c), excluded))
end

# ╔═╡ 30c41780-4e5b-11eb-0508-3b2e824a0607
md"""
The priority is determined based on two measures
1. the sum of the weights of the edges between a neighbor and the cluster,
2. the number of edges between a neighbor and the cluster.
"""

# ╔═╡ d4c26d30-4e3f-11eb-07d2-2d722ea4c878
function count_edges(g, cluster, n)
	count = 0
	for c in cluster
		count += has_edge(g, c, n)
	end
	return count
end

# ╔═╡ 95d86970-4e45-11eb-0929-692e7fab4639
# Remove nodes from the graph that are already in the cluster
function rem_nodes(g, cluster)
	for c in cluster
		rem_vertex!(g, c)
	end
	g
end

# ╔═╡ 7b666270-4e5b-11eb-1f65-b5eafbc71680
md"""
**Density** is given as:

$d = \frac{2 \times \left|E\right|}{\left|N\right| \times (\left|N\right| - 1)}$
"""

# ╔═╡ 3df86fd0-4e3f-11eb-3a9c-1b73eabf5568
function density(N, E)
	return (2 * E)/(N * (N - 1))
end

# ╔═╡ c063ab80-4e5b-11eb-283f-6749471fae2d
md"""
To determine whether node is part of periphery we use **cluster property** `cp` of node. It is given as:

$cp = \frac{\left|E_c\right|}{d \times \left|N_c\right|}$
"""

# ╔═╡ ab04f4e0-4e49-11eb-19be-49390f4285e3
function periphery(N, E, d)
	return E/(d * N)
end

# ╔═╡ eba1e8b0-4e57-11eb-1cee-252cd18c350c
md"""
## Algorithm
- Start with a single node as a cluster
- Grow cluster by adding nodes from neighbors one by one
- Continue expanding cluster as long as following 2 conditions are satisfied
  - Density of cluster > = d’
  - Node is in periphery of cluster
- Remove Cluster from graph
- Apply same procedure to rest of nodes to find other clusters in graph.

---

We will follow the algorithm as shown below:
$(LocalResource("algo.png"))
"""

# ╔═╡ ba80451e-4d6b-11eb-0949-6de777cd3780
filename = joinpath("data", "PPI-I.txt")

# ╔═╡ 5655df00-4d76-11eb-01d6-9d21ce3aa823
df, d = get_df(filename)

# ╔═╡ baf68690-4d6b-11eb-3151-b5e91d478309
g = SimpleWeightedGraph(
	Vector(df.sources), 
	Vector(df.destinations), 
	Vector(df.weights)
)

# ╔═╡ ebca8580-4e3a-11eb-2a8b-711f8bc91692
function get_priorities(graph, cluster, neighbors, edge_weights)
	m1 = Vector()
	m2 = Vector()
	
	for n in neighbors
		weight_sum = 0
		edge_count = 0
		
		for c in cluster
			weight_sum += edge_weights[n, c]
			edge_count += has_edge(g, n, c)
		end
		push!(m1, weight_sum)
		push!(m2, edge_count)
	end
	return m1, m2
end

# ╔═╡ 843b4660-4d7d-11eb-387c-6ddb73c11e73
function density_clustering(G, dʼ, cpʼ)
	g = deepcopy(G)
	clusters = Vector()

	# Δ(g) returns highest node degree (LightGraphs)
	while (Δ(g) != 0)
		
		edge_weights = adjacency_matrix(g) * adjacency_matrix(g)
		for i in 1:size(edge_weights)[1]
			edge_weights[i, i] = 0
		end
		node_weights = degree_matrix(g)

		highest_nw = findall(node_weights .== maximum(node_weights))[1]
		starting_node = highest_nw != 0 ? highest_nw[1] : Δ(g)

		cluster = Vector()
		push!(cluster, starting_node)
		cluster_edges = 0
		excluded = Set()

		while true
			cluster_neighbors = get_neighbors(g, cluster, excluded)
			m1, m2 = get_priorities(g, cluster, cluster_neighbors, edge_weights)
			sorted_priorities = sortperm(m1)
			
			if (isempty(sorted_priorities))
				break
			end

			@label addNode
			highest_priority = pop!(sorted_priorities)
			highest_node = cluster_neighbors[highest_priority]
			
			push!(cluster, highest_node)
			cluster_edges += count_edges(g, cluster, highest_node)
			
			d = density(length(cluster), cluster_edges)
			cp = periphery(length(cluster), count_edges(g, cluster, highest_node), d)
			
			println("Density: $d and CP: $cp")
			
			if (d < dʼ && cp < cpʼ)
				push!(excluded, pop!(cluster))
				if (!isempty(sorted_priorities))
					println("Remaining Neighbors: $(length(sorted_priorities))")
					@goto addNode
				else
					println("No neighbors left")
					break
				end
			end
		end

		push!(clusters, cluster)
		println("Added $(length(clusters)) clusters.") 
		g = rem_nodes(g, cluster)
	end
	[c for c in clusters if length(c) > 2]
end

# ╔═╡ 845006e0-4d7d-11eb-1fe7-1d59f31b1ca6
clusters = density_clustering(g, 0.6, 0.6)

# ╔═╡ d9f94720-4e43-11eb-063f-49c9038aa982
# gplot(g)

# ╔═╡ 822fb7b0-4e69-11eb-3cd4-07f9188ca131
md"""
### Clusters
"""

# ╔═╡ 2710db30-4e6d-11eb-2441-b7d19fe74697
with_terminal() do
    for (i, c) in enumerate(clusters)
		println("Cluster $i: $(convert.(Int, c))")
		println()
	end
end

# ╔═╡ e1bb91a0-4e6d-11eb-1214-09f5dcf2e4a2
md"""
#### Save to File
"""

# ╔═╡ 7f986070-4e68-11eb-16a3-93a29ab63135
output_file = joinpath("output", "clusters.txt")

# ╔═╡ da256032-4e43-11eb-3690-c186bee53f59
open(output_file, "w") do io
	for c in clusters
		println(io, convert.(Int, c))
		println(io)
	end
end

# ╔═╡ 409381b0-4e6e-11eb-1654-21a675b7eb7e
md"""
Save clusters with their original node labels.
"""

# ╔═╡ f1324f20-4e6d-11eb-21fc-e3461700c517
output_file_names = joinpath("output", "named_clusters.txt")

# ╔═╡ 06fcec20-4e6e-11eb-3a44-f754f81f3290
open(output_file_names, "w") do io
	for c in clusters
		println(io, [get_key(i, d) for i in c])
		println(io)
	end
end

# ╔═╡ Cell order:
# ╟─8d2a2140-4d6b-11eb-1ed0-5dfeba2603af
# ╟─a4778db2-4d6b-11eb-0e97-c368743a47ea
# ╠═b9452d10-4d6b-11eb-1074-83294cc14eec
# ╟─b046df70-4e5a-11eb-350f-d1abb711d409
# ╠═822e6af0-4d73-11eb-32e2-216f0814a3c0
# ╠═f0bd31ee-4d77-11eb-2834-85e4981d8680
# ╠═f60d9610-4d75-11eb-002e-6f23a07f66ad
# ╠═097a9b52-4e37-11eb-2afc-0102d9e2c69d
# ╟─30c41780-4e5b-11eb-0508-3b2e824a0607
# ╠═ebca8580-4e3a-11eb-2a8b-711f8bc91692
# ╠═d4c26d30-4e3f-11eb-07d2-2d722ea4c878
# ╠═95d86970-4e45-11eb-0929-692e7fab4639
# ╟─7b666270-4e5b-11eb-1f65-b5eafbc71680
# ╠═3df86fd0-4e3f-11eb-3a9c-1b73eabf5568
# ╟─c063ab80-4e5b-11eb-283f-6749471fae2d
# ╠═ab04f4e0-4e49-11eb-19be-49390f4285e3
# ╟─eba1e8b0-4e57-11eb-1cee-252cd18c350c
# ╠═843b4660-4d7d-11eb-387c-6ddb73c11e73
# ╠═ba80451e-4d6b-11eb-0949-6de777cd3780
# ╠═5655df00-4d76-11eb-01d6-9d21ce3aa823
# ╠═baf68690-4d6b-11eb-3151-b5e91d478309
# ╠═845006e0-4d7d-11eb-1fe7-1d59f31b1ca6
# ╠═d9f94720-4e43-11eb-063f-49c9038aa982
# ╟─822fb7b0-4e69-11eb-3cd4-07f9188ca131
# ╟─2710db30-4e6d-11eb-2441-b7d19fe74697
# ╟─e1bb91a0-4e6d-11eb-1214-09f5dcf2e4a2
# ╠═7f986070-4e68-11eb-16a3-93a29ab63135
# ╠═da256032-4e43-11eb-3690-c186bee53f59
# ╟─409381b0-4e6e-11eb-1654-21a675b7eb7e
# ╠═f1324f20-4e6d-11eb-21fc-e3461700c517
# ╠═06fcec20-4e6e-11eb-3a44-f754f81f3290
