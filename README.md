# M³GLVQ – Multi-Matrix Median Generalized Learning Vector Quantization

This repository implements M³GLVQ (Multi-Matrix Median Generalized Learning Vector Quantization), a prototype-based machine learning method that operates directly on multiple heterogeneous proximity matrices rather than fixed feature vectors.

It is particularly suited for applications such as industrial customer profiling, where entities (e.g., companies) are naturally described through diverse, non-Euclidean attributes:
- Industry classification (NAICS)
- Traded products (HS codes)
- Application/material standards (DIN/ISO)
- Geographic location
- Company size (e.g., employees, revenue)

These attributes:
- lie in incompatible domains (categorical, sets, coordinates, numbers)
- often have hierarchical structure (e.g., NAICS, HS)
- are not easily embedded into a common vector space

M³GLVQ avoids explicit vector embeddings and instead learns directly from pairwise dissimilarities.

Proximity functions here are **dissimilarity measures** \(D(i, j) \in [0, 1]\) that quantify how different two entities \(i\) and \(j\) are in a specific aspect. They are built for:

- **categorical / hierarchical codes**, and  
- **numeric / geospatial attributes**.

Each dissimilarity matrix corresponds to a specific attribute group. Below we define the structure and metric for each.

## D1 — Industry (NAICS Code Distance)

### Feature Structure
NAICS codes are 6-digit hierarchical industry identifiers. Early digits represent broad sectors; later digits add detail. A suitable metric must respect digitwise categorical comparison and the hierarchy depth.

### Metric: Hamming / Weighted Hamming Distance

**1. Standard Hamming Distance (categorical string distance)**

Measures the proportion of positions in which the two codes differ [3]:

$$
D_H(i,j) = 
\frac{1}{L} \sum_{k=1}^{L} \mathbf{1}[x_{ik} \neq x_{jk}]
$$

This treats all positions equally and ignores hierarchy — useful as a baseline.

**2. Weighted Hamming Distance (hierarchy-aware)**

Earlier digits (higher hierarchy) receive larger weights:

$$
D_{wH}(i,j)=
\frac{\sum_{k=1}^{L} w_k \, \mathbf{1}[x_{ik} \neq x_{jk}]}
     {\sum_{k=1}^{L} w_k}
$$

Linear hierarchical weights:

$$
w_k = L - k + 1
$$

This emphasizes mismatches at high semantic levels (e.g., sector) more strongly than mismatches at detailed levels.

## D2 — Produced Products (HS Code Set Distance)

### Feature Structure
Each entity is represented by a *set* of HS product codes. Since multiple codes can apply, similarity must be measured through set overlap.

### Metric: Weighted Dice Distance
Dice distance measures set overlap relative to average set size. A weighted version incorporates informativeness via IDF weighting [1]. Let \(A_i\) and \(A_j\) be the HS code sets of entities \(i\) and \(j\).

**Unweighted Dice:**

$$
D_D(i,j)
= 1 - \frac{2\lvert A_i \cap A_j \rvert}
              {\lvert A_i \rvert + \lvert A_j \rvert} $$ 


**IDF weight for code \(k\):**

$$
w_k = \log\left(\frac{N}{f(k)}\right)
$$

where \(N\) is the total number of entities and \(f(k)\) is the document frequency of code \(k\).

**Weighted Dice:**

$$ D_{D,w}(i,j) = 1 - \frac{2 \sum_{k \in A_i \cap A_j} w_k} {\sum_{k \in A_i} w_k + \sum_{k \in A_j} w_k}$$


This downweights frequent HS codes and highlights rare, more informative codes.

## D3 — Material / Application (DIN/ISO Code Sets)

### Feature Structure
Entities may list multiple DIN/ISO material or application codes. These behave as unordered categorical tokens; therefore set-based comparison is required.

### Metric: Weighted Dice Distance
Same principle as HS codes: measure overlap, adjust for informativeness.

**Unweighted Dice:**

$$
D_D(i,j)
= 1 - \frac{2|S_i \cap S_j|}
              {|S_i| + |S_j|}
$$

**IDF weight:**

$$
w_k = \log\left(\frac{N}{f(k)}\right)
$$

**Weighted Dice:**

$$
D_{D,w}(i,j) = 1 - \frac{2 \sum_k \min(w_{ik}, w_{jk})}     {\sum_k w_{ik} + \sum_k w_{jk}}
$$

Weighted Dice works well for multi-label technical standards where rare codes carry more discriminative power.


## D4 — Geographic Distance (Coordinates + Adjacency Adjustment)

### Feature Structure
Each entity has:
- geographic coordinates (latitude, longitude)
- country, region, subregion

The metric combines physical geodesic distance with geopolitical adjacency.

### Metric: Haversine Distance + Categorical Adjustment

**1. Haversine great-circle distance [2]:**

$$ d_{ij}=
2R\,
\arcsin\left(
\sqrt{
\sin^2\left(\frac{\Delta\phi}{2}\right) +
\cos\phi_i \cos\phi_j \sin^2\left(\frac{\Delta\lambda}{2}\right)
}
\right),
\qquad R=6371\ \text{km} $$


**2. Linear normalization:**

$$ d^{\text{lin}}_{ij} = \min\left(1,\; \frac{d_{ij}}{\kappa_{\text{km}}}\right)$$


**3. Country/region adjacency distance:**

$$
d^{\text{cat}}_{ij} =
\begin{cases}
0 & \text{same country} \\
v & \text{neighboring countries} \\
1 & \text{different continents} \\
v + (1-v)(1-\text{region similarity}) & \text{same continent}
\end{cases} $$


**4. Combined distance:**

$$ D^{\text{GEO}}_{ij} = \alpha\, d^{\text{lin}}_{ij} + (1-\alpha)\, d^{\text{cat}}_{ij} $$

This balances real physical distance and region-based proximity.

---

References

[1] Fried, Z. (2019). Application of a weighted Dice Similarity Coefficient (DSC) for structural comparison. Oregon State University.

[2] Gade, K. (1984). Virtues of the Haversine. Sky & Telescope, 68(2), 159.

[3] Hamming, R. W. (1950). Error detecting and error correcting codes. The Bell System Technical Journal, 29(2), 147–160.

[4] Levy, A. et al. (2025). A guide to similarity measures and their data science applications. Journal of Big Data.
