import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.spatial import ConvexHull
from scipy.special import softmax
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


def is_point_in_hull(point, hull, tol=1e-12):
    return np.all(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= tol)

def local_maximal_convex_position(points, n, k):
    N, d = points.shape
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    all_hulls = []
    
    with tqdm(total=n, desc="Local Maximal Convex Position Constructing") as pbar:
        while len(all_hulls) < n:
            anchor = random.randint(0, N - 1)
            _, cand = nbrs.kneighbors(points[anchor].reshape(1, -1))
            cand = list(cand[0])
            cand.remove(anchor)
            
            base = random.sample(cand, d)
            base = [anchor] + base
            
            hull_base = ConvexHull(points[base], qhull_options="QJ")
            
            # Check if hull_base use all base point as vertices
            # And no other point inside hull
            cand = list(set(cand) - set(base))
            others = points[cand]
            if (len(hull_base.vertices) == len(base)) and (all(not is_point_in_hull(p, hull_base) for p in others)):
                for c in cand:
                    new = base + [c]
                    hull_new = ConvexHull(points[new], qhull_options="QJ")

                    others = points[list(set(cand) - set(new))]
                    if (len(hull_new.vertices) == len(new)) and (all(not is_point_in_hull(p, hull_new) for p in others)):
                        hull_base = hull_new
                        base = new
                        
                if base not in all_hulls:
                    all_hulls.append(sorted(base))
                    pbar.update(1)
    return all_hulls

def computeTransitionMatrix(hulls):
    n = len(hulls)
    adjMatrix = np.zeros((n,n))
    for i in tqdm(range(n), desc="Adjacency Matrix Constructing"):
        source = set(hulls[i])
        for j in range(i):
            target = set(hulls[j])
            intersection_set = source.intersection(target)
            if len(intersection_set)>0:
                adjMatrix[i,j] = 1
                adjMatrix[j,i] = 1
            else:
                adjMatrix[i,j] = np.inf
                adjMatrix[j,i] = np.inf
    print("Transition Matrix Constructing . . .", end = " ")
    transitionMatrix = floyd_warshall(csgraph=csr_matrix(adjMatrix), directed=False)
    print("Completed")
    return transitionMatrix

def randomByPrevioushull(transMatrix, PrevioushullIdx):
    n,_ = transMatrix.shape
    hIdx = list(range(0,n))
    hIdx.pop(PrevioushullIdx)
    p = transMatrix[PrevioushullIdx]
    p = np.delete(p,PrevioushullIdx)
    p = 1/p
    p[np.isposinf(p)] = 1e300
    p = softmax(p)
    sampleIdx = np.random.choice(hIdx,1,p=p)
    return sampleIdx

def random_point_in_convex_hull(points):
    N, d = points.shape
    weights = np.random.dirichlet(np.ones(N))
    random_point = np.dot(weights, points)
    return random_point
    
def _run_mcmc_heuristic(
    phase_name,
    total_required,
    n_hulls,
    trans_matrix,
    data_points,
    hulls,
    kde,
    current_state,
):
    points, densities, decisions = [], [], []
    proposals = 0
    accepted_count = 0

    current_hull_idx, current_point, d_current = current_state
    pbar = tqdm(total=total_required, desc=f"MCMC Phase: {phase_name}")
    while accepted_count < total_required:
        proposals += 1

        if phase_name == "burn_in" and proposals == 1:
            random_hull_idx = np.random.randint(n_hulls, size=1)[0]
            proposed_point = random_point_in_convex_hull(
                data_points[hulls[random_hull_idx], :]
            )
            d_proposed = kde.pdf(proposed_point)
            accept = True
        else:
            random_hull_idx = randomByPrevioushull(
                trans_matrix, current_hull_idx
            )[0]
            proposed_point = random_point_in_convex_hull(
                data_points[hulls[random_hull_idx], :]
            )
            d_proposed = kde.pdf(proposed_point)
            accept_prob = min(1, (d_proposed + 1e-999) / (d_current + 1e-999))
            accept = np.random.rand() <= accept_prob

        if accept:
            current_hull_idx = random_hull_idx
            current_point = proposed_point
            d_current = d_proposed

            densities.append(d_current)
            if phase_name == "synthesis":
                points.append(current_point)

            accepted_count += 1
            decisions.append(
                {"phase": phase_name, "proposal_num": proposals, "accepted": 1}
            )
        else:
            decisions.append(
                {"phase": phase_name, "proposal_num": proposals, "accepted": 0}
            )

        pbar.update(accepted_count - pbar.n)
    pbar.close()

    # Update state
    current_state[0], current_state[1], current_state[2] = (
        current_hull_idx,
        current_point,
        d_current,
    )
    return points, densities, decisions, proposals

def CHDS_synthesizer(input_df : pd.DataFrame,
                     num_records_to_generate : int,
                     numerical_cols : list[str],
                     k_neighbor : int = 15,
                     n_hull : int = 7500,
                     burn_in : int = 2000):
    # 1. Data preprocessing
    categorical_cols = [col for col in input_df.columns if col not in numerical_cols]
    print("#Samples:", input_df.shape)
    print("#Numerical features:", len(numerical_cols))
    print("#Categorical features:", len(categorical_cols))

    encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(input_df[numerical_cols])
    encoded_categorical = encoder.fit_transform(input_df[categorical_cols])
    processed_data = np.hstack((scaled_continuous, encoded_categorical))
    print("#Dimension after preprocessing:", processed_data.shape)
    
    data_points = processed_data
    N, d = data_points.shape
    
    # 2. Geometric structure construction
    k_neighbor = max(d+3, k_neighbor)
    hulls = local_maximal_convex_position(
        data_points, k=k_neighbor, n=n_hull
    )
    n_hulls = len(hulls)
    print(f"{n_hulls} convex positions constructed")
    
    # 3. Data density estimation
    var_type = ("c" * scaled_continuous.shape[1]) + ("u" * encoded_categorical.shape[1])
    kde_temp = KDEMultivariate(
        data=data_points,
        var_type=var_type,
        bw = 'normal_reference',
    )
    
    kde_temp_bw = kde_temp.bw # https://github.com/statsmodels/statsmodels/blob/main/statsmodels/nonparametric/bandwidths.py
    bw = []
    for i, v in enumerate(var_type):
        bw_res = kde_temp_bw[i]
        if v == "u":
            bw_res = np.clip(bw_res, 0.5, 1)
        bw.append(bw_res)
    kde = KDEMultivariate(data=data_points, var_type=var_type, bw=bw)
    
    # 4. Data sampling using MCMC-Based Heuristic
    trans_matrix = computeTransitionMatrix(hulls)
    
    # Initial state [hull_idx, point, density]
    mcmc_state = [0, None, 0.0]
    # Run Burn-in Phase
    _, burn_densities, burn_decisions, burn_props = _run_mcmc_heuristic(
        "burn_in",
        burn_in,
        n_hulls,
        trans_matrix,
        data_points,
        hulls,
        kde,
        mcmc_state,
    )

    # Run Synthesis Phase
    synth_points, synth_densities, synth_decisions, synth_props = (
        _run_mcmc_heuristic(
            "synthesis",
            num_records_to_generate,
            n_hulls,
            trans_matrix,
            data_points,
            hulls,
            kde,
            mcmc_state,
        )
    )

    burn_in_acc_rate = burn_in / burn_props
    synth_acc_rate = num_records_to_generate / synth_props
    print(f" -> Burn-in Acceptance Rate  : {burn_in_acc_rate:.2%}")
    print(f" -> Synthesis Acceptance Rate : {synth_acc_rate:.2%}")
    
    # 5. Postprocessing
    synth_points = np.array(synth_points)
    synth_numerical = synth_points[:, :scaled_continuous.shape[1]]
    inversed_continuous = scaler.inverse_transform(synth_numerical)
    df_numerical = pd.DataFrame(inversed_continuous, columns=numerical_cols)
    
    synth_categorical = synth_points[:, scaled_continuous.shape[1]:]
    inversed_categorical = encoder.inverse_transform(synth_categorical)
    df_categorical = pd.DataFrame(inversed_categorical, columns=categorical_cols)
    
    synthetic_data = pd.concat([df_numerical, df_categorical], axis=1)
    synthetic_data = synthetic_data[input_df.columns]
    synthetic_data = synthetic_data.astype(input_df.dtypes.to_dict())
    return synthetic_data