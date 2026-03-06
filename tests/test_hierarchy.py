import pytest
from fdc.hierarchy import build_dendrogram


def test_two_cluster_merge():
    """Two initial centers merge into one at the second noise level."""
    import numpy as np

    # cluster_labels is indexed by the center index value, so array must be large enough
    labels0 = np.zeros(21, dtype=int)
    labels0[10] = 0
    labels0[20] = 1

    labels1 = np.zeros(21, dtype=int)
    labels1[10] = 0
    labels1[20] = 0  # 20 maps to cluster 0 whose center is 10

    hierarchy = [
        {"idx_centers": [10, 20], "cluster_labels": labels0},
        {"idx_centers": [10], "cluster_labels": labels1},
    ]
    noise_range = [0.1, 0.5]

    Z = build_dendrogram(hierarchy, noise_range)

    assert len(Z) == 1
    z = Z[0]
    # Both centers are unmerged originals -> indices 0 and 1 in initial_idx_centers
    assert sorted([z[0], z[1]]) == [0, 1]
    assert z[2] == 0.5
    assert z[3] == 2


def test_three_cluster_sequential_merge():
    """Three clusters: A merges into B first, then AB merges into C."""
    # initial centers: [100, 200, 300]
    # Step 1: 100 merges into 200 (cluster_labels maps idx 100 -> cluster 1 whose center is 200)
    # Step 2: 200 merges into 300

    # cluster_labels needs enough elements so that indexing works.
    # idx 100 needs cluster_labels[100], idx 200 needs cluster_labels[200], etc.
    # We'll use arrays large enough.
    import numpy as np

    labels_step0 = np.zeros(301, dtype=int)
    labels_step0[100] = 0
    labels_step0[200] = 1
    labels_step0[300] = 2

    # Step 1: 100 merges into 200. Centers become [200, 300].
    labels_step1 = np.zeros(301, dtype=int)
    labels_step1[200] = 0
    labels_step1[300] = 1
    # 100 is no longer a center; its label maps to center index 0 -> center 200
    labels_step1[100] = 0

    # Step 2: 200 merges into 300. Centers become [300].
    labels_step2 = np.zeros(301, dtype=int)
    labels_step2[300] = 0
    labels_step2[200] = 0
    labels_step2[100] = 0

    hierarchy = [
        {"idx_centers": [100, 200, 300], "cluster_labels": labels_step0},
        {"idx_centers": [200, 300], "cluster_labels": labels_step1},
        {"idx_centers": [300], "cluster_labels": labels_step2},
    ]
    noise_range = [0.1, 0.3, 0.7]

    Z = build_dendrogram(hierarchy, noise_range)

    assert len(Z) == 2

    # First merge: 100 (index 0) and 200 (index 1), both unmerged originals
    z0 = Z[0]
    assert sorted([z0[0], z0[1]]) == [0, 1]
    assert z0[2] == 0.3
    assert z0[3] == 2

    # Second merge: the new cluster (index 3 = n_init_centers + 0) merges with 300 (index 2)
    z1 = Z[1]
    assert sorted([z1[0], z1[1]]) == [2, 3]
    assert z1[2] == 0.7
    assert z1[3] == 3  # 2 members + 1 new = 3


def test_no_merges():
    """If no centers disappear across steps, Z should be empty."""
    hierarchy = [
        {"idx_centers": [5, 10], "cluster_labels": [0, 1]},
        {"idx_centers": [5, 10], "cluster_labels": [0, 1]},
    ]
    noise_range = [0.1, 0.5]

    Z = build_dendrogram(hierarchy, noise_range)

    assert Z == []


def test_single_step_no_change():
    """A hierarchy with only one level has no transitions, so Z is empty."""
    hierarchy = [
        {"idx_centers": [1, 2, 3], "cluster_labels": [0, 1, 2]},
    ]
    noise_range = [0.1]

    Z = build_dendrogram(hierarchy, noise_range)

    assert Z == []


def test_linkage_matrix_shape():
    """With N initial centers all merging sequentially, Z has N-1 rows of 4 columns."""
    import numpy as np

    n_centers = 5
    centers = list(range(n_centers))

    # Build a hierarchy where one center merges per step
    hierarchy = []
    remaining = list(centers)

    for step in range(n_centers):
        labels = np.zeros(n_centers, dtype=int)
        for i, c in enumerate(remaining):
            labels[c] = i
        hierarchy.append({"idx_centers": list(remaining), "cluster_labels": labels})
        if len(remaining) > 1:
            remaining = remaining[1:]  # remove first center, it merges into second

    noise_range = [0.1 * (i + 1) for i in range(n_centers)]

    Z = build_dendrogram(hierarchy, noise_range)

    assert len(Z) == n_centers - 1
    for z in Z:
        assert len(z) == 4


def test_member_counts_accumulate():
    """When two already-merged clusters merge, member counts sum correctly."""
    import numpy as np

    # 4 centers: [0, 1, 2, 3]
    # Step 1: 0 merges into 1, and 2 merges into 3 (two independent merges)
    # Step 2: 1 merges into 3 (two groups of 2 merge)

    labels0 = np.array([0, 1, 2, 3])
    labels1 = np.array([0, 0, 1, 1])  # 0->cluster 0 (center 1), 2->cluster 1 (center 3)
    labels2 = np.array([0, 0, 0, 0])  # everything -> cluster 0 (center 3)

    hierarchy = [
        {"idx_centers": [0, 1, 2, 3], "cluster_labels": labels0},
        {"idx_centers": [1, 3], "cluster_labels": labels1},
        {"idx_centers": [3], "cluster_labels": labels2},
    ]
    noise_range = [0.1, 0.4, 0.8]

    Z = build_dendrogram(hierarchy, noise_range)

    assert len(Z) == 3

    # Two merges at step 1 (order depends on iteration over pre_idx_centers)
    step1_merges = [z for z in Z if z[2] == 0.4]
    assert len(step1_merges) == 2
    for z in step1_merges:
        assert z[3] == 2  # each merge combines 2 unmerged originals

    # One merge at step 2: two groups of 2 -> 4 members
    step2_merges = [z for z in Z if z[2] == 0.8]
    assert len(step2_merges) == 1
    assert step2_merges[0][3] == 4


def test_noise_range_values_propagated():
    """The distance column (z[2]) should match the noise_range at the merge step."""
    hierarchy = [
        {"idx_centers": [0, 1], "cluster_labels": [0, 1]},
        {"idx_centers": [1], "cluster_labels": [0, 0]},
    ]
    noise_range = [0.0, 42.5]

    Z = build_dendrogram(hierarchy, noise_range)

    assert len(Z) == 1
    assert Z[0][2] == 42.5
