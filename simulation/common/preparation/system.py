"""Box scaling that preserves the internal geometry of each molecule."""
from collections import deque
from typing import Sequence, Tuple
import numpy as np


def rescale_box_preserving_molecules(
    positions_nm: np.ndarray,
    old_box_nm: Sequence[float],
    new_box_nm: Sequence[float],
    molecule_groups: Sequence[Sequence[int]],
    unwrap_edges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Scale molecule centers while preserving all internal geometry.

    ``unwrap_edges`` may include virtual native-base-pair edges, allowing both
    antiparallel strands of a duplex to remain one rigid registry group during
    the six exact-box corrections.
    """

    positions = np.asarray(positions_nm, dtype=np.float64)
    old_box = np.asarray(old_box_nm, dtype=np.float64)
    new_box = np.asarray(new_box_nm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_nm must be shape (n,3)")
    if old_box.shape != (3,) or new_box.shape != (3,) or np.any(old_box <= 0) or np.any(new_box <= 0):
        raise ValueError("old/new boxes must contain three positive lengths")
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(positions))}
    for first, second in unwrap_edges:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    output = np.array(positions, copy=True)
    seen_global: set[int] = set()
    for raw_group in molecule_groups:
        group = tuple(int(index) for index in raw_group)
        if not group:
            raise ValueError("molecule group may not be empty")
        if any(index < 0 or index >= len(positions) for index in group):
            raise IndexError("molecule group particle out of range")
        if seen_global.intersection(group):
            raise ValueError("molecule groups overlap")
        seen_global.update(group)
        allowed = set(group)
        unwrapped: dict[int, np.ndarray] = {}
        # Handle a disconnected group deterministically, although production
        # duplex groups should be connected by virtual native-pair edges.
        for root in group:
            if root in unwrapped:
                continue
            if not unwrapped:
                unwrapped[root] = np.array(positions[root], copy=True)
            else:
                reference = np.mean(np.asarray(list(unwrapped.values())), axis=0)
                delta = positions[root] - np.mod(reference, old_box)
                delta -= old_box * np.rint(delta / old_box)
                unwrapped[root] = reference + delta
            queue = deque([root])
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if neighbor not in allowed or neighbor in unwrapped:
                        continue
                    delta = positions[neighbor] - positions[current]
                    delta -= old_box * np.rint(delta / old_box)
                    unwrapped[neighbor] = unwrapped[current] + delta
                    queue.append(neighbor)
        group_unwrapped = np.asarray([unwrapped[index] for index in group])
        center = np.mean(group_unwrapped, axis=0)
        fractional_center = np.mod(center, old_box) / old_box
        new_center = fractional_center * new_box
        shifted = group_unwrapped - center + new_center
        output[np.asarray(group, dtype=int)] = np.mod(shifted, new_box)
        # Pair distances along every unwrap edge in this group must be exact.
        for first in group:
            for second in adjacency[first]:
                if second not in allowed or first >= second:
                    continue
                before = positions[second] - positions[first]
                before -= old_box * np.rint(before / old_box)
                after = output[second] - output[first]
                after -= new_box * np.rint(after / new_box)
                if not np.allclose(before, after, rtol=0.0, atol=1e-10):
                    raise RuntimeError("molecule-preserving box scale changed an internal edge")
    if seen_global != set(range(len(positions))):
        raise ValueError("molecule groups must partition every particle")
    return output

