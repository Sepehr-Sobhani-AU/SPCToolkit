## ISSUES:

- ~~We need to show branch type in tree structure~~
  - **Report (Fixed):** Added a **Type** column to the tree (`Branch | Type | Cache`). `TreeStructureWidget.add_branch` resolves each node's `data_type` from the `DataNodes` singleton via `_resolve_branch_type`, so all call sites populate it automatically. Cache column moved from index 1 → 2.
  - The branch type is shown now. The **Value** node types should show **Value ([type])**. I can't remember exact format. *(some info should be saved in Tag)*
  - ~~**↳ Done:**~~ `values` branches now render as **`values (<source>)`** in the Type column, where `<source>` is the producing operation read from the node's `tags[0]` (e.g. `values (average_distance)`, `values (knn_analysis)`). No plugin changes needed — the source is already stored in tags. Format is trivial to adjust if you want a different label.

- ~~For some reason subtracting ground branch from estimated normal branch is extremely slow. If I'm not wrong It should be a mask array applying on the estimate normal branch.~~
  - **Report (Round 1):** Added `_lineage_keep_mask` fast path (compose boolean masks instead of coordinate matching).
  - *NOT FIXED YET (round 1).* → **Root cause found & fixed (round 2):** the fast path was bailing to the slow coordinate match whenever the ground lineage crossed a **`cluster_labels`** node (dbscan / surface_region_growing). `ClustersTransformer` keeps the parent's points unchanged (it only *attaches* labels), so `cluster_labels` is actually an **identity** step — it's now classified as such. Also added **`class_reference`** support (selects via `isin(labels, cluster_ids)` using the nearest cluster_labels ancestor). Verified exact on the realistic dbscan→region-growing→separate-clusters lineage and the class-select lineage; all unsafe cases still fall back. Subtract should now be instant for these.
  - ~~***OPEN (new, separate task):*** Change the background color of branches whose datanode data length ≠ their root ancestor's point count.~~ **Done:** rows whose data-array length ≠ root point count are highlighted **soft amber** across the whole row (`_set_row_background`). Uses new `ApplicationController.get_node_data_length` (cache-free) vs `get_root_point_count`. `class_reference`/`container` (no per-point array) are never falsely flagged.

- ~~We need to add a column showing the number of points for each branch.~~
  - ~~**Report (Open):**~~ **Done:** added a **Points** column (`Branch | Type | Points | Cache`; Cache moved 2 → 3). Shows each node's **data-array length** with thousands separators (cache-free, O(1) — no reconstruction). Note this is the *length* the data spans, not the selected count: a root-aligned mask shows the full root count (and stays un-highlighted), while a shorter branch shows its smaller length and gets the amber highlight — together visualizing the item-④ invariant. Tooltip shows the count and, on mismatch, the root count too.

- I expect the length of data nodes in all the branches under the root point cloud be exactly the same as number of points in the root. This way applying logical operations on the branches will be very fast.
  - **Report (Open / design):** Invariant does **not** hold today (masks are boolean over their *immediate parent*; nested masks / point_cloud-subsets are shorter).
  - ***Do you have any suggestion?***
  - **↳ Suggestion:** Move to **root-anchored, bit-packed boolean masks** — every selection branch stores a boolean mask of length = **root** point count; nested filters AND into the parent's root-space mask. Wins all at once: logical ops (AND/OR/NOT/subtract) become native O(n) array ops; mask memory drops to 1 bit/point (≈9 MB for 73M — the issue you raised last round); the length invariant holds, so the color cue only flags genuine new-root branches (merge / duplicate-to-root). Cost: a data-model change (mask producers must emit root-space masks, or a transformer lifts immediate-parent masks once). This is Definition-level — wants a `DECISIONS.md` entry before building. Recommend doing the color cue + points column first, then schedule this as its own change.

- The menus and their contents should be rearranged, as many of the plugins are not in a proper location.
  - **Report (Open):** Organizational; needs a deliberate menu taxonomy pass (Definition-level).
  - ***OK, leave it for later***

- ~~The "Cut cluster" plugin should be renamed to "split clusters" (including .py file)~~
  - ~~**Report (Open):**~~ **Done:** file `050_cut_cluster_plugin.py` → `050_split_clusters_plugin.py` (git mv, history preserved); class `CutClusterPlugin` → `SplitClustersPlugin`; `get_name()` `"cut_cluster"` → `"split_clusters"` (menu now shows **"Split Clusters"**); keybinding `Key_C` in `_key_input.py` updated to `"split_clusters"`. No `cut_cluster` references remain. Note: distinct from existing "Split Classes" plugin.
