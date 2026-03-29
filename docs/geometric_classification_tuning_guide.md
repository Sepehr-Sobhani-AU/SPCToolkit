# Threshold Tuning Guide for Geometric Classification

## The Key Math: Features Sum to 1

The three geometric features are defined as ratios over λ₃ (largest eigenvalue):

```
planarity  = (λ₂ - λ₁) / λ₃
linearity  = (λ₃ - λ₂) / λ₃
sphericity = λ₁ / λ₃
```

**They always sum to exactly 1.0:**

```
planarity + linearity + sphericity = (λ₂ - λ₁ + λ₃ - λ₂ + λ₁) / λ₃ = λ₃ / λ₃ = 1.0
```

This means a point's geometry is a position in a **ternary diagram** — it can't score high on all three simultaneously. If linearity is 0.6, then planarity + sphericity can only be 0.4.

## Where Cylinders Live in the Ternary Diagram

Cylinder surfaces have the eigenvalue signature λ₃ > λ₂ >> λ₁ — spread in two directions (along the axis and around the circumference) but thin in the normal direction. This gives them:
- **Moderate planarity** (λ₂ > λ₁, but not as extreme as a flat plane)
- **Moderate linearity** (λ₃ > λ₂, but not as extreme as a thin pole)
- **Low sphericity** (λ₁ is small relative to λ₃)

They sit on the **planarity-linearity edge** of the ternary diagram — exactly the mixed zone between pure planar and pure linear. The Cylindrical class captures this diagonal band using `min(linearity, planarity) > threshold`.

## What the Eigenvalue Visualization Shows You

The RGB visualization maps these features to color channels:
- **Red** = planarity (walls, ground, roofs — flat surfaces)
- **Green** = linearity (cables, edges, poles — elongated features)
- **Blue** = sphericity × 3 (clumped/isotropic points — vegetation, noise)

The `× 3` scaling on blue means sphericity gets visually amplified. In the raw features, a perfectly spherical point (λ₁=λ₂=λ₃) has sphericity = 0.33 (since all three features = 0.33). The `× 3` makes this appear as bright blue.

**How to read the colors in the viewer:**

| Color | Meaning | Typical objects |
|-------|---------|-----------------|
| Bright red | Strongly planar | Walls, ground, roofs |
| Bright green | Strongly linear | Power lines, thin poles, edges |
| **Yellow (R+G)** | **Mix of planar + linear** | **Cylinder surfaces, tree trunks, pipes** |
| Bright blue | Strongly spherical/isotropic | Dense vegetation, scatter |
| Cyan (G+B) | Mix of linear + spherical | Thin branches, sparse linear |
| Magenta (R+B) | Mix of planar + spherical | Rough flat surfaces |
| White/gray | All channels similar | Transition geometry |
| Dark | Low total variance | Sparse/isolated points |

**Key:** Yellow/orange-ish points in the eigenvalue view are your cylinder candidates — they have both red (planarity) and green (linearity) channels active, which is exactly what the Cylindrical class captures.

## Decision Tree and Cascade Order

The classification checks in this order — **first match wins**:

```
1. SPARSE:       total_variance outlier (bottom/top N%)              → Gray
2. CYLINDRICAL:  min(linearity, planarity) > cylindrical_threshold   → Cyan
3. LINEAR:       linearity > linear_threshold                        → Green
4. PLANAR:       planarity > planar_threshold                        → Red
5. SCATTER:      sphericity > sphericity_threshold                   → Blue
6. TRANSITION:   everything else                                     → Amber
```

**Cascade implications:**
- Sparse is checked first, so noisy/isolated points are removed before feature-based classification
- **Cylindrical is checked before Linear and Planar** — this prevents cylinder surfaces from being split between the two. A point with linearity=0.35 and planarity=0.30 (both > 0.25) is Cylindrical, not Linear
- Linear is checked before Planar — a point with linearity=0.45 and planarity=0.15 is Linear
- Scatter is checked last of the feature-based classes

## Tuning Each Parameter

### `cylindrical_threshold` (default: 0.25)

**What it controls:** A point is Cylindrical when BOTH its linearity AND planarity exceed this threshold. This selects the diagonal band on the planarity-linearity edge of the ternary diagram.

- **0.0:** Disabled. No Cylindrical class (reverts to the old 5-class scheme).
- **Lower (0.1–0.2):** Very broad — captures points with even weak mixed geometry. May over-classify noisy/transition points as Cylindrical.
- **Default (0.25):** Both features must be at least 0.25, meaning sphericity ≤ 0.5. Captures clear cylinder surfaces (tree trunks, pipes, utility poles with some cross-section).
- **Higher (0.3–0.4):** Only very strong mixed geometry — larger cylinder surfaces with well-defined curvature.

**Visual check:** Points that appear **yellow** (red + green) in the eigenvalue view are cylinder candidates. If yellow points are being classified as Linear or Planar, lower the threshold.

**Interaction with Linear/Planar thresholds:** Cylindrical is checked first, so it "steals" points from both Linear and Planar. Points classified as Cylindrical have both features above `cylindrical_threshold` — if you raise it, more mixed points fall through to Linear or Planar depending on which feature is higher.

### `linear_threshold` (default: 0.4)

**What it controls:** Minimum linearity to classify as Linear (green), checked after Cylindrical.

- **Lower (0.2–0.3):** More points classified as Linear. Catches weaker linear features like thin branches, edges of surfaces.
- **Default (0.4):** Good for distinct linear features — cables, poles, strong edges.
- **Higher (0.5–0.7):** Only very strong linear features. Cables and utility poles but not surface edges.

**Visual check:** Enable the eigenvalue color view. Points that appear **bright green** should be captured by Linear. If you see green points classified as something else, lower the threshold.

### `planar_threshold` (default: 0.3)

**What it controls:** Minimum planarity to classify as Planar (red), checked after both Cylindrical and Linear.

- **Lower (0.15–0.25):** More points classified as Planar. Catches rough surfaces, partially vegetated ground.
- **Default (0.3):** Good for clear flat surfaces — walls, well-maintained ground, roofs.
- **Higher (0.4–0.6):** Only very clean planar surfaces. Smooth walls and concrete.

**Visual check:** Points that appear **bright red** in the eigenvalue view should be Planar. If red points are falling into Transition, lower the threshold.

### `sphericity_threshold` (default: 0.3)

**What it controls:** Minimum sphericity to classify as Scatter (blue), checked after Cylindrical, Linear, and Planar.

- **Lower (0.1–0.2):** Almost everything remaining becomes Scatter. Transition class nearly empty.
- **Default (0.3):** Captures most remaining points as Scatter with default thresholds.
- **Higher (0.4–0.5):** More points fall into Transition. Only strongly isotropic points are Scatter.

### `sparse_percentile` (default: 2.0)

**What it controls:** Bottom and top N% of total_variance are classified as Sparse before any feature-based classification.

- **0.0:** Disabled entirely. No Sparse class.
- **Default (2.0):** Removes the 2% most extreme points (both very low and very high variance).
- **Higher (5.0–10.0):** Aggressively removes outliers. Good if your data has lots of noise.

## Recommended Starting Configurations

### Urban scenes (buildings, roads, poles, cables, tree trunks)
```
cylindrical_threshold: 0.25   (capture pipes, tree trunks, larger poles)
linear_threshold:      0.4    (strong linear features)
planar_threshold:      0.3    (clean surfaces)
sphericity_threshold:  0.35   (vegetation/scatter clearly different)
sparse_percentile:     2.0    (remove scan edges)
```

### Vegetation-heavy scenes (forests, parks)
```
cylindrical_threshold: 0.2    (catch tree trunks with rougher surfaces)
linear_threshold:      0.5    (only very clear linear features)
planar_threshold:      0.25   (catch rough ground)
sphericity_threshold:  0.25   (more vegetation classified as Scatter)
sparse_percentile:     1.0    (less aggressive sparse removal)
```

### Infrastructure inspection (pipes, columns, rails)
```
cylindrical_threshold: 0.2    (aggressive cylinder detection)
linear_threshold:      0.45   (rails, cables)
planar_threshold:      0.35   (clean structural surfaces)
sphericity_threshold:  0.3    (standard)
sparse_percentile:     2.0    (standard)
```

### Clean, well-sampled data
```
cylindrical_threshold: 0.25
linear_threshold:      0.4
planar_threshold:      0.3
sphericity_threshold:  0.3
sparse_percentile:     0.0    (disable — data is clean)
```

## Interaction Between Thresholds

Because features sum to 1.0 and the cascade is ordered, changing one threshold affects what's available for later classes:

1. **Raising `cylindrical_threshold`** → fewer Cylindrical, more go to Linear/Planar
2. **Raising `linear_threshold`** → fewer Linear, more go to Planar/Scatter/Transition
3. **Raising `planar_threshold`** → fewer Planar, more go to Scatter/Transition
4. **Raising `sphericity_threshold`** → fewer Scatter, more Transition
5. **Raising `sparse_percentile`** → more Sparse, fewer of everything else

The **Transition class size** is a good diagnostic:
- **Very large Transition (>30%):** Your thresholds are too high — lower one or more
- **Very small Transition (<5%):** Your thresholds effectively partition the space — this is fine
- **Zero Transition:** Expected when thresholds cover the full ternary diagram
