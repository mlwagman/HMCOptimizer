# Hasenbusch mass preconditioning — deep reference

## Terminology note

"Pure QCD" here means QCD only (gauge field + quarks), without
electroweak or BSM sectors — it still includes fermion determinants and
therefore can use Hasenbusch preconditioning. The fermion-free case is
**pure gauge** / **Yang-Mills**; that case has no fermion determinants
and no use for Hasenbusch.

## A reference ladder lifted from Grid TXQCD source

```
{0.005, 0.0145, 0.045, 0.108, 0.25, 0.35, 0.51, 0.6, 0.8}
```

Nine rungs, roughly log-spaced. This is the ladder hard-coded in
`Grid-TXQCD/HMC/Mobius2p1f_EOFA_96I_hmc.cc:231` as the `hasenbusch_mass`
vector for a 2+1f Möbius DWF ensemble with light mass `7.8e-4`. It is a
reasonable *starting point* for tuning — not a validated optimum for any
other physics point. Adjust per the rules below.

## Why this style of ladder

Hasenbusch preconditioning splits the light-fermion determinant as a
telescoping ratio

    det[ M(m_l) ] = det[ M(m_1) ] · ∏_k det[ M(m_{k+1}) / M(m_k) ]

with `m_l < m_1 < m_2 < ... < m_N`. Each ratio is represented by its own
pseudofermion with its own CG solve and its own force contribution to MD.
Heuristic: the force from a ratio `M(m_{k+1}) / M(m_k)` is small when
`m_{k+1} / m_k ≈ O(1.5 to 3)`, because both determinants see similar low
modes.

Rungs too close → many pseudofermions, much CG work, diminishing force
reduction. Rungs too spread → large force per rung, small dt forced by MD
truncation. A roughly log-spaced ladder between `m_l` and a UV cutoff
(often ~1.0) is a common compromise.

## Rung-dropping

For smaller lattices or different `m_l`, a shorter ladder may suffice.
Heuristics:

- Drop rungs near the heavy end first (0.8, 0.6, 0.51) — they still cost
  CG work but contribute little force when they are already well above
  the dynamical scale.
- Keep the lightest 2–3 rungs intact if `m_l` is small; that is where
  Hasenbusch buys the most.

## Rung-adding

Add rungs only when dH blows up despite integrator tuning *and* the
blowup correlates with force spikes from one specific ratio. Grid prints
force norms per pseudofermion — inspect those before adding.

## Interaction with integrator tuning

- Integrator + dt determine truncation error given a fixed set of forces.
- Hasenbusch ladder determines the forces themselves.
- Tuning in the order *integrator → dt → ladder* tends to be stable.
  Tuning *ladder* and *dt* together gives noisy feedback because the
  force distribution changes under you.

## Pure-gauge special case

For **pure gauge** (Yang-Mills) runs there are no fermion determinants,
so the Hasenbusch ladder is unused. Our Jinja templates carry the
`hasenbusch` variable as a list; pass `[]` for pure gauge and the
rendered source should not instantiate any Hasenbusch pseudofermions.
