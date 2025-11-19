# Proyecto B Caso 3 - Rural/Offshore with Resupply (La Guajira)

## Project Context

**Advanced case with resupply/re-stocking logic**. Same context as Caso 2 but vehicles can:
- **Return to depot mid-route** to reload/recharge
- **Make multiple trips** in a single operational period
- **Still respect time windows** for all deliveries

This models realistic long-distance scenarios where range/capacity limit single-trip coverage.

**Case Level**: Advanced realistic (20% + bonus eligible)

## What's New in Caso 3?

**RESUPPLY ALLOWED**: Vehicles can return to depot (CD01), reload/recharge, and continue serving clients.

Example route: `CD01-C004-C011-CD01-C008-C012-CD01`
- Trip 1: Serve C004, C011
- Return to depot, resupply
- Trip 2: Serve C008, C012
- Return to depot

## Data Files

Same structure as Caso 2 - see Project B Caso 2 README for details.

**Key difference**: More clients and/or tighter time windows may REQUIRE resupply for feasibility.

## Cost Parameters

**Identical to Caso 2** - see `parameters_rural.csv`:
- Same fixed, distance, and time costs
- Same energy/fuel prices and consumption rates
- Additional flag: `resupply_allowed=TRUE`

**Important**: Multiple returns to depot mean multiple "activations" are NOT charged. C_fixed is charged ONCE per vehicle used, not per trip.

## Project-Specific Constraints

All Caso 2 constraints PLUS:

1. **Resupply Logic**:
   - Vehicle can visit depot (CD01) multiple times
   - Each return resets capacity to full
   - Each return resets range/fuel/battery to full
   - Time windows still must be respected

2. **Time Continuity**:
   - Time flows continuously across resupply
   - Resupply takes time (travel back, reload, travel out)
   - Cannot "teleport" - must account for all distances

3. **Feasibility**:
   - Even with resupply, individual client demands cannot exceed vehicle capacity
   - Time windows must still be achievable given travel times

## Verification File Format

Create `verificacion_caso3.csv`:

```csv
VehicleId,VehicleType,InitLoad,RouteSequence,Clients,DemandSatisfied,ArrivalTimes,Resup,ResupAmounts,Distance,Time,Cost
V001,Drone,35,CD01-C004-C011-CD01-C008-C012-CD01,4,15-20-25-25,10:35-11:15-12:30-13:20,1,50,62.5,185.0,350000
```

### New Columns:
- **Resup** (Resupplies): Number of times vehicle returned to depot for resupply
- **ResupAmounts**: Amount reloaded at each resupply, separated by hyphens

**Note**: `Clients=4` but `RouteSequence` has 3 instances of CD01 (start + 1 resupply + end)

## Deliverables for Caso 3

Same as Caso 2 PLUS:

1. **Resupply Analysis**:
   - Average resupplies per vehicle type
   - Impact of resupply on total cost vs. using more vehicles
   - Sensitivity: How do time windows affect need for resupply?

2. **Business Insights**:
   - Is resupply cost-effective vs. deploying more vehicles?
   - Which configurations are most robust for remote zones?
   - Trade-offs between drone quick-resupply vs. truck fewer-resupply

## Visualization Requirements

Same as Caso 2 PLUS:
- Clearly mark resupply points on route maps
- Show load evolution over time (resets at depot)
- Timeline showing resupply events

## Solving Strategy

Resupply dramatically increases solution space:

1. **Model carefully**: Resupply visits are "optional" depot visits mid-route
2. **Track state**: Capacity and range reset at depot, time accumulates
3. **Avoid over-resupply**: Model should not resupply unnecessarily (costs time)
4. **Test incrementally**: Start with cases that don't need resupply, then add complexity

## Important Notes

- Resupply is ALLOWED, not REQUIRED (use only if beneficial)
- C_fixed charged once per vehicle, regardless of trips
- Time windows remain HARD constraints
- Distinguish depot visits: start, resupply, end
- Model should optimize: fewer resupplies is generally better (less time)

## ID Standardization

Same as all cases:
- Clients: C001, C002, ...
- Vehicles: V001, V002, ...
- Depot: CD01 (appears multiple times in routes with resupply)
- Vehicle types: "Drone" or "Truck"

## Competitive Bonus Eligibility

This case qualifies for up to +20% bonus. Focus on:
- Efficient resupply strategies
- Cost comparison with non-resupply baseline
- Deep analysis of when/why resupply is beneficial
- Innovative formulation approaches
