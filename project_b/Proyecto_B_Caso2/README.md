# Proyecto B Caso 2 - Rural/Offshore Critical Delivery (La Guajira)

## Project Context

**LogistiCo Rural/Offshore Logistics Division** delivers medical supplies and essentials to remote communities in La Guajira. This case features:
- **Hybrid fleet**: Drones (fast, limited capacity) + 4x4 trucks (slower, higher capacity)
- **Time windows** (HARD constraints - penalties NOT allowed)
- **Long distances** to remote locations
- **Single depot** with operations starting at 07:45

**Important**: Both drones and trucks can access ALL clients. Vehicle choice is based purely on cost, capacity, and operational constraints.

**Case Level**: Simplified project-specific (25% of implementation grade)

## Data Files

### 1. `clients.csv`
Remote community locations with delivery time windows.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| ClientID | Integer | - | Numeric client identifier |
| StandardizedID | String | - | Standardized ID (C001, C002, ...) for verification files |
| LocationID | Integer | - | Unified location identifier |
| Latitude | Float | degrees | Geographic latitude in La Guajira |
| Longitude | Float | degrees | Geographic longitude in La Guajira |
| Demand | Float | kg | Medical supplies / essentials required |
| **TimeWindow** | String | HH:MM-HH:MM | **Delivery must occur within this time range** |

**Time Window Example**: `09:00-09:30` means delivery must happen between 9:00 AM and 9:30 AM.

### 2. `vehicles.csv`
Hybrid fleet specifications.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| VehicleID | Integer | - | Numeric vehicle identifier |
| StandardizedID | String | - | Standardized ID (V001, V002, ...) for verification files |
| **Type** | String | - | **"drone" or "4x4"** (determines cost structure) |
| Capacity | Float | kg | Maximum payload |
| Range | Float | km | Maximum distance on full charge/tank |
| **Speed** | Float | km/h | **Average travel speed (drones only, trucks use estimated speed)** |

### 3. `depots.csv`
Single operations depot.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| DepotID | Integer | - | Numeric depot identifier (1) |
| StandardizedID | String | - | Standardized ID (CD01) for verification files |
| LocationID | Integer | - | Unified location identifier |
| Latitude | Float | degrees | Depot latitude |
| Longitude | Float | degrees | Depot longitude |

## Cost Parameters

See `parameters_rural.csv`. Key parameters:

### Drone Costs:
- **C_fixed_drone**: 30,000 COP/vehicle
- **C_dist_drone**: 800 COP/km
- **C_time_drone**: 2,000 COP/hour
- **energy_price_drone**: 937.81 COP/kWh
- **Energy consumption**: 0.06-0.12 kWh/km (0.09 typical)

### 4x4 Truck Costs:
- **C_fixed_truck**: 60,000 COP/vehicle
- **C_dist_truck**: 3,000 COP/km
- **C_time_truck**: 8,000 COP/hour
- **fuel_price_truck**: 16,300 COP/gallon
- **Fuel efficiency**: 2.1-2.5 km/L (8-9.5 km/gal)

### Objective Function:
```
min Z = Σ(C_fixed_type × y_v) + Σ(C_dist_type × d_v) + Σ(C_time_type × t_v) + C_energy

Where:
- type = "drone" or "truck" (determines which cost parameters to use)
- C_energy = Σ(energy consumption) for drones + Σ(fuel cost) for trucks
```

### Operations Schedule:
- **Start time**: 07:45 (all vehicles depart from depot at or after this time)
- Vehicles travel to clients and must arrive within time windows
- Travel time = distance / speed

## Project-Specific Constraints

1. **Time Windows (HARD)**: Arrival at each client must be within [start, end] window
2. **Hybrid Fleet**: Model must handle two vehicle types with different costs
3. **No Resupply**: Vehicles cannot return to depot mid-route (Caso 2 only - see Caso 3 for resupply)
4. **Capacity & Range**: Standard constraints apply per vehicle type
5. **Coverage**: All clients visited exactly once

**NOTE**: Time window violations are INFEASIBLE - do NOT use penalty functions!

## Verification File Format

Create `verificacion_caso2.csv`:

```csv
VehicleId,VehicleType,InitialLoad,RouteSequence,ClientsServed,DemandSatisfied,ArrivalTimes,TotalDistance,TotalTime,Cost
V001,Drone,35,CD01-C004-C011-CD01,2,15-20,10:35-11:15,28.5,55.0,185000
V002,Truck,150,CD01-C002-C007-C009-CD01,3,45-60-45,09:45-10:30-11:15,42.8,120.0,275000
```

### Key Columns:
- **VehicleType**: "Drone" or "Truck" (capitalized)
- **ArrivalTimes**: Actual arrival time at each client (HH:MM format), separated by hyphens
- **TotalTime**: Total operation time in MINUTES
- **Cost**: Total operational cost for this vehicle (all components)

**Validation**: Arrival times will be checked against time windows in clients.csv!

## Deliverables for Caso 2

1. ✓ Correct hybrid fleet model with type-specific costs (15%)
2. ✓ Feasible solution meeting ALL time windows (7%)
3. ✓ Visualization and basic reporting (3%)
   - Map with routes differentiated by vehicle type
   - Gantt chart showing time window compliance

## Visualization Requirements

**Required**:
- Map with drone routes (one color) and truck routes (another color)
- Gantt chart or timeline showing:
  - Each client's time window (as bars)
  - Actual arrival times (as points/markers)
  - Visual confirmation of time window compliance

**Recommended**:
- Load evolution per vehicle
- Cost comparison: drone vs. truck efficiency

## Important Notes

- **Start time is 07:45** - all routes begin at or after this time
- **No resupply in Caso 2** - vehicles make single trip from/to depot
- **Time windows are HARD** - infeasible if violated
- Calculate arrival times carefully: departure time + Σ(travel times)
- Drones are faster but have limited capacity and range
- Trucks are slower but can carry more and travel farther

## Time Calculation Example

```
Vehicle V001 (Drone, Speed = 135 km/h):
- Depart depot: 07:45
- Travel to C004 (25 km): 25/135 = 0.185 hours = 11.1 minutes
- Arrive C004: 07:56
- Service time: ~5 minutes (estimated)
- Depart C004: 08:01
- Travel to C011 (30 km): 30/135 = 0.222 hours = 13.3 minutes
- Arrive C011: 08:14
...
```

## ID Standardization

Use StandardizedID in verification files:
- Clients: C001, C002, C003, ...
- Vehicles: V001, V002, V003, ...
- Depot: CD01

Vehicle type names: "Drone" or "Truck" (capitalize first letter)
