# Benchmark Results (device: cpu)

## Non-stiff logistic

| Backend | Time (s) | Max Error | CPU MB | GPU MB | Notes |
|---------|----------|-----------|--------|--------|-------|
| diffsol | 0.0020 | 4.39e-07 | nan | 0.00 | adjoint grad final=-1.510e-01 |

## Stiff logistic

| Backend | Time (s) | Max Error | CPU MB | GPU MB | Notes |
|---------|----------|-----------|--------|--------|-------|
| diffsol | 0.0027 | 7.34e-07 | nan | 0.00 | adjoint grad final=-0.000e+00 |

## Neural ODE block

| Backend | Time (s) | Max Error | CPU MB | GPU MB | Notes |
|---------|----------|-----------|--------|--------|-------|
| diffsol | 0.0021 | 0.00e+00 | nan | 0.00 | ||z(T)||=1.861 |
