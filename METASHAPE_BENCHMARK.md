# Metashape whole-field benchmark

One representative date per field-year, full pipeline (align → depth maps → point cloud → DEM → orthomosaic) on GPU. Projection size = orthomosaic raster dimensions and ground coverage in UTM 16N; storage size = exported single GeoTIFF (EPSG:32616).

| Field | Year | Date | Captures | Proc. time | Ortho pixels (W×H) | Resolution | Ground coverage | Storage size |
|-------|------|------|---------:|-----------:|--------------------|-----------:|-----------------|-------------:|
| PPAC-B3 | 2021 | 073021 | 178 | 5.8 min | 10,240 × 7,166 | 3.77 cm/px | 386 × 270 m | 615.0 MB |
| PPAC-B3 | 2022 | 072922 | 129 | 4.2 min | 9,728 × 5,632 | 3.44 cm/px | 335 × 194 m | 561.2 MB |
| PPAC-B3 | 2023 | 071023 | 216 | 7.0 min | 9,728 × 6,144 | 3.45 cm/px | 336 × 212 m | 560.0 MB |
| PPAC-B3 | 2024 | 072324 | 286 | 7.4 min | 8,704 × 5,632 | 3.43 cm/px | 299 × 193 m | 525.1 MB |
| RiceFarm-SouthPivot | 2021 | 080421 | 1074 | 41.8 min | 14,848 × 16,384 | 3.80 cm/px | 564 × 622 m | 2.7 GB |
| SWPAC | 2021 | 080221 | 979 | 37.3 min | 10,752 × 11,776 | 3.53 cm/px | 379 × 415 m | 1.2 GB |
| Rice-NorthPivot | 2022 | 071222 | 1091 | 47.9 min | 27,135 × 19,285 | 3.46 cm/px | 939 × 667 m | 4.0 GB |
| Rice-SouthPivot | 2023 | 071423 | 1458 | 64.5 min | 17,919 × 16,383 | 3.46 cm/px | 620 × 567 m | 2.6 GB |
| SEPAC-D3 | 2022 | 070722 | 288 | 9.7 min | 9,728 × 11,776 | 3.47 cm/px | 338 × 409 m | 735.5 MB |
| SEPAC-D3 | 2023 | 080423 | 564 | 22.5 min | 9,215 × 11,775 | 3.43 cm/px | 316 × 404 m | 704.0 MB |
