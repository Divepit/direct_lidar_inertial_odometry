#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$1"
OUT_DIR="$2"
CSV_FILE="${OUT_DIR}/run_stats.csv"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

cat > "${CSV_FILE}" <<'CSV'
time_s,stamp_sec,p_x_m,p_y_m,p_z_m,q_w,q_x,q_y,q_z,roll_rad,pitch_rad,yaw_rad,vlin_b_x_mps,vlin_b_y_mps,vlin_b_z_mps,vang_b_x_radps,vang_b_y_radps,vang_b_z_radps,accel_bias_x_mps2,accel_bias_y_mps2,accel_bias_z_mps2,gyro_bias_x_radps,gyro_bias_y_radps,gyro_bias_z_radps
0.0,100.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01,0.02,0.03,0.001,0.002,0.003
0.1,100.1,0.1,0.0,0.0,0.999,0.0,0.0,0.045,0.0,0.0,0.09,1.0,0.0,0.0,0.0,0.0,0.9,0.01,0.02,0.03,0.001,0.002,0.003
0.2,100.2,0.2,0.1,0.0,0.996,0.0,0.0,0.089,0.0,0.0,0.18,1.0,0.5,0.0,0.0,0.0,0.9,0.01,0.02,0.03,0.001,0.002,0.003
CSV

python3 "${SCRIPT}" --csv "${CSV_FILE}" --out-dir "${OUT_DIR}" --dpi 600

for file in \
  pose_position.pdf pose_position.png \
  pose_orientation_rpy.pdf pose_orientation_rpy.png \
  twist_linear_body.pdf twist_linear_body.png \
  twist_angular_body.pdf twist_angular_body.png \
  bias_accel.pdf bias_accel.png \
  bias_gyro.pdf bias_gyro.png \
  run_stats_summary.txt; do
  test -s "${OUT_DIR}/${file}"
done
