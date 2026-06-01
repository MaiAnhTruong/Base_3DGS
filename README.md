## Hướng dẫn sử dụng

```powershell
python train.py `
  -s "{path_đến_data_full}" `
  -m "{path_đầu_ra_output}" `
  --eval `
  --disable_viewer `
  --iterations 10000 `
  --test_iterations 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 `
  --save_iterations 7000 10000 `
  --checkpoint_iterations 7000 10000 `
  --metrics_log_interval 1000 `
  --metrics_eval_train_count -1 `
  --metrics_eval_per_view `
  --metrics_compute_lpips `
  --split_train_views 12 `
  --split_hold 8 `
  --split_train_sample_mode paper_even `
  --split_output_root "{path_để_lưu_data_đã_chia}" `
  --split_copy_mode copy `
  --split_init_policy sparsegs_triangulate `
  --split_colmap_matcher exhaustive `
  --split_min_triangulated_points 100 `
  --split_force
```
