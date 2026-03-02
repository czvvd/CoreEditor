ns-train CoreEditor --load-checkpoint unedited_models/bear/bearGS/splatfacto/2025-12-23_195550/nerfstudio_models/step-000029999.ckpt \
--experiment-name bear \
--viewer.quit-on-train-completion True \
--output-dir outputs \
--pipeline.datamanager.data data/bear \
--pipeline.render_rate 1500 \
--pipeline.edit_prompt "a photo of a jade bear" \
--pipeline.reverse_prompt "a photo of a bear statue in the forest" \
--pipeline.added_prompt "best quality" \
--pipeline.negative_prompt "longbody, lowres, bad anatomy, bad hands, missing finger, extra digit, fewer digits, cropped, worst quality" \
--pipeline.mask_factor 2.0 \
--pipeline.guidance_scale 6 \
--pipeline.inject_step 20 \
--pipeline.feature_threshold 2.5 \
--pipeline.attn_fusion_rate 0.7 \
--pipeline.diff_t 261 \
--pipeline.datamanager.subset_num 40 \
--pipeline.edited_save_path "edited_test/Ours/" \
--pipeline.edited_save_name "Ours" \
--pipeline.langsam_obj 'bear'









