.hsa_code_object_version 2,0
.hsa_code_object_isa 10, 3, 0, "AMD", "AMDGPU"

.text
.p2align 8
.amdgpu_hsa_kernel hello

hello:

   .amd_kernel_code_t
      enable_sgpr_kernarg_segment_ptr = 1
      is_ptr64 = 1
      compute_pgm_rsrc1_vgprs = 8
      compute_pgm_rsrc1_sgprs = 8
      compute_pgm_rsrc2_user_sgpr = 2
      kernarg_segment_byte_size = 0x18
      wavefront_sgpr_count = 6
      workitem_vgpr_count = 5
  .end_amd_kernel_code_t

  s_load_dwordx2 s[4:5], s[0:1] 0x10 // load out_C into s[4:5] from kernarg
  s_load_dwordx4 s[0:3], s[0:1] 0x00 // load in_A into s[0:1], in_B into s[2:3] from kernarg
  v_lshlrev_b32  v0, 2, v0           // v0 *= 4;	v0 hold workitem id x
  s_waitcnt lgkmcnt(0)

  // compute address of corresponding element of in_A buffer
  // i.e. v[1:2] = &in_A[workitem_id]
  v_add_co_u32 v1, vcc_lo, s0, v0
  v_mov_b32 v2, s1
  v_add_co_ci_u32_e32 v2, vcc_lo, 0, v2, vcc_lo

  // compute address of corresponding element of in_B buffer
  // i.e. v[3:4] = &in_B[workitem_id]
  v_add_co_u32 v3, vcc_lo, s2, v0
  v_mov_b32 v4, s3
  v_add_co_ci_u32_e32 v4, vcc_lo, 0, v4, vcc_lo

  flat_load_dword  v1, v[1:2] // load in_A[workitem_id] into v1
  flat_load_dword  v2, v[3:4] // load in_B[workitem_id] into v2
  s_waitcnt vmcnt(0) & lgkmcnt(0) // wait for memory reads to finish

  // compute address of corresponding element of out_C buffer
  // i.e. v[3:4] = &out_C[workitem_id]
  v_add_co_u32 v3, vcc_lo, s4, v0
  v_mov_b32 v4, s5
  v_add_co_ci_u32_e32 v4, vcc_lo, 0, v4, vcc_lo

  v_add_f32 v0, v1, v2
  flat_store_dword v[3:4], v0
  s_waitcnt 0
  s_endpgm
