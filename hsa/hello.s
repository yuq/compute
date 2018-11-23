.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 0, "AMD", "AMDGPU"

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
      kernarg_segment_byte_size = 8
      wavefront_sgpr_count = 2
      workitem_vgpr_count = 3
  .end_amd_kernel_code_t

  s_load_dwordx2 s[0:1], s[0:1] 0x0 // load out into s[0:1] from kernarg
  v_lshlrev_b32  v0, 2, v0          // v0 *= 4;	v0 hold workitem id x
  s_waitcnt lgkmcnt(0)

  // compute address of corresponding element of out buffer
  // i.e. v[1:2] = &out[workitem_id]
  v_add_co_u32 v1, vcc, s0, v0
  v_mov_b32 v2, s1
  v_addc_co_u32 v2, vcc, v2, 0, vcc

  v_mov_b32 v0, 3.14159
  flat_store_dword v[1:2], v0
  s_waitcnt 0
  s_endpgm
