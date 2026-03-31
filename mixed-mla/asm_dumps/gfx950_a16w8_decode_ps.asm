
mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co:	file format elf64-amdgpu

Disassembly of section .text:

0000000000002300 <_ZN5aiter41mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_psE>:
	s_and_b32 s1, s1, 0xffff                                   // 000000002300: 8601FF01 0000FFFF
	s_load_dwordx2 s[28:29], s[0:1], 0xe0                      // 000000002308: C0060700 000000E0
	v_lshrrev_b32_e32 v1, 10, v0                               // 000000002310: 2002008A
	v_lshrrev_b32_e32 v2, 10, v1                               // 000000002314: 2004028A
	v_and_b32_e32 v2, 0x3ff, v2                                // 000000002318: 260404FF 000003FF
	v_and_b32_e32 v1, 0x3ff, v1                                // 000000002320: 260202FF 000003FF
	v_and_b32_e32 v0, 0x3ff, v0                                // 000000002328: 260000FF 000003FF
	v_lshrrev_b32_e32 v3, 6, v0                                // 000000002330: 20060086
	v_and_b32_e32 v0, 63, v0                                   // 000000002334: 260000BF
	s_mov_b32 s2, s2                                           // 000000002338: BE820002
	s_mov_b32 s3, s3                                           // 00000000233C: BE830003
	s_mov_b32 s4, s4                                           // 000000002340: BE840004
	v_readfirstlane_b32 s7, v3                                 // 000000002344: 7E0E0503
	s_waitcnt lgkmcnt(0)                                       // 000000002348: BF8CC07F
	s_and_b32 s29, s29, 0xffff                                 // 00000000234C: 861DFF1D 0000FFFF
	s_load_dwordx2 s[32:33], s[28:29], 0x0                     // 000000002354: C006080E 00000000
	s_load_dwordx2 s[30:31], s[28:29], 0x8                     // 00000000235C: C006078E 00000008
	s_waitcnt lgkmcnt(0)                                       // 000000002364: BF8CC07F
	s_mul_i32 s56, s2, 4                                       // 000000002368: 92388402
	s_and_b32 s33, s33, 0xffff                                 // 00000000236C: 8621FF21 0000FFFF
	s_add_u32 s32, s56, s32                                    // 000000002374: 80202038
	s_addc_u32 s33, 0, s33                                     // 000000002378: 82212180
	s_load_dword s85, s[32:33], 0x0                            // 00000000237C: C0021550 00000000
	s_load_dword s86, s[32:33], 0x4                            // 000000002384: C0021590 00000004
	s_and_b32 s31, s31, 0xffff                                 // 00000000238C: 861FFF1F 0000FFFF
	s_waitcnt lgkmcnt(0)                                       // 000000002394: BF8CC07F
	s_cmp_eq_i32 s85, s86                                      // 000000002398: BF005655
	s_cbranch_scc1 label_1A0D                                  // 00000000239C: BF8519E5
	s_mul_i32 s56, s85, 32                                     // 0000000023A0: 9238A055

00000000000023a4 <label_0029>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 0000000023A4: BF8C0000
	s_barrier                                                  // 0000000023A8: BF8A0000
	s_add_u32 s30, s56, s30                                    // 0000000023AC: 801E1E38
	s_addc_u32 s31, 0, s31                                     // 0000000023B0: 821F1F80
	s_load_dword s87, s[30:31], 0x4                            // 0000000023B4: C00215CF 00000004
	s_load_dword s78, s[30:31], 0x8                            // 0000000023BC: C002138F 00000008
	s_load_dword s79, s[30:31], 0xc                            // 0000000023C4: C00213CF 0000000C
	s_load_dword s46, s[30:31], 0x10                           // 0000000023CC: C0020B8F 00000010
	s_load_dword s47, s[30:31], 0x14                           // 0000000023D4: C0020BCF 00000014
	s_load_dword s62, s[30:31], 0x18                           // 0000000023DC: C0020F8F 00000018
	s_load_dwordx2 s[8:9], s[0:1], 0x0                         // 0000000023E4: C0060200 00000000
	s_load_dwordx2 s[12:13], s[0:1], 0x10                      // 0000000023EC: C0060300 00000010
	s_load_dwordx2 s[16:17], s[0:1], 0x20                      // 0000000023F4: C0060400 00000020
	s_load_dwordx2 s[20:21], s[0:1], 0x30                      // 0000000023FC: C0060500 00000030
	s_load_dwordx2 s[24:25], s[0:1], 0x50                      // 000000002404: C0060600 00000050
	s_load_dword s64, s[0:1], 0x70                             // 00000000240C: C0021000 00000070
	s_load_dword s65, s[0:1], 0x80                             // 000000002414: C0021040 00000080
	s_load_dword s66, s[0:1], 0xa0                             // 00000000241C: C0021080 000000A0
	s_load_dword s68, s[0:1], 0xb0                             // 000000002424: C0021100 000000B0
	s_load_dword s69, s[0:1], 0xc0                             // 00000000242C: C0021140 000000C0
	s_load_dwordx2 s[88:89], s[0:1], 0xf0                      // 000000002434: C0061600 000000F0
	s_load_dwordx2 s[40:41], s[0:1], 0x110                     // 00000000243C: C0060A00 00000110
	s_waitcnt lgkmcnt(0)                                       // 000000002444: BF8CC07F
	s_mov_b32 s65, 16                                          // 000000002448: BEC10090
	s_mul_i32 s75, 0x800, s65                                  // 00000000244C: 924B41FF 00000800
	s_mul_i32 s74, 0x480, s65                                  // 000000002454: 924A41FF 00000480
	s_mul_i32 s56, 4, s65                                      // 00000000245C: 92384184
	s_mov_b32 s10, s75                                         // 000000002460: BE8A004B
	s_mov_b32 s18, -16                                         // 000000002464: BE9200D0
	s_mov_b32 s14, -16                                         // 000000002468: BE8E00D0
	s_mov_b32 s22, -16                                         // 00000000246C: BE9600D0
	s_mov_b32 s26, -16                                         // 000000002470: BE9A00D0
	s_mov_b32 s11, 0x20000                                     // 000000002474: BE8B00FF 00020000
	s_mov_b32 s91, 0x20000                                     // 00000000247C: BEDB00FF 00020000
	s_mov_b32 s19, 0x20000                                     // 000000002484: BE9300FF 00020000
	s_mov_b32 s15, 0x20000                                     // 00000000248C: BE8F00FF 00020000
	s_mov_b32 s23, 0x20000                                     // 000000002494: BE9700FF 00020000
	s_mov_b32 s27, 0x20000                                     // 00000000249C: BE9B00FF 00020000
	s_and_b32 s9, s9, 0xffff                                   // 0000000024A4: 8609FF09 0000FFFF
	s_and_b32 s89, s89, 0xffff                                 // 0000000024AC: 8659FF59 0000FFFF
	s_and_b32 s17, s17, 0xffff                                 // 0000000024B4: 8611FF11 0000FFFF
	s_and_b32 s13, s13, 0xffff                                 // 0000000024BC: 860DFF0D 0000FFFF
	s_and_b32 s21, s21, 0xffff                                 // 0000000024C4: 8615FF15 0000FFFF
	s_and_b32 s25, s25, 0xffff                                 // 0000000024CC: 8619FF19 0000FFFF
	s_and_b32 s41, s41, 0xffff                                 // 0000000024D4: 8629FF29 0000FFFF
	s_or_b32 s9, s9, 0x40000                                   // 0000000024DC: 8709FF09 00040000
	s_or_b32 s89, s89, 0x40000                                 // 0000000024E4: 8759FF59 00040000
	s_or_b32 s17, s17, 0x40000                                 // 0000000024EC: 8711FF11 00040000
	s_or_b32 s13, s13, 0x40000                                 // 0000000024F4: 870DFF0D 00040000
	s_or_b32 s21, s21, 0x40000                                 // 0000000024FC: 8715FF15 00040000
	s_or_b32 s25, s25, 0x40000                                 // 000000002504: 8719FF19 00040000
	s_waitcnt lgkmcnt(0)                                       // 00000000250C: BF8CC07F
	s_mov_b32 s67, 1                                           // 000000002510: BEC30081
	s_load_dword s43, s[40:41], 0x0                            // 000000002514: C0020AD4 00000000
	s_mov_b32 s80, 0                                           // 00000000251C: BED00080
	s_sub_u32 s81, s79, s78                                    // 000000002520: 80D14E4F
	s_mov_b32 s69, 0                                           // 000000002524: BEC50080
	s_lshr_b32 s44, 16, s69                                    // 000000002528: 8F2C4590
	s_mul_i32 s73, s44, 4                                      // 00000000252C: 9249842C
	s_mul_i32 s73, s73, s67                                    // 000000002530: 92494349
	s_mul_i32 s45, s4, s44                                     // 000000002534: 922D2C04
	s_sub_u32 s50, s47, s46                                    // 000000002538: 80B22E2F
	s_lshl_b32 s56, s50, s69                                   // 00000000253C: 8E384532
	s_sub_u32 s82, s56, s81                                    // 000000002540: 80D25138
	s_mov_b32 s58, s62                                         // 000000002544: BEBA003E
	s_add_u32 s82, s82, s58                                    // 000000002548: 80523A52
	s_add_u32 s57, s82, 8                                      // 00000000254C: 80398852
	s_min_u32 s56, s56, s57                                    // 000000002550: 83B83938
	s_lshr_b32 s50, s56, s69                                   // 000000002554: 8F324538
	s_lshl_b32 s56, s45, s69                                   // 000000002558: 8E38452D
	s_add_u32 s83, s56, 15                                     // 00000000255C: 80538F38
	s_mul_i32 s84, s67, 16                                     // 000000002560: 92549043
	s_cmp_le_u32 s50, s45                                      // 000000002564: BF0B2D32
	s_cbranch_scc1 label_1A0D                                  // 000000002568: BF851972
	s_mul_i32 s56, s50, 4                                      // 00000000256C: 92388432
	s_mov_b32 s26, s56                                         // 000000002570: BE9A0038
	s_mul_i32 s56, s46, 4                                      // 000000002574: 9238842E
	s_add_u32 s24, s56, s24                                    // 000000002578: 80181838
	s_addc_u32 s25, 0, s25                                     // 00000000257C: 82191980
	s_mov_b32 s70, 0                                           // 000000002580: BEC60080
	s_sub_u32 s71, s50, s45                                    // 000000002584: 80C72D32
	s_mul_i32 s39, s67, s44                                    // 000000002588: 92272C43
	s_mov_b32 s38, s71                                         // 00000000258C: BEA60047
	v_cvt_f32_u32_e32 v20, s39                                 // 000000002590: 7E280C27
	s_sub_i32 s56, 0, s39                                      // 000000002594: 81B82780
	v_rcp_iflag_f32_e32 v20, v20                               // 000000002598: 7E284714
	s_nop 0                                                    // 00000000259C: BF800000
	v_mul_f32_e32 v20, 0x4f7ffffe, v20                         // 0000000025A0: 0A2828FF 4F7FFFFE
	v_cvt_u32_f32_e32 v20, v20                                 // 0000000025A8: 7E280F14
	v_mul_lo_u32 v21, s56, v20                                 // 0000000025AC: D2850015 00022838
	v_mul_hi_u32 v21, v20, v21                                 // 0000000025B4: D2860015 00022B14
	v_add_u32_e32 v20, v20, v21                                // 0000000025BC: 68282B14
	v_mul_hi_u32 v20, s38, v20                                 // 0000000025C0: D2860014 00022826
	v_mul_lo_u32 v21, v20, s39                                 // 0000000025C8: D2850015 00004F14
	v_sub_u32_e32 v23, s38, v21                                // 0000000025D0: 6A2E2A26
	v_add_u32_e32 v22, 1, v20                                  // 0000000025D4: 682C2881
	v_cmp_le_u32_e32 vcc, s39, v23                             // 0000000025D8: 7D962E27
	v_subrev_u32_e32 v21, s39, v23                             // 0000000025DC: 6C2A2E27
	s_nop 0                                                    // 0000000025E0: BF800000
	v_cndmask_b32_e32 v20, v20, v22, vcc                       // 0000000025E4: 00282D14
	v_cndmask_b32_e32 v23, v23, v21, vcc                       // 0000000025E8: 002E2B17
	v_add_u32_e32 v21, 1, v20                                  // 0000000025EC: 682A2881
	v_cmp_le_u32_e32 vcc, s39, v23                             // 0000000025F0: 7D962E27
	s_nop 1                                                    // 0000000025F4: BF800001
	v_cndmask_b32_e32 v23, v20, v21, vcc                       // 0000000025F8: 002E2B14
	s_nop 3                                                    // 0000000025FC: BF800003
	v_readfirstlane_b32 s40, v23                               // 000000002600: 7E500517
	s_nop 3                                                    // 000000002604: BF800003
	s_mov_b32 s71, s40                                         // 000000002608: BEC70028
	s_mul_i32 s56, s71, s39                                    // 00000000260C: 92382747
	s_sub_u32 s56, s38, s56                                    // 000000002610: 80B83826
	s_mov_b32 s57, 0                                           // 000000002614: BEB90080
	s_cmp_lt_u32 s56, s44                                      // 000000002618: BF0A2C38
	s_cselect_b32 s57, s57, 1                                  // 00000000261C: 85398139
	s_add_u32 s71, s57, s71                                    // 000000002620: 80474739
	s_cmpk_eq_u32 s57, 0x1                                     // 000000002624: B4390001
	s_cselect_b32 s49, 0, s56                                  // 000000002628: 85313880
	s_mov_b32 s48, s49                                         // 00000000262C: BEB00031
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000002630: 20280084
	v_lshlrev_b32_e32 v8, 2, v20                               // 000000002634: 24102882
	v_add_u32_e64 v8, v8, s7                                   // 000000002638: D1340008 00000F08
	v_add_u32_e32 v8, s45, v8                                  // 000000002640: 6810102D
	v_lshlrev_b32_e32 v8, 2, v8                                // 000000002644: 24101082
	buffer_load_dword v10, v8, s[24:27], 0 offen               // 000000002648: E0501000 80060A08
	v_add_u32_e32 v8, s73, v8                                  // 000000002650: 68101049
	buffer_load_dword v11, v8, s[24:27], 0 offen               // 000000002654: E0501000 80060B08
	v_add_u32_e32 v8, s73, v8                                  // 00000000265C: 68101049
	s_add_u32 s56, s80, s78                                    // 000000002660: 80384E50
	v_mov_b32_e32 v20, s56                                     // 000000002664: 7E280238
	v_mul_lo_u32 v21, s74, v20                                 // 000000002668: D2850015 0002284A
	v_mul_hi_u32 v22, s74, v20                                 // 000000002670: D2860016 0002284A
	s_nop 2                                                    // 000000002678: BF800002
	v_readfirstlane_b32 s56, v21                               // 00000000267C: 7E700515
	v_readfirstlane_b32 s57, v22                               // 000000002680: 7E720516
	s_nop 4                                                    // 000000002684: BF800004
	s_add_u32 s16, s56, s16                                    // 000000002688: 80101038
	s_addc_u32 s17, s57, s17                                   // 00000000268C: 82111139
	s_sub_u32 s56, s81, s80                                    // 000000002690: 80B85051
	s_mul_i32 s56, s56, s74                                    // 000000002694: 92384A38
	s_mov_b32 s18, s56                                         // 000000002698: BE920038
	s_mul_i32 s56, s7, 0x480                                   // 00000000269C: 9238FF07 00000480
	v_lshlrev_b32_e32 v30, 2, v0                               // 0000000026A4: 243C0082
	v_add_u32_e32 v30, s56, v30                                // 0000000026A8: 683C3C38
	s_mul_i32 s56, s7, 0x1420                                  // 0000000026AC: 9238FF07 00001420
	s_add_u32 s34, 0, s56                                      // 0000000026B4: 80223880
	s_add_u32 s35, 0x5080, s34                                 // 0000000026B8: 802322FF 00005080
	s_add_u32 s36, 0x5080, s35                                 // 0000000026C0: 802423FF 00005080
	v_lshrrev_b32_e32 v20, 4, v0                               // 0000000026C8: 20280084
	v_lshlrev_b32_e32 v21, 2, v20                              // 0000000026CC: 242A2882
	v_and_b32_e32 v20, 15, v0                                  // 0000000026D0: 2628008F
	v_lshrrev_b32_e32 v22, 2, v20                              // 0000000026D4: 202C2882
	v_mul_i32_i24_e32 v22, 0x140, v22                          // 0000000026D8: 0C2C2CFF 00000140
	v_add_u32_e32 v21, v22, v21                                // 0000000026E0: 682A2B16
	v_and_b32_e32 v20, 3, v0                                   // 0000000026E4: 26280083
	v_mul_i32_i24_e32 v22, 0x508, v20                          // 0000000026E8: 0C2C28FF 00000508
	v_add_u32_e32 v21, v22, v21                                // 0000000026F0: 682A2B16
	v_lshlrev_b32_e32 v29, 2, v21                              // 0000000026F4: 243A2A82
	s_mov_b32 m0, s34                                          // 0000000026F8: BEFC0022
	v_add_u32_e32 v28, 0, v30                                  // 0000000026FC: 68383C80
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002700: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002708: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002710: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002718: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002720: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002728: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002730: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002738: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002740: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002748: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002750: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002758: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002760: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002768: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002770: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002778: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002780: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002788: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002790: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002798: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 0000000027A0: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 0000000027A8: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 0000000027B0: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 0000000027B8: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 0000000027C0: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 0000000027C8: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 0000000027D0: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 0000000027D8: 683838FF 00001200
	s_mov_b32 m0, s35                                          // 0000000027E0: BEFC0023
	v_add_u32_e32 v28, 0x4800, v30                             // 0000000027E4: 68383CFF 00004800
	buffer_load_dword v28, s[16:19], 0 offen lds               // 0000000027EC: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 0000000027F4: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 0000000027FC: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002804: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 00000000280C: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002814: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 00000000281C: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002824: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 00000000282C: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002834: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 00000000283C: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002844: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 00000000284C: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002854: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 00000000285C: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002864: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 00000000286C: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002874: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 00000000287C: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002884: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 00000000288C: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002894: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 00000000289C: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 0000000028A4: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 0000000028AC: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 0000000028B4: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 0000000028BC: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 0000000028C4: 683838FF 00001200
	s_waitcnt vmcnt(20)                                        // 0000000028CC: BF8C4F74
	s_barrier                                                  // 0000000028D0: BF8A0000
	s_mov_b32 m0, s36                                          // 0000000028D4: BEFC0024
	v_add_u32_e32 v28, 0x9000, v30                             // 0000000028D8: 68383CFF 00009000
	buffer_load_dword v28, s[16:19], 0 offen lds               // 0000000028E0: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 0000000028E8: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 0000000028F0: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 0000000028F8: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002900: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002908: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002910: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002918: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002920: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002928: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002930: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002938: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002940: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002948: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002950: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002958: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002960: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002968: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002970: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002978: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002980: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002988: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002990: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002998: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 0000000029A0: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 0000000029A8: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 0000000029B0: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 0000000029B8: 683838FF 00001200
	s_cmp_eq_i32 s7, 0                                         // 0000000029C0: BF008007
	s_cbranch_scc0 label_01D7                                  // 0000000029C4: BF840025
	ds_read_b128 a[0:3], v29                                   // 0000000029C8: DBFE0000 0000001D
	ds_read_b128 a[4:7], v29 offset:64                         // 0000000029D0: DBFE0040 0400001D
	ds_read_b128 a[8:11], v29 offset:128                       // 0000000029D8: DBFE0080 0800001D
	ds_read_b128 a[12:15], v29 offset:192                      // 0000000029E0: DBFE00C0 0C00001D
	ds_read_b128 a[16:19], v29 offset:256                      // 0000000029E8: DBFE0100 1000001D
	ds_read_b128 a[20:23], v29 offset:320                      // 0000000029F0: DBFE0140 1400001D
	ds_read_b128 a[24:27], v29 offset:384                      // 0000000029F8: DBFE0180 1800001D
	ds_read_b128 a[28:31], v29 offset:448                      // 000000002A00: DBFE01C0 1C00001D
	ds_read_b128 a[32:35], v29 offset:512                      // 000000002A08: DBFE0200 2000001D
	ds_read_b128 a[36:39], v29 offset:576                      // 000000002A10: DBFE0240 2400001D
	ds_read_b128 a[40:43], v29 offset:640                      // 000000002A18: DBFE0280 2800001D
	ds_read_b128 a[44:47], v29 offset:704                      // 000000002A20: DBFE02C0 2C00001D
	ds_read_b128 a[48:51], v29 offset:768                      // 000000002A28: DBFE0300 3000001D
	ds_read_b128 a[52:55], v29 offset:832                      // 000000002A30: DBFE0340 3400001D
	ds_read_b128 a[56:59], v29 offset:896                      // 000000002A38: DBFE0380 3800001D
	ds_read_b128 a[60:63], v29 offset:960                      // 000000002A40: DBFE03C0 3C00001D
	ds_read_b128 a[64:67], v29 offset:1024                     // 000000002A48: DBFE0400 4000001D
	ds_read_b128 a[68:71], v29 offset:1088                     // 000000002A50: DBFE0440 4400001D
	s_waitcnt lgkmcnt(0)                                       // 000000002A58: BF8CC07F

0000000000002a5c <label_01D7>:
	s_waitcnt vmcnt(20)                                        // 000000002A5C: BF8C4F74
	s_barrier                                                  // 000000002A60: BF8A0000
	s_mov_b32 m0, s34                                          // 000000002A64: BEFC0022
	v_add_u32_e32 v28, 0xd800, v30                             // 000000002A68: 68383CFF 0000D800
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002A70: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002A78: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002A80: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002A88: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002A90: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002A98: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002AA0: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002AA8: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002AB0: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002AB8: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002AC0: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002AC8: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002AD0: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002AD8: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002AE0: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002AE8: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002AF0: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002AF8: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002B00: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002B08: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002B10: 683838FF 00001200
	buffer_load_dword v28, s[16:19], 0 offen lds               // 000000002B18: E0511000 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds    // 000000002B20: E0511100 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds    // 000000002B28: E0511200 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds    // 000000002B30: E0511300 8004001C
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds   // 000000002B38: E0511400 8004001C
	s_add_u32 m0, m0, 0x500                                    // 000000002B40: 807CFF7C 00000500
	v_add_u32_e32 v28, 0x1200, v28                             // 000000002B48: 683838FF 00001200
	s_cmp_eq_i32 s7, 1                                         // 000000002B50: BF008107
	s_cbranch_scc0 label_023B                                  // 000000002B54: BF840025
	ds_read_b128 a[0:3], v29 offset:20608                      // 000000002B58: DBFE5080 0000001D
	ds_read_b128 a[4:7], v29 offset:20672                      // 000000002B60: DBFE50C0 0400001D
	ds_read_b128 a[8:11], v29 offset:20736                     // 000000002B68: DBFE5100 0800001D
	ds_read_b128 a[12:15], v29 offset:20800                    // 000000002B70: DBFE5140 0C00001D
	ds_read_b128 a[16:19], v29 offset:20864                    // 000000002B78: DBFE5180 1000001D
	ds_read_b128 a[20:23], v29 offset:20928                    // 000000002B80: DBFE51C0 1400001D
	ds_read_b128 a[24:27], v29 offset:20992                    // 000000002B88: DBFE5200 1800001D
	ds_read_b128 a[28:31], v29 offset:21056                    // 000000002B90: DBFE5240 1C00001D
	ds_read_b128 a[32:35], v29 offset:21120                    // 000000002B98: DBFE5280 2000001D
	ds_read_b128 a[36:39], v29 offset:21184                    // 000000002BA0: DBFE52C0 2400001D
	ds_read_b128 a[40:43], v29 offset:21248                    // 000000002BA8: DBFE5300 2800001D
	ds_read_b128 a[44:47], v29 offset:21312                    // 000000002BB0: DBFE5340 2C00001D
	ds_read_b128 a[48:51], v29 offset:21376                    // 000000002BB8: DBFE5380 3000001D
	ds_read_b128 a[52:55], v29 offset:21440                    // 000000002BC0: DBFE53C0 3400001D
	ds_read_b128 a[56:59], v29 offset:21504                    // 000000002BC8: DBFE5400 3800001D
	ds_read_b128 a[60:63], v29 offset:21568                    // 000000002BD0: DBFE5440 3C00001D
	ds_read_b128 a[64:67], v29 offset:21632                    // 000000002BD8: DBFE5480 4000001D
	ds_read_b128 a[68:71], v29 offset:21696                    // 000000002BE0: DBFE54C0 4400001D
	s_waitcnt lgkmcnt(0)                                       // 000000002BE8: BF8CC07F

0000000000002bec <label_023B>:
	s_waitcnt vmcnt(20)                                        // 000000002BEC: BF8C4F74
	s_barrier                                                  // 000000002BF0: BF8A0000
	s_cmp_eq_i32 s7, 2                                         // 000000002BF4: BF008207
	s_cbranch_scc0 label_0264                                  // 000000002BF8: BF840025
	ds_read_b128 a[0:3], v29 offset:41216                      // 000000002BFC: DBFEA100 0000001D
	ds_read_b128 a[4:7], v29 offset:41280                      // 000000002C04: DBFEA140 0400001D
	ds_read_b128 a[8:11], v29 offset:41344                     // 000000002C0C: DBFEA180 0800001D
	ds_read_b128 a[12:15], v29 offset:41408                    // 000000002C14: DBFEA1C0 0C00001D
	ds_read_b128 a[16:19], v29 offset:41472                    // 000000002C1C: DBFEA200 1000001D
	ds_read_b128 a[20:23], v29 offset:41536                    // 000000002C24: DBFEA240 1400001D
	ds_read_b128 a[24:27], v29 offset:41600                    // 000000002C2C: DBFEA280 1800001D
	ds_read_b128 a[28:31], v29 offset:41664                    // 000000002C34: DBFEA2C0 1C00001D
	ds_read_b128 a[32:35], v29 offset:41728                    // 000000002C3C: DBFEA300 2000001D
	ds_read_b128 a[36:39], v29 offset:41792                    // 000000002C44: DBFEA340 2400001D
	ds_read_b128 a[40:43], v29 offset:41856                    // 000000002C4C: DBFEA380 2800001D
	ds_read_b128 a[44:47], v29 offset:41920                    // 000000002C54: DBFEA3C0 2C00001D
	ds_read_b128 a[48:51], v29 offset:41984                    // 000000002C5C: DBFEA400 3000001D
	ds_read_b128 a[52:55], v29 offset:42048                    // 000000002C64: DBFEA440 3400001D
	ds_read_b128 a[56:59], v29 offset:42112                    // 000000002C6C: DBFEA480 3800001D
	ds_read_b128 a[60:63], v29 offset:42176                    // 000000002C74: DBFEA4C0 3C00001D
	ds_read_b128 a[64:67], v29 offset:42240                    // 000000002C7C: DBFEA500 4000001D
	ds_read_b128 a[68:71], v29 offset:42304                    // 000000002C84: DBFEA540 4400001D
	s_waitcnt lgkmcnt(0)                                       // 000000002C8C: BF8CC07F

0000000000002c90 <label_0264>:
	s_waitcnt vmcnt(0)                                         // 000000002C90: BF8C0F70
	s_barrier                                                  // 000000002C94: BF8A0000
	s_cmp_eq_i32 s7, 3                                         // 000000002C98: BF008307
	s_cbranch_scc0 label_028D                                  // 000000002C9C: BF840025
	ds_read_b128 a[0:3], v29                                   // 000000002CA0: DBFE0000 0000001D
	ds_read_b128 a[4:7], v29 offset:64                         // 000000002CA8: DBFE0040 0400001D
	ds_read_b128 a[8:11], v29 offset:128                       // 000000002CB0: DBFE0080 0800001D
	ds_read_b128 a[12:15], v29 offset:192                      // 000000002CB8: DBFE00C0 0C00001D
	ds_read_b128 a[16:19], v29 offset:256                      // 000000002CC0: DBFE0100 1000001D
	ds_read_b128 a[20:23], v29 offset:320                      // 000000002CC8: DBFE0140 1400001D
	ds_read_b128 a[24:27], v29 offset:384                      // 000000002CD0: DBFE0180 1800001D
	ds_read_b128 a[28:31], v29 offset:448                      // 000000002CD8: DBFE01C0 1C00001D
	ds_read_b128 a[32:35], v29 offset:512                      // 000000002CE0: DBFE0200 2000001D
	ds_read_b128 a[36:39], v29 offset:576                      // 000000002CE8: DBFE0240 2400001D
	ds_read_b128 a[40:43], v29 offset:640                      // 000000002CF0: DBFE0280 2800001D
	ds_read_b128 a[44:47], v29 offset:704                      // 000000002CF8: DBFE02C0 2C00001D
	ds_read_b128 a[48:51], v29 offset:768                      // 000000002D00: DBFE0300 3000001D
	ds_read_b128 a[52:55], v29 offset:832                      // 000000002D08: DBFE0340 3400001D
	ds_read_b128 a[56:59], v29 offset:896                      // 000000002D10: DBFE0380 3800001D
	ds_read_b128 a[60:63], v29 offset:960                      // 000000002D18: DBFE03C0 3C00001D
	ds_read_b128 a[64:67], v29 offset:1024                     // 000000002D20: DBFE0400 4000001D
	ds_read_b128 a[68:71], v29 offset:1088                     // 000000002D28: DBFE0440 4400001D
	s_waitcnt lgkmcnt(0)                                       // 000000002D30: BF8CC07F

0000000000002d34 <label_028D>:
	s_waitcnt vmcnt(0)                                         // 000000002D34: BF8C0F70
	s_barrier                                                  // 000000002D38: BF8A0000
	s_mov_b32 s52, 0x7060302                                   // 000000002D3C: BEB400FF 07060302
	s_mov_b32 s53, 0x5040100                                   // 000000002D44: BEB500FF 05040100
	s_mov_b32 s6, 0x3fb8aa3b                                   // 000000002D4C: BE8600FF 3FB8AA3B
	v_mov_b32_e32 v21, s6                                      // 000000002D54: 7E2A0206
	v_mov_b32_e32 v20, s64                                     // 000000002D58: 7E280240
	v_mul_f32_e32 v20, s6, v20                                 // 000000002D5C: 0A282806
	v_rcp_f32_e32 v21, v21                                     // 000000002D60: 7E2A4515
	v_mov_b32_e32 v12, 0xff7fffff                              // 000000002D64: 7E1802FF FF7FFFFF
	v_mov_b32_e32 v13, 0xff7fffff                              // 000000002D6C: 7E1A02FF FF7FFFFF
	v_mov_b32_e32 v16, 0                                       // 000000002D74: 7E200280
	v_mov_b32_e32 v17, 0                                       // 000000002D78: 7E220280
	v_mov_b32_e32 v14, 0                                       // 000000002D7C: 7E1C0280
	v_mov_b32_e32 v15, 0                                       // 000000002D80: 7E1E0280
	v_mov_b32_e32 v9, s68                                      // 000000002D84: 7E120244
	v_readfirstlane_b32 s5, v20                                // 000000002D88: 7E0A0514
	v_readfirstlane_b32 s63, v21                               // 000000002D8C: 7E7E0515
	v_mov_b32_e32 v20, s43                                     // 000000002D90: 7E28022B
	v_mul_f32_e32 v21, s5, v20                                 // 000000002D94: 0A2A2805
	v_mul_f32_e32 v23, s64, v20                                // 000000002D98: 0A2E2840
	v_readfirstlane_b32 s5, v21                                // 000000002D9C: 7E0A0515
	v_readfirstlane_b32 s64, v23                               // 000000002DA0: 7E800517
	v_and_b32_e32 v2, 15, v0                                   // 000000002DA4: 2604008F
	v_lshlrev_b32_e32 v2, 2, v2                                // 000000002DA8: 24040482
	s_mul_i32 s56, 0x100, s7                                   // 000000002DAC: 923807FF 00000100
	v_add_u32_e32 v2, s56, v2                                  // 000000002DB4: 68040438
	v_lshlrev_b32_e32 v3, 2, v0                                // 000000002DB8: 24060082
	s_mul_i32 s56, 0x100, s7                                   // 000000002DBC: 923807FF 00000100
	v_add_u32_e32 v3, s56, v3                                  // 000000002DC4: 68060638
	v_and_b32_e32 v20, 31, v0                                  // 000000002DC8: 2628009F
	v_lshlrev_b32_e32 v20, 3, v20                              // 000000002DCC: 24282883
	v_lshrrev_b32_e32 v21, 5, v0                               // 000000002DD0: 202A0085
	v_mul_i32_i24_e32 v21, 0x900, v21                          // 000000002DD4: 0C2A2AFF 00000900
	v_add_u32_e32 v20, v21, v20                                // 000000002DDC: 68282915
	s_mul_i32 s56, 0x1220, s7                                  // 000000002DE0: 923807FF 00001220
	v_add_u32_e32 v176, s56, v20                               // 000000002DE8: 69602838
	v_and_b32_e32 v20, 15, v0                                  // 000000002DEC: 2628008F
	v_lshlrev_b32_e32 v1, 2, v20                               // 000000002DF0: 24022882
	s_mul_i32 s34, s7, 0x1220                                  // 000000002DF4: 9222FF07 00001220
	s_add_u32 s34, 0, s34                                      // 000000002DFC: 80222280
	s_add_u32 s35, 0x900, s34                                  // 000000002E00: 802322FF 00000900
	s_add_u32 s36, 0x4880, s34                                 // 000000002E08: 802422FF 00004880
	s_add_u32 s37, 0x4880, s35                                 // 000000002E10: 802523FF 00004880
	s_waitcnt vmcnt(0)                                         // 000000002E18: BF8C0F70
	v_mov_b32_e32 v234, 0xffff0000                             // 000000002E1C: 7FD402FF FFFF0000
	v_mov_b32_e32 v235, 0x7fff0000                             // 000000002E24: 7FD602FF 7FFF0000
	v_mov_b32_e32 v236, 0x7fff                                 // 000000002E2C: 7FD802FF 00007FFF
	v_mul_u32_u24_e32 v18, v10, v9                             // 000000002E34: 1024130A
	v_add_u32_e32 v18, v18, v1                                 // 000000002E38: 68240312
	s_mov_b32 m0, s35                                          // 000000002E3C: BEFC0023
	buffer_load_dword v178, v18, s[20:23], 0 offen             // 000000002E40: E0501000 8005B212
	buffer_load_dword v184, v18, s[20:23], 0 offen offset:64   // 000000002E48: E0501040 8005B812
	buffer_load_dword v190, v18, s[20:23], 0 offen offset:128  // 000000002E50: E0501080 8005BE12
	buffer_load_dword v196, v18, s[20:23], 0 offen offset:192  // 000000002E58: E05010C0 8005C412
	buffer_load_dword v202, v18, s[20:23], 0 offen offset:256  // 000000002E60: E0501100 8005CA12
	buffer_load_dword v208, v18, s[20:23], 0 offen offset:320  // 000000002E68: E0501140 8005D012
	buffer_load_dword v214, v18, s[20:23], 0 offen offset:384  // 000000002E70: E0501180 8005D612
	buffer_load_dword v220, v18, s[20:23], 0 offen offset:448  // 000000002E78: E05011C0 8005DC12
	buffer_load_dword v226, v18, s[20:23], 0 offen offset:512  // 000000002E80: E0501200 8005E212
	s_waitcnt vmcnt(0)                                         // 000000002E88: BF8C0F70
	v_cvt_pk_f32_fp8_sdwa v[180:181], v178 src0_sel:WORD_0     // 000000002E8C: 7F68ACF9 000406B2
	v_cvt_pk_f32_fp8_sdwa v[182:183], v178 src0_sel:WORD_1     // 000000002E94: 7F6CACF9 000506B2
	v_cvt_pk_f32_fp8_sdwa v[186:187], v184 src0_sel:WORD_0     // 000000002E9C: 7F74ACF9 000406B8
	v_cvt_pk_f32_fp8_sdwa v[188:189], v184 src0_sel:WORD_1     // 000000002EA4: 7F78ACF9 000506B8
	v_cvt_pk_f32_fp8_sdwa v[192:193], v190 src0_sel:WORD_0     // 000000002EAC: 7F80ACF9 000406BE
	v_cvt_pk_f32_fp8_sdwa v[194:195], v190 src0_sel:WORD_1     // 000000002EB4: 7F84ACF9 000506BE
	v_cvt_pk_f32_fp8_sdwa v[198:199], v196 src0_sel:WORD_0     // 000000002EBC: 7F8CACF9 000406C4
	v_cvt_pk_f32_fp8_sdwa v[200:201], v196 src0_sel:WORD_1     // 000000002EC4: 7F90ACF9 000506C4
	v_cvt_pk_f32_fp8_sdwa v[204:205], v202 src0_sel:WORD_0     // 000000002ECC: 7F98ACF9 000406CA
	v_cvt_pk_f32_fp8_sdwa v[206:207], v202 src0_sel:WORD_1     // 000000002ED4: 7F9CACF9 000506CA
	v_cvt_pk_f32_fp8_sdwa v[210:211], v208 src0_sel:WORD_0     // 000000002EDC: 7FA4ACF9 000406D0
	v_cvt_pk_f32_fp8_sdwa v[212:213], v208 src0_sel:WORD_1     // 000000002EE4: 7FA8ACF9 000506D0
	v_cvt_pk_f32_fp8_sdwa v[216:217], v214 src0_sel:WORD_0     // 000000002EEC: 7FB0ACF9 000406D6
	v_cvt_pk_f32_fp8_sdwa v[218:219], v214 src0_sel:WORD_1     // 000000002EF4: 7FB4ACF9 000506D6
	v_cvt_pk_f32_fp8_sdwa v[222:223], v220 src0_sel:WORD_0     // 000000002EFC: 7FBCACF9 000406DC
	v_cvt_pk_f32_fp8_sdwa v[224:225], v220 src0_sel:WORD_1     // 000000002F04: 7FC0ACF9 000506DC
	v_cvt_pk_f32_fp8_sdwa v[228:229], v226 src0_sel:WORD_0     // 000000002F0C: 7FC8ACF9 000406E2
	v_cvt_pk_f32_fp8_sdwa v[230:231], v226 src0_sel:WORD_1     // 000000002F14: 7FCCACF9 000506E2
	v_perm_b32 v180, v181, v180, s52                           // 000000002F1C: D1ED00B4 00D369B5
	v_perm_b32 v181, v183, v182, s52                           // 000000002F24: D1ED00B5 00D36DB7
	v_perm_b32 v186, v187, v186, s52                           // 000000002F2C: D1ED00BA 00D375BB
	v_perm_b32 v187, v189, v188, s52                           // 000000002F34: D1ED00BB 00D379BD
	v_perm_b32 v192, v193, v192, s52                           // 000000002F3C: D1ED00C0 00D381C1
	v_perm_b32 v193, v195, v194, s52                           // 000000002F44: D1ED00C1 00D385C3
	v_perm_b32 v198, v199, v198, s52                           // 000000002F4C: D1ED00C6 00D38DC7
	v_perm_b32 v199, v201, v200, s52                           // 000000002F54: D1ED00C7 00D391C9
	v_perm_b32 v204, v205, v204, s52                           // 000000002F5C: D1ED00CC 00D399CD
	v_perm_b32 v205, v207, v206, s52                           // 000000002F64: D1ED00CD 00D39DCF
	v_perm_b32 v210, v211, v210, s52                           // 000000002F6C: D1ED00D2 00D3A5D3
	v_perm_b32 v211, v213, v212, s52                           // 000000002F74: D1ED00D3 00D3A9D5
	v_perm_b32 v216, v217, v216, s52                           // 000000002F7C: D1ED00D8 00D3B1D9
	v_perm_b32 v217, v219, v218, s52                           // 000000002F84: D1ED00D9 00D3B5DB
	v_perm_b32 v222, v223, v222, s52                           // 000000002F8C: D1ED00DE 00D3BDDF
	v_perm_b32 v223, v225, v224, s52                           // 000000002F94: D1ED00DF 00D3C1E1
	v_perm_b32 v228, v229, v228, s52                           // 000000002F9C: D1ED00E4 00D3C9E5
	v_perm_b32 v229, v231, v230, s52                           // 000000002FA4: D1ED00E5 00D3CDE7
	ds_write_b64 v176, v[180:181]                              // 000000002FAC: D89A0000 0000B4B0
	ds_write_b64 v176, v[186:187] offset:256                   // 000000002FB4: D89A0100 0000BAB0
	ds_write_b64 v176, v[192:193] offset:512                   // 000000002FBC: D89A0200 0000C0B0
	ds_write_b64 v176, v[198:199] offset:768                   // 000000002FC4: D89A0300 0000C6B0
	ds_write_b64 v176, v[204:205] offset:1024                  // 000000002FCC: D89A0400 0000CCB0
	ds_write_b64 v176, v[210:211] offset:1280                  // 000000002FD4: D89A0500 0000D2B0
	ds_write_b64 v176, v[216:217] offset:1536                  // 000000002FDC: D89A0600 0000D8B0
	ds_write_b64 v176, v[222:223] offset:1792                  // 000000002FE4: D89A0700 0000DEB0
	ds_write_b64 v176, v[228:229] offset:2048                  // 000000002FEC: D89A0800 0000E4B0
	buffer_load_dword v10, v8, s[24:27], 0 offen               // 000000002FF4: E0501000 80060A08
	v_add_u32_e32 v8, s73, v8                                  // 000000002FFC: 68101049
	v_mov_b32_e32 v40, 0                                       // 000000003000: 7E500280
	v_mov_b32_e32 v41, 0                                       // 000000003004: 7E520280
	v_mov_b32_e32 v42, 0                                       // 000000003008: 7E540280
	v_mov_b32_e32 v43, 0                                       // 00000000300C: 7E560280
	v_mov_b32_e32 v44, 0                                       // 000000003010: 7E580280
	v_mov_b32_e32 v45, 0                                       // 000000003014: 7E5A0280
	v_mov_b32_e32 v46, 0                                       // 000000003018: 7E5C0280
	v_mov_b32_e32 v47, 0                                       // 00000000301C: 7E5E0280
	v_mov_b32_e32 v48, 0                                       // 000000003020: 7E600280
	v_mov_b32_e32 v49, 0                                       // 000000003024: 7E620280
	v_mov_b32_e32 v50, 0                                       // 000000003028: 7E640280
	v_mov_b32_e32 v51, 0                                       // 00000000302C: 7E660280
	v_mov_b32_e32 v52, 0                                       // 000000003030: 7E680280
	v_mov_b32_e32 v53, 0                                       // 000000003034: 7E6A0280
	v_mov_b32_e32 v54, 0                                       // 000000003038: 7E6C0280
	v_mov_b32_e32 v55, 0                                       // 00000000303C: 7E6E0280
	v_mov_b32_e32 v56, 0                                       // 000000003040: 7E700280
	v_mov_b32_e32 v57, 0                                       // 000000003044: 7E720280
	v_mov_b32_e32 v58, 0                                       // 000000003048: 7E740280
	v_mov_b32_e32 v59, 0                                       // 00000000304C: 7E760280
	v_mov_b32_e32 v60, 0                                       // 000000003050: 7E780280
	v_mov_b32_e32 v61, 0                                       // 000000003054: 7E7A0280
	v_mov_b32_e32 v62, 0                                       // 000000003058: 7E7C0280
	v_mov_b32_e32 v63, 0                                       // 00000000305C: 7E7E0280
	v_mov_b32_e32 v64, 0                                       // 000000003060: 7E800280
	v_mov_b32_e32 v65, 0                                       // 000000003064: 7E820280
	v_mov_b32_e32 v66, 0                                       // 000000003068: 7E840280
	v_mov_b32_e32 v67, 0                                       // 00000000306C: 7E860280
	v_mov_b32_e32 v68, 0                                       // 000000003070: 7E880280
	v_mov_b32_e32 v69, 0                                       // 000000003074: 7E8A0280
	v_mov_b32_e32 v70, 0                                       // 000000003078: 7E8C0280
	v_mov_b32_e32 v71, 0                                       // 00000000307C: 7E8E0280
	v_mov_b32_e32 v72, 0                                       // 000000003080: 7E900280
	v_mov_b32_e32 v73, 0                                       // 000000003084: 7E920280
	v_mov_b32_e32 v74, 0                                       // 000000003088: 7E940280
	v_mov_b32_e32 v75, 0                                       // 00000000308C: 7E960280
	v_mov_b32_e32 v76, 0                                       // 000000003090: 7E980280
	v_mov_b32_e32 v77, 0                                       // 000000003094: 7E9A0280
	v_mov_b32_e32 v78, 0                                       // 000000003098: 7E9C0280
	v_mov_b32_e32 v79, 0                                       // 00000000309C: 7E9E0280
	v_mov_b32_e32 v80, 0                                       // 0000000030A0: 7EA00280
	v_mov_b32_e32 v81, 0                                       // 0000000030A4: 7EA20280
	v_mov_b32_e32 v82, 0                                       // 0000000030A8: 7EA40280
	v_mov_b32_e32 v83, 0                                       // 0000000030AC: 7EA60280
	v_mov_b32_e32 v84, 0                                       // 0000000030B0: 7EA80280
	v_mov_b32_e32 v85, 0                                       // 0000000030B4: 7EAA0280
	v_mov_b32_e32 v86, 0                                       // 0000000030B8: 7EAC0280
	v_mov_b32_e32 v87, 0                                       // 0000000030BC: 7EAE0280
	v_mov_b32_e32 v88, 0                                       // 0000000030C0: 7EB00280
	v_mov_b32_e32 v89, 0                                       // 0000000030C4: 7EB20280
	v_mov_b32_e32 v90, 0                                       // 0000000030C8: 7EB40280
	v_mov_b32_e32 v91, 0                                       // 0000000030CC: 7EB60280
	v_mov_b32_e32 v92, 0                                       // 0000000030D0: 7EB80280
	v_mov_b32_e32 v93, 0                                       // 0000000030D4: 7EBA0280
	v_mov_b32_e32 v94, 0                                       // 0000000030D8: 7EBC0280
	v_mov_b32_e32 v95, 0                                       // 0000000030DC: 7EBE0280
	v_mov_b32_e32 v96, 0                                       // 0000000030E0: 7EC00280
	v_mov_b32_e32 v97, 0                                       // 0000000030E4: 7EC20280
	v_mov_b32_e32 v98, 0                                       // 0000000030E8: 7EC40280
	v_mov_b32_e32 v99, 0                                       // 0000000030EC: 7EC60280
	v_mov_b32_e32 v100, 0                                      // 0000000030F0: 7EC80280
	v_mov_b32_e32 v101, 0                                      // 0000000030F4: 7ECA0280
	v_mov_b32_e32 v102, 0                                      // 0000000030F8: 7ECC0280
	v_mov_b32_e32 v103, 0                                      // 0000000030FC: 7ECE0280
	v_mov_b32_e32 v104, 0                                      // 000000003100: 7ED00280
	v_mov_b32_e32 v105, 0                                      // 000000003104: 7ED20280
	v_mov_b32_e32 v106, 0                                      // 000000003108: 7ED40280
	v_mov_b32_e32 v107, 0                                      // 00000000310C: 7ED60280
	v_mov_b32_e32 v108, 0                                      // 000000003110: 7ED80280
	v_mov_b32_e32 v109, 0                                      // 000000003114: 7EDA0280
	v_mov_b32_e32 v110, 0                                      // 000000003118: 7EDC0280
	v_mov_b32_e32 v111, 0                                      // 00000000311C: 7EDE0280
	v_mov_b32_e32 v112, 0                                      // 000000003120: 7EE00280
	v_mov_b32_e32 v113, 0                                      // 000000003124: 7EE20280
	v_mov_b32_e32 v114, 0                                      // 000000003128: 7EE40280
	v_mov_b32_e32 v115, 0                                      // 00000000312C: 7EE60280
	v_mov_b32_e32 v116, 0                                      // 000000003130: 7EE80280
	v_mov_b32_e32 v117, 0                                      // 000000003134: 7EEA0280
	v_mov_b32_e32 v118, 0                                      // 000000003138: 7EEC0280
	v_mov_b32_e32 v119, 0                                      // 00000000313C: 7EEE0280
	v_mov_b32_e32 v120, 0                                      // 000000003140: 7EF00280
	v_mov_b32_e32 v121, 0                                      // 000000003144: 7EF20280
	v_mov_b32_e32 v122, 0                                      // 000000003148: 7EF40280
	v_mov_b32_e32 v123, 0                                      // 00000000314C: 7EF60280
	v_mov_b32_e32 v124, 0                                      // 000000003150: 7EF80280
	v_mov_b32_e32 v125, 0                                      // 000000003154: 7EFA0280
	v_mov_b32_e32 v126, 0                                      // 000000003158: 7EFC0280
	v_mov_b32_e32 v127, 0                                      // 00000000315C: 7EFE0280
	v_mov_b32_e32 v128, 0                                      // 000000003160: 7F000280
	v_mov_b32_e32 v129, 0                                      // 000000003164: 7F020280
	v_mov_b32_e32 v130, 0                                      // 000000003168: 7F040280
	v_mov_b32_e32 v131, 0                                      // 00000000316C: 7F060280
	v_mov_b32_e32 v132, 0                                      // 000000003170: 7F080280
	v_mov_b32_e32 v133, 0                                      // 000000003174: 7F0A0280
	v_mov_b32_e32 v134, 0                                      // 000000003178: 7F0C0280
	v_mov_b32_e32 v135, 0                                      // 00000000317C: 7F0E0280
	v_mov_b32_e32 v136, 0                                      // 000000003180: 7F100280
	v_mov_b32_e32 v137, 0                                      // 000000003184: 7F120280
	v_mov_b32_e32 v138, 0                                      // 000000003188: 7F140280
	v_mov_b32_e32 v139, 0                                      // 00000000318C: 7F160280
	v_mov_b32_e32 v140, 0                                      // 000000003190: 7F180280
	v_mov_b32_e32 v141, 0                                      // 000000003194: 7F1A0280
	v_mov_b32_e32 v142, 0                                      // 000000003198: 7F1C0280
	v_mov_b32_e32 v143, 0                                      // 00000000319C: 7F1E0280
	v_mov_b32_e32 v144, 0                                      // 0000000031A0: 7F200280
	v_mov_b32_e32 v145, 0                                      // 0000000031A4: 7F220280
	v_mov_b32_e32 v146, 0                                      // 0000000031A8: 7F240280
	v_mov_b32_e32 v147, 0                                      // 0000000031AC: 7F260280
	v_mov_b32_e32 v148, 0                                      // 0000000031B0: 7F280280
	v_mov_b32_e32 v149, 0                                      // 0000000031B4: 7F2A0280
	v_mov_b32_e32 v150, 0                                      // 0000000031B8: 7F2C0280
	v_mov_b32_e32 v151, 0                                      // 0000000031BC: 7F2E0280
	v_mov_b32_e32 v152, 0                                      // 0000000031C0: 7F300280
	v_mov_b32_e32 v153, 0                                      // 0000000031C4: 7F320280
	v_mov_b32_e32 v154, 0                                      // 0000000031C8: 7F340280
	v_mov_b32_e32 v155, 0                                      // 0000000031CC: 7F360280
	v_mov_b32_e32 v156, 0                                      // 0000000031D0: 7F380280
	v_mov_b32_e32 v157, 0                                      // 0000000031D4: 7F3A0280
	v_mov_b32_e32 v158, 0                                      // 0000000031D8: 7F3C0280
	v_mov_b32_e32 v159, 0                                      // 0000000031DC: 7F3E0280
	v_mov_b32_e32 v160, 0                                      // 0000000031E0: 7F400280
	v_mov_b32_e32 v161, 0                                      // 0000000031E4: 7F420280
	v_mov_b32_e32 v162, 0                                      // 0000000031E8: 7F440280
	v_mov_b32_e32 v163, 0                                      // 0000000031EC: 7F460280
	v_mov_b32_e32 v164, 0                                      // 0000000031F0: 7F480280
	v_mov_b32_e32 v165, 0                                      // 0000000031F4: 7F4A0280
	v_mov_b32_e32 v166, 0                                      // 0000000031F8: 7F4C0280
	v_mov_b32_e32 v167, 0                                      // 0000000031FC: 7F4E0280
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000003200: 20280084
	v_mul_i32_i24_e32 v21, 4, v20                              // 000000003204: 0C2A2884
	v_and_b32_e32 v20, 15, v0                                  // 000000003208: 2628008F
	v_and_b32_e32 v22, 3, v20                                  // 00000000320C: 262C2883
	v_mul_i32_i24_e32 v22, 0x488, v22                          // 000000003210: 0C2C2CFF 00000488
	v_add_u32_e32 v4, v22, v21                                 // 000000003218: 68082B16
	v_lshrrev_b32_e32 v20, 2, v20                              // 00000000321C: 20282882
	v_and_b32_e32 v21, 1, v20                                  // 000000003220: 262A2881
	v_mul_i32_i24_e32 v21, 32, v21                             // 000000003224: 0C2A2AA0
	v_add_u32_e32 v4, v4, v21                                  // 000000003228: 68082B04
	v_and_b32_e32 v21, 2, v20                                  // 00000000322C: 262A2882
	v_mul_i32_i24_e32 v21, 0x120, v21                          // 000000003230: 0C2A2AFF 00000120
	v_add_u32_e32 v4, v4, v21                                  // 000000003238: 68082B04
	v_lshlrev_b32_e32 v4, 2, v4                                // 00000000323C: 24080882
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000003240: 20280084
	v_and_b32_e32 v21, 1, v20                                  // 000000003244: 262A2881
	v_mul_i32_i24_e32 v5, 32, v21                              // 000000003248: 0C0A2AA0
	v_and_b32_e32 v21, 2, v20                                  // 00000000324C: 262A2882
	v_mul_i32_i24_e32 v21, 0x120, v21                          // 000000003250: 0C2A2AFF 00000120
	v_add_u32_e32 v5, v5, v21                                  // 000000003258: 680A2B05
	v_and_b32_e32 v20, 15, v0                                  // 00000000325C: 2628008F
	v_mul_i32_i24_e32 v21, 2, v20                              // 000000003260: 0C2A2882
	v_add_u32_e32 v5, v5, v21                                  // 000000003264: 680A2B05
	s_mul_i32 s56, 64, s7                                      // 000000003268: 923807C0
	v_add_u32_e64 v5, v5, s56                                  // 00000000326C: D1340005 00007105
	v_lshlrev_b32_e32 v5, 2, v5                                // 000000003274: 240A0A82
	v_lshlrev_b32_e32 v6, 2, v0                                // 000000003278: 240C0082
	s_mul_i32 s56, 0x200, s7                                   // 00000000327C: 923807FF 00000200
	v_add_u32_e64 v6, v6, s56                                  // 000000003284: D1340006 00007106
	v_lshlrev_b32_e32 v6, 2, v6                                // 00000000328C: 240C0C82
	v_lshlrev_b32_e32 v7, 4, v0                                // 000000003290: 240E0084
	v_mul_u32_u24_e32 v18, v11, v9                             // 000000003294: 1024130B
	v_add_u32_e32 v18, v18, v1                                 // 000000003298: 68240312
	s_mov_b32 m0, s37                                          // 00000000329C: BEFC0025
	buffer_load_dword v11, v8, s[24:27], 0 offen               // 0000000032A0: E0501000 80060B08
	v_add_u32_e32 v8, s73, v8                                  // 0000000032A8: 68101049
	buffer_load_dword v179, v18, s[20:23], 0 offen             // 0000000032AC: E0501000 8005B312
	buffer_load_dword v185, v18, s[20:23], 0 offen offset:64   // 0000000032B4: E0501040 8005B912
	buffer_load_dword v191, v18, s[20:23], 0 offen offset:128  // 0000000032BC: E0501080 8005BF12
	buffer_load_dword v197, v18, s[20:23], 0 offen offset:192  // 0000000032C4: E05010C0 8005C512
	buffer_load_dword v203, v18, s[20:23], 0 offen offset:256  // 0000000032CC: E0501100 8005CB12
	buffer_load_dword v209, v18, s[20:23], 0 offen offset:320  // 0000000032D4: E0501140 8005D112
	buffer_load_dword v215, v18, s[20:23], 0 offen offset:384  // 0000000032DC: E0501180 8005D712
	buffer_load_dword v221, v18, s[20:23], 0 offen offset:448  // 0000000032E4: E05011C0 8005DD12
	buffer_load_dword v227, v18, s[20:23], 0 offen offset:512  // 0000000032EC: E0501200 8005E312
	s_waitcnt vmcnt(10) lgkmcnt(0)                             // 0000000032F4: BF8C007A
	s_barrier                                                  // 0000000032F8: BF8A0000
	v_mul_u32_u24_e32 v18, v10, v9                             // 0000000032FC: 1024130A
	v_add_u32_e32 v18, v18, v1                                 // 000000003300: 68240312
	s_mov_b32 m0, s35                                          // 000000003304: BEFC0023
	ds_read_b128 a[144:147], v4                                // 000000003308: DBFE0000 90000004
	ds_read_b128 a[148:151], v4 offset:64                      // 000000003310: DBFE0040 94000004
	ds_read_b128 a[152:155], v4 offset:256                     // 000000003318: DBFE0100 98000004
	ds_read_b128 a[156:159], v4 offset:320                     // 000000003320: DBFE0140 9C000004
	ds_read_b128 a[160:163], v4 offset:512                     // 000000003328: DBFE0200 A0000004
	ds_read_b128 a[164:167], v4 offset:576                     // 000000003330: DBFE0240 A4000004
	ds_read_b128 a[168:171], v4 offset:768                     // 000000003338: DBFE0300 A8000004
	ds_read_b128 a[172:175], v4 offset:832                     // 000000003340: DBFE0340 AC000004
	ds_read_b128 a[176:179], v4 offset:1024                    // 000000003348: DBFE0400 B0000004
	ds_read_b128 a[180:183], v4 offset:1088                    // 000000003350: DBFE0440 B4000004
	ds_read_b128 a[184:187], v4 offset:1280                    // 000000003358: DBFE0500 B8000004
	ds_read_b128 a[188:191], v4 offset:1344                    // 000000003360: DBFE0540 BC000004
	ds_read_b128 a[192:195], v4 offset:1536                    // 000000003368: DBFE0600 C0000004
	ds_read_b128 a[196:199], v4 offset:1600                    // 000000003370: DBFE0640 C4000004
	ds_read_b128 a[200:203], v4 offset:1792                    // 000000003378: DBFE0700 C8000004
	ds_read_b128 a[204:207], v4 offset:1856                    // 000000003380: DBFE0740 CC000004
	ds_read_b128 a[208:211], v4 offset:2048                    // 000000003388: DBFE0800 D0000004
	ds_read_b128 a[212:215], v4 offset:2112                    // 000000003390: DBFE0840 D4000004
	ds_read_b64 v[20:21], v5                                   // 000000003398: D8EC0000 14000005
	ds_read_b64 v[22:23], v5 offset:4640                       // 0000000033A0: D8EC1220 16000005
	ds_read_b64 v[24:25], v5 offset:9280                       // 0000000033A8: D8EC2440 18000005
	ds_read_b64 v[26:27], v5 offset:13920                      // 0000000033B0: D8EC3660 1A000005
	s_waitcnt lgkmcnt(0)                                       // 0000000033B8: BF8CC07F
	v_perm_b32 v168, v22, v20, s53                             // 0000000033BC: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 0000000033C4: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 0000000033CC: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 0000000033D4: D1ED00AB 00D2311A
	ds_write_b128 v6, v[168:171] offset:37120                  // 0000000033DC: D9BE9100 0000A806
	v_perm_b32 v168, v23, v21, s53                             // 0000000033E4: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 0000000033EC: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 0000000033F4: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 0000000033FC: D1ED00AB 00D2331B
	ds_write_b128 v6, v[168:171] offset:38144                  // 000000003404: D9BE9500 0000A806
	ds_read_b64 v[20:21], v5 offset:1024                       // 00000000340C: D8EC0400 14000005
	ds_read_b64 v[22:23], v5 offset:5664                       // 000000003414: D8EC1620 16000005
	ds_read_b64 v[24:25], v5 offset:10304                      // 00000000341C: D8EC2840 18000005
	ds_read_b64 v[26:27], v5 offset:14944                      // 000000003424: D8EC3A60 1A000005
	s_waitcnt lgkmcnt(0)                                       // 00000000342C: BF8CC07F
	v_perm_b32 v168, v22, v20, s53                             // 000000003430: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 000000003438: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000003440: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 000000003448: D1ED00AB 00D2311A
	ds_write_b128 v6, v[168:171] offset:45312                  // 000000003450: D9BEB100 0000A806
	v_perm_b32 v168, v23, v21, s53                             // 000000003458: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 000000003460: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 000000003468: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 000000003470: D1ED00AB 00D2331B
	ds_write_b128 v6, v[168:171] offset:46336                  // 000000003478: D9BEB500 0000A806
	s_nop 0                                                    // 000000003480: BF800000
	s_cmp_lt_u32 s71, 1                                        // 000000003484: BF0A8147
	s_cbranch_scc1 label_0E98                                  // 000000003488: BF850A35
	s_cmp_lt_i32 s7, 2                                         // 00000000348C: BF048207
	s_cbranch_scc0 label_097F                                  // 000000003490: BF84051A

0000000000003494 <label_0465>:
	s_waitcnt lgkmcnt(4)                                       // 000000003494: BF8CC47F
	s_waitcnt vmcnt(0)                                         // 000000003498: BF8C0F70
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0   // 00000000349C: D3E10020 1A020190
	ds_read_b128 a[176:179], v4 offset:1024                    // 0000000034A4: DBFE0400 B0000004
	ds_read_b128 a[180:183], v4 offset:1088                    // 0000000034AC: DBFE0440 B4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]// 0000000034B4: D3E10020 1C820592
	buffer_load_dword v10, v8, s[24:27], 0 offen               // 0000000034BC: E0501000 80060A08
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]// 0000000034C4: D3E10020 1C820994
	v_cvt_pk_f32_fp8_sdwa v[180:181], v179 src0_sel:WORD_0     // 0000000034CC: 7F68ACF9 000406B3
	v_cvt_pk_f32_fp8_sdwa v[182:183], v179 src0_sel:WORD_1     // 0000000034D4: 7F6CACF9 000506B3
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]// 0000000034DC: D3E10020 1C820D96
	v_cvt_pk_f32_fp8_sdwa v[186:187], v185 src0_sel:WORD_0     // 0000000034E4: 7F74ACF9 000406B9
	v_cvt_pk_f32_fp8_sdwa v[188:189], v185 src0_sel:WORD_1     // 0000000034EC: 7F78ACF9 000506B9
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]// 0000000034F4: D3E10020 1C821198
	ds_read_b128 a[184:187], v4 offset:1280                    // 0000000034FC: DBFE0500 B8000004
	ds_read_b128 a[188:191], v4 offset:1344                    // 000000003504: DBFE0540 BC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]// 00000000350C: D3E10020 1C82159A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]// 000000003514: D3E10020 1C82199C
	v_cvt_pk_f32_fp8_sdwa v[192:193], v191 src0_sel:WORD_0     // 00000000351C: 7F80ACF9 000406BF
	v_cvt_pk_f32_fp8_sdwa v[194:195], v191 src0_sel:WORD_1     // 000000003524: 7F84ACF9 000506BF
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]// 00000000352C: D3E10020 1C821D9E
	s_waitcnt lgkmcnt(4)                                       // 000000003534: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]// 000000003538: D3E10020 1C8221A0
	ds_read_b128 a[192:195], v4 offset:1536                    // 000000003540: DBFE0600 C0000004
	ds_read_b128 a[196:199], v4 offset:1600                    // 000000003548: DBFE0640 C4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]// 000000003550: D3E10020 1C8225A2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]// 000000003558: D3E10020 1C8229A4
	v_cvt_pk_f32_fp8_sdwa v[198:199], v197 src0_sel:WORD_0     // 000000003560: 7F8CACF9 000406C5
	v_cvt_pk_f32_fp8_sdwa v[200:201], v197 src0_sel:WORD_1     // 000000003568: 7F90ACF9 000506C5
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]// 000000003570: D3E10020 1C822DA6
	v_cvt_pk_f32_fp8_sdwa v[204:205], v203 src0_sel:WORD_0     // 000000003578: 7F98ACF9 000406CB
	v_cvt_pk_f32_fp8_sdwa v[206:207], v203 src0_sel:WORD_1     // 000000003580: 7F9CACF9 000506CB
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]// 000000003588: D3E10020 1C8231A8
	ds_read_b128 a[200:203], v4 offset:1792                    // 000000003590: DBFE0700 C8000004
	ds_read_b128 a[204:207], v4 offset:1856                    // 000000003598: DBFE0740 CC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]// 0000000035A0: D3E10020 1C8235AA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]// 0000000035A8: D3E10020 1C8239AC
	v_cvt_pk_f32_fp8_sdwa v[210:211], v209 src0_sel:WORD_0     // 0000000035B0: 7FA4ACF9 000406D1
	v_cvt_pk_f32_fp8_sdwa v[212:213], v209 src0_sel:WORD_1     // 0000000035B8: 7FA8ACF9 000506D1
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]// 0000000035C0: D3E10020 1C823DAE
	s_waitcnt lgkmcnt(4)                                       // 0000000035C8: BF8CC47F
	s_barrier                                                  // 0000000035CC: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]// 0000000035D0: D3E10020 1C8241B0
	ds_read_b128 a[208:211], v4 offset:2048                    // 0000000035D8: DBFE0800 D0000004
	ds_read_b128 a[212:215], v4 offset:2112                    // 0000000035E0: DBFE0840 D4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]// 0000000035E8: D3E10020 1C8245B2
	v_cvt_pk_f32_fp8_sdwa v[216:217], v215 src0_sel:WORD_0     // 0000000035F0: 7FB0ACF9 000406D7
	v_cvt_pk_f32_fp8_sdwa v[218:219], v215 src0_sel:WORD_1     // 0000000035F8: 7FB4ACF9 000506D7
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]// 000000003600: D3E10020 1C8249B4
	v_perm_b32 v168, v22, v20, s53                             // 000000003608: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 000000003610: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000003618: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 000000003620: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]// 000000003628: D3E10020 1C824DB6
	buffer_load_dword v178, v18, s[20:23], 0 offen             // 000000003630: E0501000 8005B212
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]// 000000003638: D3E10020 1C8251B8
	ds_write_b128 v6, v[168:171] offset:45312                  // 000000003640: D9BEB100 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]// 000000003648: D3E10020 1C8255BA
	buffer_load_dword v184, v18, s[20:23], 0 offen offset:64   // 000000003650: E0501040 8005B812
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]// 000000003658: D3E10020 1C8259BC
	v_perm_b32 v168, v23, v21, s53                             // 000000003660: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 000000003668: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 000000003670: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 000000003678: D1ED00AB 00D2331B
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]// 000000003680: D3E10020 1C825DBE
	buffer_load_dword v190, v18, s[20:23], 0 offen offset:128  // 000000003688: E0501080 8005BE12
	s_waitcnt lgkmcnt(1)                                       // 000000003690: BF8CC17F
	s_barrier                                                  // 000000003694: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]// 000000003698: D3E10020 1C8261C0
	ds_write_b128 v6, v[168:171] offset:46336                  // 0000000036A0: D9BEB500 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]// 0000000036A8: D3E10020 1C8265C2
	buffer_load_dword v196, v18, s[20:23], 0 offen offset:192  // 0000000036B0: E05010C0 8005C412
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]// 0000000036B8: D3E10020 1C8269C4
	v_cvt_pk_f32_fp8_sdwa v[222:223], v221 src0_sel:WORD_0     // 0000000036C0: 7FBCACF9 000406DD
	v_cvt_pk_f32_fp8_sdwa v[224:225], v221 src0_sel:WORD_1     // 0000000036C8: 7FC0ACF9 000506DD
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]// 0000000036D0: D3E10020 1C826DC6
	buffer_load_dword v202, v18, s[20:23], 0 offen offset:256  // 0000000036D8: E0501100 8005CA12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]// 0000000036E0: D3E10020 1C8271C8
	v_cvt_pk_f32_fp8_sdwa v[228:229], v227 src0_sel:WORD_0     // 0000000036E8: 7FC8ACF9 000406E3
	v_cvt_pk_f32_fp8_sdwa v[230:231], v227 src0_sel:WORD_1     // 0000000036F0: 7FCCACF9 000506E3
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]// 0000000036F8: D3E10020 1C8275CA
	buffer_load_dword v208, v18, s[20:23], 0 offen offset:320  // 000000003700: E0501140 8005D012
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]// 000000003708: D3E10020 1C8279CC
	v_perm_b32 v180, v181, v180, s52                           // 000000003710: D1ED00B4 00D369B5
	v_perm_b32 v181, v183, v182, s52                           // 000000003718: D1ED00B5 00D36DB7
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]// 000000003720: D3E10020 1C827DCE
	buffer_load_dword v214, v18, s[20:23], 0 offen offset:384  // 000000003728: E0501180 8005D612
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]// 000000003730: D3E10020 1C8281D0
	v_perm_b32 v186, v187, v186, s52                           // 000000003738: D1ED00BA 00D375BB
	v_perm_b32 v187, v189, v188, s52                           // 000000003740: D1ED00BB 00D379BD
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]// 000000003748: D3E10020 1C8285D2
	buffer_load_dword v220, v18, s[20:23], 0 offen offset:448  // 000000003750: E05011C0 8005DC12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]// 000000003758: D3E10020 1C8289D4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]// 000000003760: D3E10020 1C828DD6
	buffer_load_dword v226, v18, s[20:23], 0 offen offset:512  // 000000003768: E0501200 8005E212
	v_add_u32_e32 v8, s73, v8                                  // 000000003770: 68101049
	s_cmp_le_i32 s83, s82                                      // 000000003774: BF055253
	s_cbranch_scc1 label_0543                                  // 000000003778: BF850024
	v_mov_b32_e32 v25, 0xff800000                              // 00000000377C: 7E3202FF FF800000
	s_add_u32 s57, s82, 0                                      // 000000003784: 80398052
	v_mov_b32_e32 v24, s57                                     // 000000003788: 7E300239
	v_add_u32_e32 v24, s7, v24                                 // 00000000378C: 68303007
	s_sub_u32 s56, s83, 15                                     // 000000003790: 80B88F53
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000003794: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 000000003798: 0C282884
	v_add_u32_e32 v20, s56, v20                                // 00000000379C: 68282838
	v_add_u32_e32 v21, 1, v20                                  // 0000000037A0: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 0000000037A4: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 0000000037A8: 682E2883
	v_cmp_le_u32_e64 s[38:39], v20, v24                        // 0000000037AC: D0CB0026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 0000000037B4: 682828C0
	s_nop 0                                                    // 0000000037B8: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 0000000037BC: D1000020 009A4119
	v_cmp_le_u32_e64 s[38:39], v21, v24                        // 0000000037C4: D0CB0026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 0000000037CC: 682A2AC0
	s_nop 0                                                    // 0000000037D0: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 0000000037D4: D1000021 009A4319
	v_cmp_le_u32_e64 s[38:39], v22, v24                        // 0000000037DC: D0CB0026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 0000000037E4: 682C2CC0
	s_nop 0                                                    // 0000000037E8: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 0000000037EC: D1000022 009A4519
	v_cmp_le_u32_e64 s[38:39], v23, v24                        // 0000000037F4: D0CB0026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 0000000037FC: 682E2EC0
	s_nop 0                                                    // 000000003800: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000003804: D1000023 009A4719

000000000000380c <label_0543>:
	s_waitcnt lgkmcnt(0)                                       // 00000000380C: BF8CC07F
	s_barrier                                                  // 000000003810: BF8A0000
	v_max3_f32 v24, v32, v33, v32                              // 000000003814: D1D30018 04824320
	v_max3_f32 v24, v34, v35, v24                              // 00000000381C: D1D30018 04624722
	ds_write_b32 v3, v24 offset:53504                          // 000000003824: D81AD100 00001803
	v_perm_b32 v192, v193, v192, s52                           // 00000000382C: D1ED00C0 00D381C1
	v_perm_b32 v193, v195, v194, s52                           // 000000003834: D1ED00C1 00D385C3
	v_perm_b32 v198, v199, v198, s52                           // 00000000383C: D1ED00C6 00D38DC7
	v_perm_b32 v199, v201, v200, s52                           // 000000003844: D1ED00C7 00D391C9
	s_waitcnt lgkmcnt(0)                                       // 00000000384C: BF8CC07F
	ds_read_b32 v20, v2 offset:53504                           // 000000003850: D86CD100 14000002
	ds_read_b32 v21, v2 offset:53568                           // 000000003858: D86CD140 15000002
	ds_read_b32 v22, v2 offset:53632                           // 000000003860: D86CD180 16000002
	ds_read_b32 v23, v2 offset:53696                           // 000000003868: D86CD1C0 17000002
	v_perm_b32 v204, v205, v204, s52                           // 000000003870: D1ED00CC 00D399CD
	v_perm_b32 v205, v207, v206, s52                           // 000000003878: D1ED00CD 00D39DCF
	v_perm_b32 v210, v211, v210, s52                           // 000000003880: D1ED00D2 00D3A5D3
	v_perm_b32 v211, v213, v212, s52                           // 000000003888: D1ED00D3 00D3A9D5
	v_perm_b32 v216, v217, v216, s52                           // 000000003890: D1ED00D8 00D3B1D9
	v_perm_b32 v217, v219, v218, s52                           // 000000003898: D1ED00D9 00D3B5DB
	s_waitcnt lgkmcnt(0)                                       // 0000000038A0: BF8CC07F
	v_max3_f32 v24, v20, v21, v24                              // 0000000038A4: D1D30018 04622B14
	v_max3_f32 v24, v22, v23, v24                              // 0000000038AC: D1D30018 04622F16
	v_perm_b32 v222, v223, v222, s52                           // 0000000038B4: D1ED00DE 00D3BDDF
	v_perm_b32 v223, v225, v224, s52                           // 0000000038BC: D1ED00DF 00D3C1E1
	v_perm_b32 v228, v229, v228, s52                           // 0000000038C4: D1ED00E4 00D3C9E5
	v_perm_b32 v229, v231, v230, s52                           // 0000000038CC: D1ED00E5 00D3CDE7
	ds_read_b128 a[144:147], v7 offset:37120                   // 0000000038D4: DBFE9100 90000007
	ds_read_b128 a[148:151], v7 offset:38144                   // 0000000038DC: DBFE9500 94000007
	ds_write_b64 v176, v[180:181] offset:18560                 // 0000000038E4: D89A4880 0000B4B0
	ds_read_b128 a[152:155], v7 offset:39168                   // 0000000038EC: DBFE9900 98000007
	ds_read_b128 a[156:159], v7 offset:40192                   // 0000000038F4: DBFE9D00 9C000007
	ds_write_b64 v176, v[186:187] offset:18816                 // 0000000038FC: D89A4980 0000BAB0
	ds_read_b128 a[160:163], v7 offset:41216                   // 000000003904: DBFEA100 A0000007
	ds_read_b128 a[164:167], v7 offset:42240                   // 00000000390C: DBFEA500 A4000007
	ds_write_b64 v176, v[192:193] offset:19072                 // 000000003914: D89A4A80 0000C0B0
	ds_read_b128 a[168:171], v7 offset:43264                   // 00000000391C: DBFEA900 A8000007
	ds_read_b128 a[172:175], v7 offset:44288                   // 000000003924: DBFEAD00 AC000007
	ds_write_b64 v176, v[198:199] offset:19328                 // 00000000392C: D89A4B80 0000C6B0
	v_mov_b32_e32 v25, 0xff7fffff                              // 000000003934: 7E3202FF FF7FFFFF
	v_cmp_eq_u32_e64 s[38:39], v25, v12                        // 00000000393C: D0CA0026 00021919
	v_max_f32_e32 v20, v24, v12                                // 000000003944: 16281918
	v_sub_f32_e32 v16, v12, v20                                // 000000003948: 0420290C
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]                    // 00000000394C: D1000010 00990110
	v_mov_b32_e32 v12, v20                                     // 000000003954: 7E180314
	v_mul_f32_e32 v21, s5, v20                                 // 000000003958: 0A2A2805
	v_mul_f32_e32 v16, s5, v16                                 // 00000000395C: 0A202005
	v_exp_f32_e32 v16, v16                                     // 000000003960: 7E204110
	v_fma_f32 v32, v32, s5, -v21                               // 000000003964: D1CB0020 84540B20
	v_fma_f32 v33, v33, s5, -v21                               // 00000000396C: D1CB0021 84540B21
	v_fma_f32 v34, v34, s5, -v21                               // 000000003974: D1CB0022 84540B22
	v_fma_f32 v35, v35, s5, -v21                               // 00000000397C: D1CB0023 84540B23
	v_exp_f32_e32 v32, v32                                     // 000000003984: 7E404120
	v_exp_f32_e32 v33, v33                                     // 000000003988: 7E424121
	v_exp_f32_e32 v34, v34                                     // 00000000398C: 7E444122
	v_exp_f32_e32 v35, v35                                     // 000000003990: 7E464123
	v_mul_f32_e32 v14, v16, v14                                // 000000003994: 0A1C1D10
	v_mov_b32_e32 v22, v32                                     // 000000003998: 7E2C0320
	v_add_f32_e32 v22, v33, v22                                // 00000000399C: 022C2D21
	v_add_f32_e32 v22, v34, v22                                // 0000000039A0: 022C2D22
	v_add_f32_e32 v22, v35, v22                                // 0000000039A4: 022C2D23
	v_add_f32_e32 v14, v22, v14                                // 0000000039A8: 021C1D16
	v_mov_b32_e32 v29, 0xffff0000                              // 0000000039AC: 7E3A02FF FFFF0000
	v_mov_b32_e32 v30, 0x7fff0000                              // 0000000039B4: 7E3C02FF 7FFF0000
	v_mov_b32_e32 v31, 0x7fff                                  // 0000000039BC: 7E3E02FF 00007FFF
	v_cmp_u_f32_e64 s[38:39], v32, v32                         // 0000000039C4: D0480026 00024120
	v_add3_u32 v28, v32, v31, 1                                // 0000000039CC: D1FF001C 02063F20
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000039D4: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v33, v33                         // 0000000039DC: D0480026 00024321
	v_add3_u32 v28, v33, v31, 1                                // 0000000039E4: D1FF001C 02063F21
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000039EC: D1000015 009A3D1C
	v_perm_b32 v32, v21, v20, s52                              // 0000000039F4: D1ED0020 00D22915
	v_cmp_u_f32_e64 s[38:39], v34, v34                         // 0000000039FC: D0480026 00024522
	v_add3_u32 v28, v34, v31, 1                                // 000000003A04: D1FF001C 02063F22
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000003A0C: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v35, v35                         // 000000003A14: D0480026 00024723
	v_add3_u32 v28, v35, v31, 1                                // 000000003A1C: D1FF001C 02063F23
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000003A24: D1000015 009A3D1C
	v_perm_b32 v33, v21, v20, s52                              // 000000003A2C: D1ED0021 00D22915
	s_nop 2                                                    // 000000003A34: BF800002
	s_add_u32 s83, s84, s83                                    // 000000003A38: 80535354
	s_nop 0                                                    // 000000003A3C: BF800000
	v_mul_u32_u24_e32 v18, v11, v9                             // 000000003A40: 1024130B
	v_add_u32_e32 v18, v18, v1                                 // 000000003A44: 68240312
	s_mov_b32 m0, s37                                          // 000000003A48: BEFC0025
	v_mov_b32_e32 v22, v16                                     // 000000003A4C: 7E2C0310
	v_mov_b32_e32 v23, v16                                     // 000000003A50: 7E2E0310
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]                  // 000000003A54: D3B14028 18025116
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]                  // 000000003A5C: D3B1402A 18025516
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]                  // 000000003A64: D3B1402C 18025916
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]                  // 000000003A6C: D3B1402E 18025D16
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]                  // 000000003A74: D3B14030 18026116
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]                  // 000000003A7C: D3B14032 18026516
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]                  // 000000003A84: D3B14034 18026916
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]                  // 000000003A8C: D3B14036 18026D16
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]                  // 000000003A94: D3B14038 18027116
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]                  // 000000003A9C: D3B1403A 18027516
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]                  // 000000003AA4: D3B1403C 18027916
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]                  // 000000003AAC: D3B1403E 18027D16
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]                  // 000000003AB4: D3B14040 18028116
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]                  // 000000003ABC: D3B14042 18028516
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]                  // 000000003AC4: D3B14044 18028916
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]                  // 000000003ACC: D3B14046 18028D16
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]                  // 000000003AD4: D3B14048 18029116
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]                  // 000000003ADC: D3B1404A 18029516
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]                  // 000000003AE4: D3B1404C 18029916
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]                  // 000000003AEC: D3B1404E 18029D16
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]                  // 000000003AF4: D3B14050 1802A116
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]                  // 000000003AFC: D3B14052 1802A516
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]                  // 000000003B04: D3B14054 1802A916
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]                  // 000000003B0C: D3B14056 1802AD16
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]                  // 000000003B14: D3B14058 1802B116
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]                  // 000000003B1C: D3B1405A 1802B516
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]                  // 000000003B24: D3B1405C 1802B916
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]                  // 000000003B2C: D3B1405E 1802BD16
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]                  // 000000003B34: D3B14060 1802C116
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]                  // 000000003B3C: D3B14062 1802C516
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]              // 000000003B44: D3B14064 1802C916
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]              // 000000003B4C: D3B14066 1802CD16
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]              // 000000003B54: D3B14068 1802D116
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]              // 000000003B5C: D3B1406A 1802D516
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]              // 000000003B64: D3B1406C 1802D916
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]              // 000000003B6C: D3B1406E 1802DD16
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]              // 000000003B74: D3B14070 1802E116
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]              // 000000003B7C: D3B14072 1802E516
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]              // 000000003B84: D3B14074 1802E916
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]              // 000000003B8C: D3B14076 1802ED16
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]              // 000000003B94: D3B14078 1802F116
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]              // 000000003B9C: D3B1407A 1802F516
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]              // 000000003BA4: D3B1407C 1802F916
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]              // 000000003BAC: D3B1407E 1802FD16
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]              // 000000003BB4: D3B14080 18030116
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]              // 000000003BBC: D3B14082 18030516
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]              // 000000003BC4: D3B14084 18030916
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]              // 000000003BCC: D3B14086 18030D16
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]              // 000000003BD4: D3B14088 18031116
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]              // 000000003BDC: D3B1408A 18031516
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]              // 000000003BE4: D3B1408C 18031916
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]              // 000000003BEC: D3B1408E 18031D16
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]              // 000000003BF4: D3B14090 18032116
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]              // 000000003BFC: D3B14092 18032516
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]              // 000000003C04: D3B14094 18032916
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]              // 000000003C0C: D3B14096 18032D16
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]              // 000000003C14: D3B14098 18033116
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]              // 000000003C1C: D3B1409A 18033516
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]              // 000000003C24: D3B1409C 18033916
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]              // 000000003C2C: D3B1409E 18033D16
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]              // 000000003C34: D3B140A0 18034116
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]              // 000000003C3C: D3B140A2 18034516
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]              // 000000003C44: D3B140A4 18034916
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]              // 000000003C4C: D3B140A6 18034D16
	s_waitcnt lgkmcnt(0)                                       // 000000003C54: BF8CC07F
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]// 000000003C58: D3E10028 0CA24190
	ds_read_b128 a[176:179], v7 offset:45312                   // 000000003C60: DBFEB100 B0000007
	ds_read_b128 a[180:183], v7 offset:46336                   // 000000003C68: DBFEB500 B4000007
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]// 000000003C70: D3E1002C 0CB24192
	ds_write_b64 v176, v[192:193] offset:19072                 // 000000003C78: D89A4A80 0000C0B0
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]// 000000003C80: D3E10030 0CC24194
	ds_write_b64 v176, v[198:199] offset:19328                 // 000000003C88: D89A4B80 0000C6B0
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]// 000000003C90: D3E10034 0CD24196
	ds_write_b64 v176, v[204:205] offset:19584                 // 000000003C98: D89A4C80 0000CCB0
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]// 000000003CA0: D3E10038 0CE24198
	ds_read_b128 a[184:187], v7 offset:47360                   // 000000003CA8: DBFEB900 B8000007
	ds_read_b128 a[188:191], v7 offset:48384                   // 000000003CB0: DBFEBD00 BC000007
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]// 000000003CB8: D3E1003C 0CF2419A
	ds_write_b64 v176, v[210:211] offset:19840                 // 000000003CC0: D89A4D80 0000D2B0
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]// 000000003CC8: D3E10040 0D02419C
	ds_write_b64 v176, v[216:217] offset:20096                 // 000000003CD0: D89A4E80 0000D8B0
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]// 000000003CD8: D3E10044 0D12419E
	ds_write_b64 v176, v[222:223] offset:20352                 // 000000003CE0: D89A4F80 0000DEB0
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]// 000000003CE8: D3E10048 0D2241A0
	ds_read_b128 a[192:195], v7 offset:49408                   // 000000003CF0: DBFEC100 C0000007
	ds_read_b128 a[196:199], v7 offset:50432                   // 000000003CF8: DBFEC500 C4000007
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]// 000000003D00: D3E1004C 0D3241A2
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]// 000000003D08: D3E10050 0D4241A4
	ds_write_b64 v176, v[228:229] offset:20608                 // 000000003D10: D89A5080 0000E4B0
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]// 000000003D18: D3E10054 0D5241A6
	s_waitcnt lgkmcnt(4)                                       // 000000003D20: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]// 000000003D24: D3E10058 0D6241A8
	ds_read_b128 a[200:203], v7 offset:51456                   // 000000003D2C: DBFEC900 C8000007
	ds_read_b128 a[204:207], v7 offset:52480                   // 000000003D34: DBFECD00 CC000007
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]// 000000003D3C: D3E1005C 0D7241AA
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]// 000000003D44: D3E10060 0D8241AC
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]// 000000003D4C: D3E10064 0D9241AE
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]// 000000003D54: D3E10068 0DA241B0
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]// 000000003D5C: D3E1006C 0DB241B2
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]// 000000003D64: D3E10070 0DC241B4
	s_waitcnt vmcnt(9) lgkmcnt(9)                              // 000000003D6C: BF8C0979
	s_barrier                                                  // 000000003D70: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]// 000000003D74: D3E10074 0DD241B6
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]// 000000003D7C: D3E10078 0DE241B8
	ds_read_b64 v[20:21], v5 offset:18560                      // 000000003D84: D8EC4880 14000005
	ds_read_b64 v[22:23], v5 offset:23200                      // 000000003D8C: D8EC5AA0 16000005
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]// 000000003D94: D3E1007C 0DF241BA
	ds_read_b64 v[24:25], v5 offset:27840                      // 000000003D9C: D8EC6CC0 18000005
	ds_read_b64 v[26:27], v5 offset:32480                      // 000000003DA4: D8EC7EE0 1A000005
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]// 000000003DAC: D3E10080 0E0241BC
	ds_read_b128 a[144:147], v4 offset:18560                   // 000000003DB4: DBFE4880 90000004
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]// 000000003DBC: D3E10084 0E1241BE
	ds_read_b128 a[148:151], v4 offset:18624                   // 000000003DC4: DBFE48C0 94000004
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]// 000000003DCC: D3E10088 0E2241C0
	ds_read_b128 a[152:155], v4 offset:18816                   // 000000003DD4: DBFE4980 98000004
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]// 000000003DDC: D3E1008C 0E3241C2
	ds_read_b128 a[156:159], v4 offset:18880                   // 000000003DE4: DBFE49C0 9C000004
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]// 000000003DEC: D3E10090 0E4241C4
	ds_read_b128 a[160:163], v4 offset:19072                   // 000000003DF4: DBFE4A80 A0000004
	v_mfma_f32_16x16x16_bf16 v[148:151], a[198:199], v[32:33], v[148:151]// 000000003DFC: D3E10094 0E5241C6
	ds_read_b128 a[164:167], v4 offset:19136                   // 000000003E04: DBFE4AC0 A4000004
	v_mfma_f32_16x16x16_bf16 v[152:155], a[200:201], v[32:33], v[152:155]// 000000003E0C: D3E10098 0E6241C8
	ds_read_b128 a[168:171], v4 offset:19328                   // 000000003E14: DBFE4B80 A8000004
	v_mfma_f32_16x16x16_bf16 v[156:159], a[202:203], v[32:33], v[156:159]// 000000003E1C: D3E1009C 0E7241CA
	ds_read_b128 a[172:175], v4 offset:19392                   // 000000003E24: DBFE4BC0 AC000004
	v_mfma_f32_16x16x16_bf16 v[160:163], a[204:205], v[32:33], v[160:163]// 000000003E2C: D3E100A0 0E8241CC
	s_waitcnt lgkmcnt(8)                                       // 000000003E34: BF8CC87F
	v_perm_b32 v168, v22, v20, s53                             // 000000003E38: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 000000003E40: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000003E48: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 000000003E50: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[164:167], a[206:207], v[32:33], v[164:167]// 000000003E58: D3E100A4 0E9241CE
	ds_write_b128 v6, v[168:171] offset:37120                  // 000000003E60: D9BE9100 0000A806
	v_perm_b32 v168, v23, v21, s53                             // 000000003E68: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 000000003E70: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 000000003E78: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 000000003E80: D1ED00AB 00D2331B
	ds_write_b128 v6, v[168:171] offset:38144                  // 000000003E88: D9BE9500 0000A806
	ds_read_b64 v[20:21], v5 offset:19584                      // 000000003E90: D8EC4C80 14000005
	ds_read_b64 v[22:23], v5 offset:24224                      // 000000003E98: D8EC5EA0 16000005
	ds_read_b64 v[24:25], v5 offset:28864                      // 000000003EA0: D8EC70C0 18000005
	ds_read_b64 v[26:27], v5 offset:33504                      // 000000003EA8: D8EC82E0 1A000005
	s_nop 0                                                    // 000000003EB0: BF800000
	s_addk_i32 s70, 0x1                                        // 000000003EB4: B7460001
	s_cmp_lt_i32 s70, s71                                      // 000000003EB8: BF044746
	s_cbranch_scc0 label_097C                                  // 000000003EBC: BF84028C
	s_waitcnt lgkmcnt(4)                                       // 000000003EC0: BF8CC47F
	s_waitcnt vmcnt(0)                                         // 000000003EC4: BF8C0F70
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0   // 000000003EC8: D3E10020 1A020190
	ds_read_b128 a[176:179], v4 offset:19584                   // 000000003ED0: DBFE4C80 B0000004
	ds_read_b128 a[180:183], v4 offset:19648                   // 000000003ED8: DBFE4CC0 B4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]// 000000003EE0: D3E10020 1C820592
	buffer_load_dword v11, v8, s[24:27], 0 offen               // 000000003EE8: E0501000 80060B08
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]// 000000003EF0: D3E10020 1C820994
	v_cvt_pk_f32_fp8_sdwa v[180:181], v178 src0_sel:WORD_0     // 000000003EF8: 7F68ACF9 000406B2
	v_cvt_pk_f32_fp8_sdwa v[182:183], v178 src0_sel:WORD_1     // 000000003F00: 7F6CACF9 000506B2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]// 000000003F08: D3E10020 1C820D96
	v_cvt_pk_f32_fp8_sdwa v[186:187], v184 src0_sel:WORD_0     // 000000003F10: 7F74ACF9 000406B8
	v_cvt_pk_f32_fp8_sdwa v[188:189], v184 src0_sel:WORD_1     // 000000003F18: 7F78ACF9 000506B8
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]// 000000003F20: D3E10020 1C821198
	ds_read_b128 a[184:187], v4 offset:19840                   // 000000003F28: DBFE4D80 B8000004
	ds_read_b128 a[188:191], v4 offset:19904                   // 000000003F30: DBFE4DC0 BC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]// 000000003F38: D3E10020 1C82159A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]// 000000003F40: D3E10020 1C82199C
	v_cvt_pk_f32_fp8_sdwa v[192:193], v190 src0_sel:WORD_0     // 000000003F48: 7F80ACF9 000406BE
	v_cvt_pk_f32_fp8_sdwa v[194:195], v190 src0_sel:WORD_1     // 000000003F50: 7F84ACF9 000506BE
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]// 000000003F58: D3E10020 1C821D9E
	s_waitcnt lgkmcnt(4)                                       // 000000003F60: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]// 000000003F64: D3E10020 1C8221A0
	ds_read_b128 a[192:195], v4 offset:20096                   // 000000003F6C: DBFE4E80 C0000004
	ds_read_b128 a[196:199], v4 offset:20160                   // 000000003F74: DBFE4EC0 C4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]// 000000003F7C: D3E10020 1C8225A2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]// 000000003F84: D3E10020 1C8229A4
	v_cvt_pk_f32_fp8_sdwa v[198:199], v196 src0_sel:WORD_0     // 000000003F8C: 7F8CACF9 000406C4
	v_cvt_pk_f32_fp8_sdwa v[200:201], v196 src0_sel:WORD_1     // 000000003F94: 7F90ACF9 000506C4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]// 000000003F9C: D3E10020 1C822DA6
	v_cvt_pk_f32_fp8_sdwa v[204:205], v202 src0_sel:WORD_0     // 000000003FA4: 7F98ACF9 000406CA
	v_cvt_pk_f32_fp8_sdwa v[206:207], v202 src0_sel:WORD_1     // 000000003FAC: 7F9CACF9 000506CA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]// 000000003FB4: D3E10020 1C8231A8
	ds_read_b128 a[200:203], v4 offset:20352                   // 000000003FBC: DBFE4F80 C8000004
	ds_read_b128 a[204:207], v4 offset:20416                   // 000000003FC4: DBFE4FC0 CC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]// 000000003FCC: D3E10020 1C8235AA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]// 000000003FD4: D3E10020 1C8239AC
	v_cvt_pk_f32_fp8_sdwa v[210:211], v208 src0_sel:WORD_0     // 000000003FDC: 7FA4ACF9 000406D0
	v_cvt_pk_f32_fp8_sdwa v[212:213], v208 src0_sel:WORD_1     // 000000003FE4: 7FA8ACF9 000506D0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]// 000000003FEC: D3E10020 1C823DAE
	s_waitcnt lgkmcnt(4)                                       // 000000003FF4: BF8CC47F
	s_barrier                                                  // 000000003FF8: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]// 000000003FFC: D3E10020 1C8241B0
	ds_read_b128 a[208:211], v4 offset:20608                   // 000000004004: DBFE5080 D0000004
	ds_read_b128 a[212:215], v4 offset:20672                   // 00000000400C: DBFE50C0 D4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]// 000000004014: D3E10020 1C8245B2
	v_cvt_pk_f32_fp8_sdwa v[216:217], v214 src0_sel:WORD_0     // 00000000401C: 7FB0ACF9 000406D6
	v_cvt_pk_f32_fp8_sdwa v[218:219], v214 src0_sel:WORD_1     // 000000004024: 7FB4ACF9 000506D6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]// 00000000402C: D3E10020 1C8249B4
	v_perm_b32 v168, v22, v20, s53                             // 000000004034: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 00000000403C: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000004044: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 00000000404C: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]// 000000004054: D3E10020 1C824DB6
	buffer_load_dword v179, v18, s[20:23], 0 offen             // 00000000405C: E0501000 8005B312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]// 000000004064: D3E10020 1C8251B8
	ds_write_b128 v6, v[168:171] offset:45312                  // 00000000406C: D9BEB100 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]// 000000004074: D3E10020 1C8255BA
	buffer_load_dword v185, v18, s[20:23], 0 offen offset:64   // 00000000407C: E0501040 8005B912
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]// 000000004084: D3E10020 1C8259BC
	v_perm_b32 v168, v23, v21, s53                             // 00000000408C: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 000000004094: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 00000000409C: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 0000000040A4: D1ED00AB 00D2331B
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]// 0000000040AC: D3E10020 1C825DBE
	buffer_load_dword v191, v18, s[20:23], 0 offen offset:128  // 0000000040B4: E0501080 8005BF12
	s_waitcnt lgkmcnt(1)                                       // 0000000040BC: BF8CC17F
	s_barrier                                                  // 0000000040C0: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]// 0000000040C4: D3E10020 1C8261C0
	ds_write_b128 v6, v[168:171] offset:46336                  // 0000000040CC: D9BEB500 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]// 0000000040D4: D3E10020 1C8265C2
	buffer_load_dword v197, v18, s[20:23], 0 offen offset:192  // 0000000040DC: E05010C0 8005C512
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]// 0000000040E4: D3E10020 1C8269C4
	v_cvt_pk_f32_fp8_sdwa v[222:223], v220 src0_sel:WORD_0     // 0000000040EC: 7FBCACF9 000406DC
	v_cvt_pk_f32_fp8_sdwa v[224:225], v220 src0_sel:WORD_1     // 0000000040F4: 7FC0ACF9 000506DC
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]// 0000000040FC: D3E10020 1C826DC6
	buffer_load_dword v203, v18, s[20:23], 0 offen offset:256  // 000000004104: E0501100 8005CB12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]// 00000000410C: D3E10020 1C8271C8
	v_cvt_pk_f32_fp8_sdwa v[228:229], v226 src0_sel:WORD_0     // 000000004114: 7FC8ACF9 000406E2
	v_cvt_pk_f32_fp8_sdwa v[230:231], v226 src0_sel:WORD_1     // 00000000411C: 7FCCACF9 000506E2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]// 000000004124: D3E10020 1C8275CA
	buffer_load_dword v209, v18, s[20:23], 0 offen offset:320  // 00000000412C: E0501140 8005D112
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]// 000000004134: D3E10020 1C8279CC
	v_perm_b32 v180, v181, v180, s52                           // 00000000413C: D1ED00B4 00D369B5
	v_perm_b32 v181, v183, v182, s52                           // 000000004144: D1ED00B5 00D36DB7
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]// 00000000414C: D3E10020 1C827DCE
	buffer_load_dword v215, v18, s[20:23], 0 offen offset:384  // 000000004154: E0501180 8005D712
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]// 00000000415C: D3E10020 1C8281D0
	v_perm_b32 v186, v187, v186, s52                           // 000000004164: D1ED00BA 00D375BB
	v_perm_b32 v187, v189, v188, s52                           // 00000000416C: D1ED00BB 00D379BD
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]// 000000004174: D3E10020 1C8285D2
	buffer_load_dword v221, v18, s[20:23], 0 offen offset:448  // 00000000417C: E05011C0 8005DD12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]// 000000004184: D3E10020 1C8289D4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]// 00000000418C: D3E10020 1C828DD6
	buffer_load_dword v227, v18, s[20:23], 0 offen offset:512  // 000000004194: E0501200 8005E312
	v_add_u32_e32 v8, s73, v8                                  // 00000000419C: 68101049
	s_cmp_le_i32 s83, s82                                      // 0000000041A0: BF055253
	s_cbranch_scc1 label_07CE                                  // 0000000041A4: BF850024
	v_mov_b32_e32 v25, 0xff800000                              // 0000000041A8: 7E3202FF FF800000
	s_add_u32 s57, s82, 0                                      // 0000000041B0: 80398052
	v_mov_b32_e32 v24, s57                                     // 0000000041B4: 7E300239
	v_add_u32_e32 v24, s7, v24                                 // 0000000041B8: 68303007
	s_sub_u32 s56, s83, 15                                     // 0000000041BC: 80B88F53
	v_lshrrev_b32_e32 v20, 4, v0                               // 0000000041C0: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 0000000041C4: 0C282884
	v_add_u32_e32 v20, s56, v20                                // 0000000041C8: 68282838
	v_add_u32_e32 v21, 1, v20                                  // 0000000041CC: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 0000000041D0: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 0000000041D4: 682E2883
	v_cmp_le_u32_e64 s[38:39], v20, v24                        // 0000000041D8: D0CB0026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 0000000041E0: 682828C0
	s_nop 0                                                    // 0000000041E4: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 0000000041E8: D1000020 009A4119
	v_cmp_le_u32_e64 s[38:39], v21, v24                        // 0000000041F0: D0CB0026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 0000000041F8: 682A2AC0
	s_nop 0                                                    // 0000000041FC: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 000000004200: D1000021 009A4319
	v_cmp_le_u32_e64 s[38:39], v22, v24                        // 000000004208: D0CB0026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 000000004210: 682C2CC0
	s_nop 0                                                    // 000000004214: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 000000004218: D1000022 009A4519
	v_cmp_le_u32_e64 s[38:39], v23, v24                        // 000000004220: D0CB0026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 000000004228: 682E2EC0
	s_nop 0                                                    // 00000000422C: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000004230: D1000023 009A4719

0000000000004238 <label_07CE>:
	s_waitcnt lgkmcnt(0)                                       // 000000004238: BF8CC07F
	s_barrier                                                  // 00000000423C: BF8A0000
	v_max3_f32 v24, v32, v33, v32                              // 000000004240: D1D30018 04824320
	v_max3_f32 v24, v34, v35, v24                              // 000000004248: D1D30018 04624722
	ds_write_b32 v3, v24 offset:53504                          // 000000004250: D81AD100 00001803
	v_perm_b32 v192, v193, v192, s52                           // 000000004258: D1ED00C0 00D381C1
	v_perm_b32 v193, v195, v194, s52                           // 000000004260: D1ED00C1 00D385C3
	v_perm_b32 v198, v199, v198, s52                           // 000000004268: D1ED00C6 00D38DC7
	v_perm_b32 v199, v201, v200, s52                           // 000000004270: D1ED00C7 00D391C9
	s_waitcnt lgkmcnt(0)                                       // 000000004278: BF8CC07F
	ds_read_b32 v20, v2 offset:53504                           // 00000000427C: D86CD100 14000002
	ds_read_b32 v21, v2 offset:53568                           // 000000004284: D86CD140 15000002
	ds_read_b32 v22, v2 offset:53632                           // 00000000428C: D86CD180 16000002
	ds_read_b32 v23, v2 offset:53696                           // 000000004294: D86CD1C0 17000002
	v_perm_b32 v204, v205, v204, s52                           // 00000000429C: D1ED00CC 00D399CD
	v_perm_b32 v205, v207, v206, s52                           // 0000000042A4: D1ED00CD 00D39DCF
	v_perm_b32 v210, v211, v210, s52                           // 0000000042AC: D1ED00D2 00D3A5D3
	v_perm_b32 v211, v213, v212, s52                           // 0000000042B4: D1ED00D3 00D3A9D5
	v_perm_b32 v216, v217, v216, s52                           // 0000000042BC: D1ED00D8 00D3B1D9
	v_perm_b32 v217, v219, v218, s52                           // 0000000042C4: D1ED00D9 00D3B5DB
	s_waitcnt lgkmcnt(0)                                       // 0000000042CC: BF8CC07F
	v_max3_f32 v24, v20, v21, v24                              // 0000000042D0: D1D30018 04622B14
	v_max3_f32 v24, v22, v23, v24                              // 0000000042D8: D1D30018 04622F16
	v_perm_b32 v222, v223, v222, s52                           // 0000000042E0: D1ED00DE 00D3BDDF
	v_perm_b32 v223, v225, v224, s52                           // 0000000042E8: D1ED00DF 00D3C1E1
	v_perm_b32 v228, v229, v228, s52                           // 0000000042F0: D1ED00E4 00D3C9E5
	v_perm_b32 v229, v231, v230, s52                           // 0000000042F8: D1ED00E5 00D3CDE7
	ds_read_b128 a[144:147], v7 offset:37120                   // 000000004300: DBFE9100 90000007
	ds_read_b128 a[148:151], v7 offset:38144                   // 000000004308: DBFE9500 94000007
	ds_write_b64 v176, v[180:181]                              // 000000004310: D89A0000 0000B4B0
	ds_read_b128 a[152:155], v7 offset:39168                   // 000000004318: DBFE9900 98000007
	ds_read_b128 a[156:159], v7 offset:40192                   // 000000004320: DBFE9D00 9C000007
	ds_write_b64 v176, v[186:187] offset:256                   // 000000004328: D89A0100 0000BAB0
	ds_read_b128 a[160:163], v7 offset:41216                   // 000000004330: DBFEA100 A0000007
	ds_read_b128 a[164:167], v7 offset:42240                   // 000000004338: DBFEA500 A4000007
	ds_write_b64 v176, v[192:193] offset:512                   // 000000004340: D89A0200 0000C0B0
	ds_read_b128 a[168:171], v7 offset:43264                   // 000000004348: DBFEA900 A8000007
	ds_read_b128 a[172:175], v7 offset:44288                   // 000000004350: DBFEAD00 AC000007
	ds_write_b64 v176, v[198:199] offset:768                   // 000000004358: D89A0300 0000C6B0
	v_mov_b32_e32 v25, 0xff7fffff                              // 000000004360: 7E3202FF FF7FFFFF
	v_cmp_eq_u32_e64 s[38:39], v25, v12                        // 000000004368: D0CA0026 00021919
	v_max_f32_e32 v20, v24, v12                                // 000000004370: 16281918
	v_sub_f32_e32 v16, v12, v20                                // 000000004374: 0420290C
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]                    // 000000004378: D1000010 00990110
	v_mov_b32_e32 v12, v20                                     // 000000004380: 7E180314
	v_mul_f32_e32 v21, s5, v20                                 // 000000004384: 0A2A2805
	v_mul_f32_e32 v16, s5, v16                                 // 000000004388: 0A202005
	v_exp_f32_e32 v16, v16                                     // 00000000438C: 7E204110
	v_fma_f32 v32, v32, s5, -v21                               // 000000004390: D1CB0020 84540B20
	v_fma_f32 v33, v33, s5, -v21                               // 000000004398: D1CB0021 84540B21
	v_fma_f32 v34, v34, s5, -v21                               // 0000000043A0: D1CB0022 84540B22
	v_fma_f32 v35, v35, s5, -v21                               // 0000000043A8: D1CB0023 84540B23
	v_exp_f32_e32 v32, v32                                     // 0000000043B0: 7E404120
	v_exp_f32_e32 v33, v33                                     // 0000000043B4: 7E424121
	v_exp_f32_e32 v34, v34                                     // 0000000043B8: 7E444122
	v_exp_f32_e32 v35, v35                                     // 0000000043BC: 7E464123
	v_mul_f32_e32 v14, v16, v14                                // 0000000043C0: 0A1C1D10
	v_mov_b32_e32 v22, v32                                     // 0000000043C4: 7E2C0320
	v_add_f32_e32 v22, v33, v22                                // 0000000043C8: 022C2D21
	v_add_f32_e32 v22, v34, v22                                // 0000000043CC: 022C2D22
	v_add_f32_e32 v22, v35, v22                                // 0000000043D0: 022C2D23
	v_add_f32_e32 v14, v22, v14                                // 0000000043D4: 021C1D16
	v_mov_b32_e32 v29, 0xffff0000                              // 0000000043D8: 7E3A02FF FFFF0000
	v_mov_b32_e32 v30, 0x7fff0000                              // 0000000043E0: 7E3C02FF 7FFF0000
	v_mov_b32_e32 v31, 0x7fff                                  // 0000000043E8: 7E3E02FF 00007FFF
	v_cmp_u_f32_e64 s[38:39], v32, v32                         // 0000000043F0: D0480026 00024120
	v_add3_u32 v28, v32, v31, 1                                // 0000000043F8: D1FF001C 02063F20
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000004400: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v33, v33                         // 000000004408: D0480026 00024321
	v_add3_u32 v28, v33, v31, 1                                // 000000004410: D1FF001C 02063F21
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000004418: D1000015 009A3D1C
	v_perm_b32 v32, v21, v20, s52                              // 000000004420: D1ED0020 00D22915
	v_cmp_u_f32_e64 s[38:39], v34, v34                         // 000000004428: D0480026 00024522
	v_add3_u32 v28, v34, v31, 1                                // 000000004430: D1FF001C 02063F22
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000004438: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v35, v35                         // 000000004440: D0480026 00024723
	v_add3_u32 v28, v35, v31, 1                                // 000000004448: D1FF001C 02063F23
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000004450: D1000015 009A3D1C
	v_perm_b32 v33, v21, v20, s52                              // 000000004458: D1ED0021 00D22915
	s_nop 2                                                    // 000000004460: BF800002
	s_add_u32 s83, s84, s83                                    // 000000004464: 80535354
	s_nop 0                                                    // 000000004468: BF800000
	v_mul_u32_u24_e32 v18, v10, v9                             // 00000000446C: 1024130A
	v_add_u32_e32 v18, v18, v1                                 // 000000004470: 68240312
	s_mov_b32 m0, s35                                          // 000000004474: BEFC0023
	v_mov_b32_e32 v22, v16                                     // 000000004478: 7E2C0310
	v_mov_b32_e32 v23, v16                                     // 00000000447C: 7E2E0310
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]                  // 000000004480: D3B14028 18025116
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]                  // 000000004488: D3B1402A 18025516
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]                  // 000000004490: D3B1402C 18025916
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]                  // 000000004498: D3B1402E 18025D16
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]                  // 0000000044A0: D3B14030 18026116
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]                  // 0000000044A8: D3B14032 18026516
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]                  // 0000000044B0: D3B14034 18026916
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]                  // 0000000044B8: D3B14036 18026D16
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]                  // 0000000044C0: D3B14038 18027116
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]                  // 0000000044C8: D3B1403A 18027516
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]                  // 0000000044D0: D3B1403C 18027916
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]                  // 0000000044D8: D3B1403E 18027D16
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]                  // 0000000044E0: D3B14040 18028116
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]                  // 0000000044E8: D3B14042 18028516
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]                  // 0000000044F0: D3B14044 18028916
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]                  // 0000000044F8: D3B14046 18028D16
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]                  // 000000004500: D3B14048 18029116
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]                  // 000000004508: D3B1404A 18029516
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]                  // 000000004510: D3B1404C 18029916
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]                  // 000000004518: D3B1404E 18029D16
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]                  // 000000004520: D3B14050 1802A116
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]                  // 000000004528: D3B14052 1802A516
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]                  // 000000004530: D3B14054 1802A916
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]                  // 000000004538: D3B14056 1802AD16
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]                  // 000000004540: D3B14058 1802B116
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]                  // 000000004548: D3B1405A 1802B516
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]                  // 000000004550: D3B1405C 1802B916
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]                  // 000000004558: D3B1405E 1802BD16
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]                  // 000000004560: D3B14060 1802C116
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]                  // 000000004568: D3B14062 1802C516
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]              // 000000004570: D3B14064 1802C916
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]              // 000000004578: D3B14066 1802CD16
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]              // 000000004580: D3B14068 1802D116
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]              // 000000004588: D3B1406A 1802D516
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]              // 000000004590: D3B1406C 1802D916
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]              // 000000004598: D3B1406E 1802DD16
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]              // 0000000045A0: D3B14070 1802E116
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]              // 0000000045A8: D3B14072 1802E516
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]              // 0000000045B0: D3B14074 1802E916
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]              // 0000000045B8: D3B14076 1802ED16
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]              // 0000000045C0: D3B14078 1802F116
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]              // 0000000045C8: D3B1407A 1802F516
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]              // 0000000045D0: D3B1407C 1802F916
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]              // 0000000045D8: D3B1407E 1802FD16
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]              // 0000000045E0: D3B14080 18030116
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]              // 0000000045E8: D3B14082 18030516
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]              // 0000000045F0: D3B14084 18030916
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]              // 0000000045F8: D3B14086 18030D16
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]              // 000000004600: D3B14088 18031116
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]              // 000000004608: D3B1408A 18031516
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]              // 000000004610: D3B1408C 18031916
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]              // 000000004618: D3B1408E 18031D16
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]              // 000000004620: D3B14090 18032116
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]              // 000000004628: D3B14092 18032516
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]              // 000000004630: D3B14094 18032916
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]              // 000000004638: D3B14096 18032D16
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]              // 000000004640: D3B14098 18033116
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]              // 000000004648: D3B1409A 18033516
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]              // 000000004650: D3B1409C 18033916
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]              // 000000004658: D3B1409E 18033D16
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]              // 000000004660: D3B140A0 18034116
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]              // 000000004668: D3B140A2 18034516
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]              // 000000004670: D3B140A4 18034916
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]              // 000000004678: D3B140A6 18034D16
	s_waitcnt lgkmcnt(0)                                       // 000000004680: BF8CC07F
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]// 000000004684: D3E10028 0CA24190
	ds_read_b128 a[176:179], v7 offset:45312                   // 00000000468C: DBFEB100 B0000007
	ds_read_b128 a[180:183], v7 offset:46336                   // 000000004694: DBFEB500 B4000007
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]// 00000000469C: D3E1002C 0CB24192
	ds_write_b64 v176, v[192:193] offset:512                   // 0000000046A4: D89A0200 0000C0B0
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]// 0000000046AC: D3E10030 0CC24194
	ds_write_b64 v176, v[198:199] offset:768                   // 0000000046B4: D89A0300 0000C6B0
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]// 0000000046BC: D3E10034 0CD24196
	ds_write_b64 v176, v[204:205] offset:1024                  // 0000000046C4: D89A0400 0000CCB0
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]// 0000000046CC: D3E10038 0CE24198
	ds_read_b128 a[184:187], v7 offset:47360                   // 0000000046D4: DBFEB900 B8000007
	ds_read_b128 a[188:191], v7 offset:48384                   // 0000000046DC: DBFEBD00 BC000007
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]// 0000000046E4: D3E1003C 0CF2419A
	ds_write_b64 v176, v[210:211] offset:1280                  // 0000000046EC: D89A0500 0000D2B0
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]// 0000000046F4: D3E10040 0D02419C
	ds_write_b64 v176, v[216:217] offset:1536                  // 0000000046FC: D89A0600 0000D8B0
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]// 000000004704: D3E10044 0D12419E
	ds_write_b64 v176, v[222:223] offset:1792                  // 00000000470C: D89A0700 0000DEB0
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]// 000000004714: D3E10048 0D2241A0
	ds_read_b128 a[192:195], v7 offset:49408                   // 00000000471C: DBFEC100 C0000007
	ds_read_b128 a[196:199], v7 offset:50432                   // 000000004724: DBFEC500 C4000007
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]// 00000000472C: D3E1004C 0D3241A2
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]// 000000004734: D3E10050 0D4241A4
	ds_write_b64 v176, v[228:229] offset:2048                  // 00000000473C: D89A0800 0000E4B0
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]// 000000004744: D3E10054 0D5241A6
	s_waitcnt lgkmcnt(4)                                       // 00000000474C: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]// 000000004750: D3E10058 0D6241A8
	ds_read_b128 a[200:203], v7 offset:51456                   // 000000004758: DBFEC900 C8000007
	ds_read_b128 a[204:207], v7 offset:52480                   // 000000004760: DBFECD00 CC000007
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]// 000000004768: D3E1005C 0D7241AA
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]// 000000004770: D3E10060 0D8241AC
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]// 000000004778: D3E10064 0D9241AE
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]// 000000004780: D3E10068 0DA241B0
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]// 000000004788: D3E1006C 0DB241B2
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]// 000000004790: D3E10070 0DC241B4
	s_waitcnt vmcnt(9) lgkmcnt(9)                              // 000000004798: BF8C0979
	s_barrier                                                  // 00000000479C: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]// 0000000047A0: D3E10074 0DD241B6
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]// 0000000047A8: D3E10078 0DE241B8
	ds_read_b64 v[20:21], v5                                   // 0000000047B0: D8EC0000 14000005
	ds_read_b64 v[22:23], v5 offset:4640                       // 0000000047B8: D8EC1220 16000005
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]// 0000000047C0: D3E1007C 0DF241BA
	ds_read_b64 v[24:25], v5 offset:9280                       // 0000000047C8: D8EC2440 18000005
	ds_read_b64 v[26:27], v5 offset:13920                      // 0000000047D0: D8EC3660 1A000005
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]// 0000000047D8: D3E10080 0E0241BC
	ds_read_b128 a[144:147], v4                                // 0000000047E0: DBFE0000 90000004
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]// 0000000047E8: D3E10084 0E1241BE
	ds_read_b128 a[148:151], v4 offset:64                      // 0000000047F0: DBFE0040 94000004
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]// 0000000047F8: D3E10088 0E2241C0
	ds_read_b128 a[152:155], v4 offset:256                     // 000000004800: DBFE0100 98000004
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]// 000000004808: D3E1008C 0E3241C2
	ds_read_b128 a[156:159], v4 offset:320                     // 000000004810: DBFE0140 9C000004
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]// 000000004818: D3E10090 0E4241C4
	ds_read_b128 a[160:163], v4 offset:512                     // 000000004820: DBFE0200 A0000004
	v_mfma_f32_16x16x16_bf16 v[148:151], a[198:199], v[32:33], v[148:151]// 000000004828: D3E10094 0E5241C6
	ds_read_b128 a[164:167], v4 offset:576                     // 000000004830: DBFE0240 A4000004
	v_mfma_f32_16x16x16_bf16 v[152:155], a[200:201], v[32:33], v[152:155]// 000000004838: D3E10098 0E6241C8
	ds_read_b128 a[168:171], v4 offset:768                     // 000000004840: DBFE0300 A8000004
	v_mfma_f32_16x16x16_bf16 v[156:159], a[202:203], v[32:33], v[156:159]// 000000004848: D3E1009C 0E7241CA
	ds_read_b128 a[172:175], v4 offset:832                     // 000000004850: DBFE0340 AC000004
	v_mfma_f32_16x16x16_bf16 v[160:163], a[204:205], v[32:33], v[160:163]// 000000004858: D3E100A0 0E8241CC
	s_waitcnt lgkmcnt(8)                                       // 000000004860: BF8CC87F
	v_perm_b32 v168, v22, v20, s53                             // 000000004864: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 00000000486C: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000004874: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 00000000487C: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[164:167], a[206:207], v[32:33], v[164:167]// 000000004884: D3E100A4 0E9241CE
	ds_write_b128 v6, v[168:171] offset:37120                  // 00000000488C: D9BE9100 0000A806
	v_perm_b32 v168, v23, v21, s53                             // 000000004894: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 00000000489C: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 0000000048A4: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 0000000048AC: D1ED00AB 00D2331B
	ds_write_b128 v6, v[168:171] offset:38144                  // 0000000048B4: D9BE9500 0000A806
	ds_read_b64 v[20:21], v5 offset:1024                       // 0000000048BC: D8EC0400 14000005
	ds_read_b64 v[22:23], v5 offset:5664                       // 0000000048C4: D8EC1620 16000005
	ds_read_b64 v[24:25], v5 offset:10304                      // 0000000048CC: D8EC2840 18000005
	ds_read_b64 v[26:27], v5 offset:14944                      // 0000000048D4: D8EC3A60 1A000005
	s_nop 0                                                    // 0000000048DC: BF800000
	s_addk_i32 s70, 0x1                                        // 0000000048E0: B7460001
	s_cmp_lt_i32 s70, s71                                      // 0000000048E4: BF044746
	s_cbranch_scc0 label_097C                                  // 0000000048E8: BF840001
	s_branch label_0465                                        // 0000000048EC: BF82FAE9

00000000000048f0 <label_097C>:
	s_nop 0                                                    // 0000000048F0: BF800000
	s_nop 0                                                    // 0000000048F4: BF800000
	s_branch label_0E96                                        // 0000000048F8: BF820517

00000000000048fc <label_097F>:
	s_waitcnt lgkmcnt(4)                                       // 0000000048FC: BF8CC47F
	s_waitcnt vmcnt(0)                                         // 000000004900: BF8C0F70
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0   // 000000004904: D3E10020 1A020190
	buffer_load_dword v10, v8, s[24:27], 0 offen               // 00000000490C: E0501000 80060A08
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]// 000000004914: D3E10020 1C820592
	ds_read_b128 a[176:179], v4 offset:1024                    // 00000000491C: DBFE0400 B0000004
	ds_read_b128 a[180:183], v4 offset:1088                    // 000000004924: DBFE0440 B4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]// 00000000492C: D3E10020 1C820994
	v_cvt_pk_f32_fp8_sdwa v[180:181], v179 src0_sel:WORD_0     // 000000004934: 7F68ACF9 000406B3
	v_cvt_pk_f32_fp8_sdwa v[182:183], v179 src0_sel:WORD_1     // 00000000493C: 7F6CACF9 000506B3
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]// 000000004944: D3E10020 1C820D96
	v_cvt_pk_f32_fp8_sdwa v[186:187], v185 src0_sel:WORD_0     // 00000000494C: 7F74ACF9 000406B9
	v_cvt_pk_f32_fp8_sdwa v[188:189], v185 src0_sel:WORD_1     // 000000004954: 7F78ACF9 000506B9
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]// 00000000495C: D3E10020 1C821198
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]// 000000004964: D3E10020 1C82159A
	ds_read_b128 a[184:187], v4 offset:1280                    // 00000000496C: DBFE0500 B8000004
	ds_read_b128 a[188:191], v4 offset:1344                    // 000000004974: DBFE0540 BC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]// 00000000497C: D3E10020 1C82199C
	v_cvt_pk_f32_fp8_sdwa v[192:193], v191 src0_sel:WORD_0     // 000000004984: 7F80ACF9 000406BF
	v_cvt_pk_f32_fp8_sdwa v[194:195], v191 src0_sel:WORD_1     // 00000000498C: 7F84ACF9 000506BF
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]// 000000004994: D3E10020 1C821D9E
	s_waitcnt lgkmcnt(4)                                       // 00000000499C: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]// 0000000049A0: D3E10020 1C8221A0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]// 0000000049A8: D3E10020 1C8225A2
	ds_read_b128 a[192:195], v4 offset:1536                    // 0000000049B0: DBFE0600 C0000004
	ds_read_b128 a[196:199], v4 offset:1600                    // 0000000049B8: DBFE0640 C4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]// 0000000049C0: D3E10020 1C8229A4
	v_cvt_pk_f32_fp8_sdwa v[198:199], v197 src0_sel:WORD_0     // 0000000049C8: 7F8CACF9 000406C5
	v_cvt_pk_f32_fp8_sdwa v[200:201], v197 src0_sel:WORD_1     // 0000000049D0: 7F90ACF9 000506C5
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]// 0000000049D8: D3E10020 1C822DA6
	v_cvt_pk_f32_fp8_sdwa v[204:205], v203 src0_sel:WORD_0     // 0000000049E0: 7F98ACF9 000406CB
	v_cvt_pk_f32_fp8_sdwa v[206:207], v203 src0_sel:WORD_1     // 0000000049E8: 7F9CACF9 000506CB
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]// 0000000049F0: D3E10020 1C8231A8
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]// 0000000049F8: D3E10020 1C8235AA
	ds_read_b128 a[200:203], v4 offset:1792                    // 000000004A00: DBFE0700 C8000004
	ds_read_b128 a[204:207], v4 offset:1856                    // 000000004A08: DBFE0740 CC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]// 000000004A10: D3E10020 1C8239AC
	v_cvt_pk_f32_fp8_sdwa v[210:211], v209 src0_sel:WORD_0     // 000000004A18: 7FA4ACF9 000406D1
	v_cvt_pk_f32_fp8_sdwa v[212:213], v209 src0_sel:WORD_1     // 000000004A20: 7FA8ACF9 000506D1
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]// 000000004A28: D3E10020 1C823DAE
	s_waitcnt lgkmcnt(4)                                       // 000000004A30: BF8CC47F
	s_barrier                                                  // 000000004A34: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]// 000000004A38: D3E10020 1C8241B0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]// 000000004A40: D3E10020 1C8245B2
	ds_read_b128 a[208:211], v4 offset:2048                    // 000000004A48: DBFE0800 D0000004
	ds_read_b128 a[212:215], v4 offset:2112                    // 000000004A50: DBFE0840 D4000004
	v_cvt_pk_f32_fp8_sdwa v[216:217], v215 src0_sel:WORD_0     // 000000004A58: 7FB0ACF9 000406D7
	v_cvt_pk_f32_fp8_sdwa v[218:219], v215 src0_sel:WORD_1     // 000000004A60: 7FB4ACF9 000506D7
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]// 000000004A68: D3E10020 1C8249B4
	buffer_load_dword v178, v18, s[20:23], 0 offen             // 000000004A70: E0501000 8005B212
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]// 000000004A78: D3E10020 1C824DB6
	v_perm_b32 v168, v22, v20, s53                             // 000000004A80: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 000000004A88: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000004A90: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 000000004A98: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]// 000000004AA0: D3E10020 1C8251B8
	buffer_load_dword v184, v18, s[20:23], 0 offen offset:64   // 000000004AA8: E0501040 8005B812
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]// 000000004AB0: D3E10020 1C8255BA
	ds_write_b128 v6, v[168:171] offset:45312                  // 000000004AB8: D9BEB100 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]// 000000004AC0: D3E10020 1C8259BC
	buffer_load_dword v190, v18, s[20:23], 0 offen offset:128  // 000000004AC8: E0501080 8005BE12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]// 000000004AD0: D3E10020 1C825DBE
	v_perm_b32 v168, v23, v21, s53                             // 000000004AD8: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 000000004AE0: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 000000004AE8: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 000000004AF0: D1ED00AB 00D2331B
	s_waitcnt lgkmcnt(1)                                       // 000000004AF8: BF8CC17F
	s_barrier                                                  // 000000004AFC: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]// 000000004B00: D3E10020 1C8261C0
	buffer_load_dword v196, v18, s[20:23], 0 offen offset:192  // 000000004B08: E05010C0 8005C412
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]// 000000004B10: D3E10020 1C8265C2
	ds_write_b128 v6, v[168:171] offset:46336                  // 000000004B18: D9BEB500 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]// 000000004B20: D3E10020 1C8269C4
	v_cvt_pk_f32_fp8_sdwa v[222:223], v221 src0_sel:WORD_0     // 000000004B28: 7FBCACF9 000406DD
	v_cvt_pk_f32_fp8_sdwa v[224:225], v221 src0_sel:WORD_1     // 000000004B30: 7FC0ACF9 000506DD
	buffer_load_dword v202, v18, s[20:23], 0 offen offset:256  // 000000004B38: E0501100 8005CA12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]// 000000004B40: D3E10020 1C826DC6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]// 000000004B48: D3E10020 1C8271C8
	v_cvt_pk_f32_fp8_sdwa v[228:229], v227 src0_sel:WORD_0     // 000000004B50: 7FC8ACF9 000406E3
	v_cvt_pk_f32_fp8_sdwa v[230:231], v227 src0_sel:WORD_1     // 000000004B58: 7FCCACF9 000506E3
	buffer_load_dword v208, v18, s[20:23], 0 offen offset:320  // 000000004B60: E0501140 8005D012
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]// 000000004B68: D3E10020 1C8275CA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]// 000000004B70: D3E10020 1C8279CC
	v_perm_b32 v180, v181, v180, s52                           // 000000004B78: D1ED00B4 00D369B5
	v_perm_b32 v181, v183, v182, s52                           // 000000004B80: D1ED00B5 00D36DB7
	buffer_load_dword v214, v18, s[20:23], 0 offen offset:384  // 000000004B88: E0501180 8005D612
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]// 000000004B90: D3E10020 1C827DCE
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]// 000000004B98: D3E10020 1C8281D0
	v_perm_b32 v186, v187, v186, s52                           // 000000004BA0: D1ED00BA 00D375BB
	v_perm_b32 v187, v189, v188, s52                           // 000000004BA8: D1ED00BB 00D379BD
	buffer_load_dword v220, v18, s[20:23], 0 offen offset:448  // 000000004BB0: E05011C0 8005DC12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]// 000000004BB8: D3E10020 1C8285D2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]// 000000004BC0: D3E10020 1C8289D4
	buffer_load_dword v226, v18, s[20:23], 0 offen offset:512  // 000000004BC8: E0501200 8005E212
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]// 000000004BD0: D3E10020 1C828DD6
	v_add_u32_e32 v8, s73, v8                                  // 000000004BD8: 68101049
	s_cmp_le_i32 s83, s82                                      // 000000004BDC: BF055253
	s_cbranch_scc1 label_0A5D                                  // 000000004BE0: BF850024
	v_mov_b32_e32 v25, 0xff800000                              // 000000004BE4: 7E3202FF FF800000
	s_add_u32 s57, s82, 0                                      // 000000004BEC: 80398052
	v_mov_b32_e32 v24, s57                                     // 000000004BF0: 7E300239
	v_add_u32_e32 v24, s7, v24                                 // 000000004BF4: 68303007
	s_sub_u32 s56, s83, 15                                     // 000000004BF8: 80B88F53
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000004BFC: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 000000004C00: 0C282884
	v_add_u32_e32 v20, s56, v20                                // 000000004C04: 68282838
	v_add_u32_e32 v21, 1, v20                                  // 000000004C08: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 000000004C0C: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 000000004C10: 682E2883
	v_cmp_le_u32_e64 s[38:39], v20, v24                        // 000000004C14: D0CB0026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 000000004C1C: 682828C0
	s_nop 0                                                    // 000000004C20: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 000000004C24: D1000020 009A4119
	v_cmp_le_u32_e64 s[38:39], v21, v24                        // 000000004C2C: D0CB0026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 000000004C34: 682A2AC0
	s_nop 0                                                    // 000000004C38: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 000000004C3C: D1000021 009A4319
	v_cmp_le_u32_e64 s[38:39], v22, v24                        // 000000004C44: D0CB0026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 000000004C4C: 682C2CC0
	s_nop 0                                                    // 000000004C50: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 000000004C54: D1000022 009A4519
	v_cmp_le_u32_e64 s[38:39], v23, v24                        // 000000004C5C: D0CB0026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 000000004C64: 682E2EC0
	s_nop 0                                                    // 000000004C68: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000004C6C: D1000023 009A4719

0000000000004c74 <label_0A5D>:
	s_waitcnt lgkmcnt(0)                                       // 000000004C74: BF8CC07F
	s_barrier                                                  // 000000004C78: BF8A0000
	v_max3_f32 v24, v32, v33, v32                              // 000000004C7C: D1D30018 04824320
	v_max3_f32 v24, v34, v35, v24                              // 000000004C84: D1D30018 04624722
	ds_write_b32 v3, v24 offset:53504                          // 000000004C8C: D81AD100 00001803
	v_perm_b32 v192, v193, v192, s52                           // 000000004C94: D1ED00C0 00D381C1
	v_perm_b32 v193, v195, v194, s52                           // 000000004C9C: D1ED00C1 00D385C3
	v_perm_b32 v198, v199, v198, s52                           // 000000004CA4: D1ED00C6 00D38DC7
	v_perm_b32 v199, v201, v200, s52                           // 000000004CAC: D1ED00C7 00D391C9
	s_waitcnt lgkmcnt(0)                                       // 000000004CB4: BF8CC07F
	ds_read_b32 v20, v2 offset:53504                           // 000000004CB8: D86CD100 14000002
	ds_read_b32 v21, v2 offset:53568                           // 000000004CC0: D86CD140 15000002
	ds_read_b32 v22, v2 offset:53632                           // 000000004CC8: D86CD180 16000002
	ds_read_b32 v23, v2 offset:53696                           // 000000004CD0: D86CD1C0 17000002
	v_perm_b32 v204, v205, v204, s52                           // 000000004CD8: D1ED00CC 00D399CD
	v_perm_b32 v205, v207, v206, s52                           // 000000004CE0: D1ED00CD 00D39DCF
	v_perm_b32 v210, v211, v210, s52                           // 000000004CE8: D1ED00D2 00D3A5D3
	v_perm_b32 v211, v213, v212, s52                           // 000000004CF0: D1ED00D3 00D3A9D5
	v_perm_b32 v216, v217, v216, s52                           // 000000004CF8: D1ED00D8 00D3B1D9
	v_perm_b32 v217, v219, v218, s52                           // 000000004D00: D1ED00D9 00D3B5DB
	s_waitcnt lgkmcnt(0)                                       // 000000004D08: BF8CC07F
	v_max3_f32 v24, v20, v21, v24                              // 000000004D0C: D1D30018 04622B14
	v_max3_f32 v24, v22, v23, v24                              // 000000004D14: D1D30018 04622F16
	v_perm_b32 v222, v223, v222, s52                           // 000000004D1C: D1ED00DE 00D3BDDF
	v_perm_b32 v223, v225, v224, s52                           // 000000004D24: D1ED00DF 00D3C1E1
	v_perm_b32 v228, v229, v228, s52                           // 000000004D2C: D1ED00E4 00D3C9E5
	v_perm_b32 v229, v231, v230, s52                           // 000000004D34: D1ED00E5 00D3CDE7
	ds_write_b64 v176, v[180:181] offset:18560                 // 000000004D3C: D89A4880 0000B4B0
	ds_read_b128 a[144:147], v7 offset:37120                   // 000000004D44: DBFE9100 90000007
	ds_read_b128 a[148:151], v7 offset:38144                   // 000000004D4C: DBFE9500 94000007
	ds_write_b64 v176, v[186:187] offset:18816                 // 000000004D54: D89A4980 0000BAB0
	ds_read_b128 a[152:155], v7 offset:39168                   // 000000004D5C: DBFE9900 98000007
	ds_read_b128 a[156:159], v7 offset:40192                   // 000000004D64: DBFE9D00 9C000007
	ds_write_b64 v176, v[192:193] offset:19072                 // 000000004D6C: D89A4A80 0000C0B0
	ds_read_b128 a[160:163], v7 offset:41216                   // 000000004D74: DBFEA100 A0000007
	ds_read_b128 a[164:167], v7 offset:42240                   // 000000004D7C: DBFEA500 A4000007
	ds_write_b64 v176, v[198:199] offset:19328                 // 000000004D84: D89A4B80 0000C6B0
	ds_read_b128 a[168:171], v7 offset:43264                   // 000000004D8C: DBFEA900 A8000007
	ds_read_b128 a[172:175], v7 offset:44288                   // 000000004D94: DBFEAD00 AC000007
	v_mov_b32_e32 v25, 0xff7fffff                              // 000000004D9C: 7E3202FF FF7FFFFF
	v_cmp_eq_u32_e64 s[38:39], v25, v12                        // 000000004DA4: D0CA0026 00021919
	v_max_f32_e32 v20, v24, v12                                // 000000004DAC: 16281918
	v_sub_f32_e32 v16, v12, v20                                // 000000004DB0: 0420290C
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]                    // 000000004DB4: D1000010 00990110
	v_mov_b32_e32 v12, v20                                     // 000000004DBC: 7E180314
	v_mul_f32_e32 v21, s5, v20                                 // 000000004DC0: 0A2A2805
	v_mul_f32_e32 v16, s5, v16                                 // 000000004DC4: 0A202005
	v_exp_f32_e32 v16, v16                                     // 000000004DC8: 7E204110
	v_fma_f32 v32, v32, s5, -v21                               // 000000004DCC: D1CB0020 84540B20
	v_fma_f32 v33, v33, s5, -v21                               // 000000004DD4: D1CB0021 84540B21
	v_fma_f32 v34, v34, s5, -v21                               // 000000004DDC: D1CB0022 84540B22
	v_fma_f32 v35, v35, s5, -v21                               // 000000004DE4: D1CB0023 84540B23
	v_exp_f32_e32 v32, v32                                     // 000000004DEC: 7E404120
	v_exp_f32_e32 v33, v33                                     // 000000004DF0: 7E424121
	v_exp_f32_e32 v34, v34                                     // 000000004DF4: 7E444122
	v_exp_f32_e32 v35, v35                                     // 000000004DF8: 7E464123
	v_mul_f32_e32 v14, v16, v14                                // 000000004DFC: 0A1C1D10
	v_mov_b32_e32 v22, v32                                     // 000000004E00: 7E2C0320
	v_add_f32_e32 v22, v33, v22                                // 000000004E04: 022C2D21
	v_add_f32_e32 v22, v34, v22                                // 000000004E08: 022C2D22
	v_add_f32_e32 v22, v35, v22                                // 000000004E0C: 022C2D23
	v_add_f32_e32 v14, v22, v14                                // 000000004E10: 021C1D16
	v_mov_b32_e32 v29, 0xffff0000                              // 000000004E14: 7E3A02FF FFFF0000
	v_mov_b32_e32 v30, 0x7fff0000                              // 000000004E1C: 7E3C02FF 7FFF0000
	v_mov_b32_e32 v31, 0x7fff                                  // 000000004E24: 7E3E02FF 00007FFF
	v_cmp_u_f32_e64 s[38:39], v32, v32                         // 000000004E2C: D0480026 00024120
	v_add3_u32 v28, v32, v31, 1                                // 000000004E34: D1FF001C 02063F20
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000004E3C: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v33, v33                         // 000000004E44: D0480026 00024321
	v_add3_u32 v28, v33, v31, 1                                // 000000004E4C: D1FF001C 02063F21
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000004E54: D1000015 009A3D1C
	v_perm_b32 v32, v21, v20, s52                              // 000000004E5C: D1ED0020 00D22915
	v_cmp_u_f32_e64 s[38:39], v34, v34                         // 000000004E64: D0480026 00024522
	v_add3_u32 v28, v34, v31, 1                                // 000000004E6C: D1FF001C 02063F22
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000004E74: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v35, v35                         // 000000004E7C: D0480026 00024723
	v_add3_u32 v28, v35, v31, 1                                // 000000004E84: D1FF001C 02063F23
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000004E8C: D1000015 009A3D1C
	v_perm_b32 v33, v21, v20, s52                              // 000000004E94: D1ED0021 00D22915
	s_nop 2                                                    // 000000004E9C: BF800002
	s_add_u32 s83, s84, s83                                    // 000000004EA0: 80535354
	s_nop 0                                                    // 000000004EA4: BF800000
	v_mul_u32_u24_e32 v18, v11, v9                             // 000000004EA8: 1024130B
	v_add_u32_e32 v18, v18, v1                                 // 000000004EAC: 68240312
	s_mov_b32 m0, s37                                          // 000000004EB0: BEFC0025
	v_mov_b32_e32 v22, v16                                     // 000000004EB4: 7E2C0310
	v_mov_b32_e32 v23, v16                                     // 000000004EB8: 7E2E0310
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]                  // 000000004EBC: D3B14028 18025116
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]                  // 000000004EC4: D3B1402A 18025516
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]                  // 000000004ECC: D3B1402C 18025916
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]                  // 000000004ED4: D3B1402E 18025D16
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]                  // 000000004EDC: D3B14030 18026116
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]                  // 000000004EE4: D3B14032 18026516
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]                  // 000000004EEC: D3B14034 18026916
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]                  // 000000004EF4: D3B14036 18026D16
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]                  // 000000004EFC: D3B14038 18027116
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]                  // 000000004F04: D3B1403A 18027516
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]                  // 000000004F0C: D3B1403C 18027916
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]                  // 000000004F14: D3B1403E 18027D16
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]                  // 000000004F1C: D3B14040 18028116
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]                  // 000000004F24: D3B14042 18028516
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]                  // 000000004F2C: D3B14044 18028916
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]                  // 000000004F34: D3B14046 18028D16
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]                  // 000000004F3C: D3B14048 18029116
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]                  // 000000004F44: D3B1404A 18029516
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]                  // 000000004F4C: D3B1404C 18029916
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]                  // 000000004F54: D3B1404E 18029D16
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]                  // 000000004F5C: D3B14050 1802A116
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]                  // 000000004F64: D3B14052 1802A516
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]                  // 000000004F6C: D3B14054 1802A916
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]                  // 000000004F74: D3B14056 1802AD16
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]                  // 000000004F7C: D3B14058 1802B116
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]                  // 000000004F84: D3B1405A 1802B516
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]                  // 000000004F8C: D3B1405C 1802B916
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]                  // 000000004F94: D3B1405E 1802BD16
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]                  // 000000004F9C: D3B14060 1802C116
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]                  // 000000004FA4: D3B14062 1802C516
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]              // 000000004FAC: D3B14064 1802C916
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]              // 000000004FB4: D3B14066 1802CD16
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]              // 000000004FBC: D3B14068 1802D116
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]              // 000000004FC4: D3B1406A 1802D516
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]              // 000000004FCC: D3B1406C 1802D916
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]              // 000000004FD4: D3B1406E 1802DD16
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]              // 000000004FDC: D3B14070 1802E116
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]              // 000000004FE4: D3B14072 1802E516
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]              // 000000004FEC: D3B14074 1802E916
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]              // 000000004FF4: D3B14076 1802ED16
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]              // 000000004FFC: D3B14078 1802F116
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]              // 000000005004: D3B1407A 1802F516
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]              // 00000000500C: D3B1407C 1802F916
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]              // 000000005014: D3B1407E 1802FD16
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]              // 00000000501C: D3B14080 18030116
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]              // 000000005024: D3B14082 18030516
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]              // 00000000502C: D3B14084 18030916
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]              // 000000005034: D3B14086 18030D16
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]              // 00000000503C: D3B14088 18031116
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]              // 000000005044: D3B1408A 18031516
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]              // 00000000504C: D3B1408C 18031916
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]              // 000000005054: D3B1408E 18031D16
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]              // 00000000505C: D3B14090 18032116
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]              // 000000005064: D3B14092 18032516
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]              // 00000000506C: D3B14094 18032916
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]              // 000000005074: D3B14096 18032D16
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]              // 00000000507C: D3B14098 18033116
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]              // 000000005084: D3B1409A 18033516
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]              // 00000000508C: D3B1409C 18033916
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]              // 000000005094: D3B1409E 18033D16
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]              // 00000000509C: D3B140A0 18034116
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]              // 0000000050A4: D3B140A2 18034516
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]              // 0000000050AC: D3B140A4 18034916
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]              // 0000000050B4: D3B140A6 18034D16
	s_waitcnt lgkmcnt(0)                                       // 0000000050BC: BF8CC07F
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]// 0000000050C0: D3E10028 0CA24190
	ds_write_b64 v176, v[192:193] offset:19072                 // 0000000050C8: D89A4A80 0000C0B0
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]// 0000000050D0: D3E1002C 0CB24192
	ds_read_b128 a[176:179], v7 offset:45312                   // 0000000050D8: DBFEB100 B0000007
	ds_read_b128 a[180:183], v7 offset:46336                   // 0000000050E0: DBFEB500 B4000007
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]// 0000000050E8: D3E10030 0CC24194
	ds_write_b64 v176, v[198:199] offset:19328                 // 0000000050F0: D89A4B80 0000C6B0
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]// 0000000050F8: D3E10034 0CD24196
	ds_write_b64 v176, v[204:205] offset:19584                 // 000000005100: D89A4C80 0000CCB0
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]// 000000005108: D3E10038 0CE24198
	ds_write_b64 v176, v[210:211] offset:19840                 // 000000005110: D89A4D80 0000D2B0
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]// 000000005118: D3E1003C 0CF2419A
	ds_read_b128 a[184:187], v7 offset:47360                   // 000000005120: DBFEB900 B8000007
	ds_read_b128 a[188:191], v7 offset:48384                   // 000000005128: DBFEBD00 BC000007
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]// 000000005130: D3E10040 0D02419C
	ds_write_b64 v176, v[216:217] offset:20096                 // 000000005138: D89A4E80 0000D8B0
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]// 000000005140: D3E10044 0D12419E
	ds_write_b64 v176, v[222:223] offset:20352                 // 000000005148: D89A4F80 0000DEB0
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]// 000000005150: D3E10048 0D2241A0
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]// 000000005158: D3E1004C 0D3241A2
	ds_read_b128 a[192:195], v7 offset:49408                   // 000000005160: DBFEC100 C0000007
	ds_read_b128 a[196:199], v7 offset:50432                   // 000000005168: DBFEC500 C4000007
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]// 000000005170: D3E10050 0D4241A4
	ds_write_b64 v176, v[228:229] offset:20608                 // 000000005178: D89A5080 0000E4B0
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]// 000000005180: D3E10054 0D5241A6
	s_waitcnt lgkmcnt(4)                                       // 000000005188: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]// 00000000518C: D3E10058 0D6241A8
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]// 000000005194: D3E1005C 0D7241AA
	ds_read_b128 a[200:203], v7 offset:51456                   // 00000000519C: DBFEC900 C8000007
	ds_read_b128 a[204:207], v7 offset:52480                   // 0000000051A4: DBFECD00 CC000007
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]// 0000000051AC: D3E10060 0D8241AC
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]// 0000000051B4: D3E10064 0D9241AE
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]// 0000000051BC: D3E10068 0DA241B0
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]// 0000000051C4: D3E1006C 0DB241B2
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]// 0000000051CC: D3E10070 0DC241B4
	s_waitcnt vmcnt(9) lgkmcnt(9)                              // 0000000051D4: BF8C0979
	s_barrier                                                  // 0000000051D8: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]// 0000000051DC: D3E10074 0DD241B6
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]// 0000000051E4: D3E10078 0DE241B8
	ds_read_b64 v[20:21], v5 offset:18560                      // 0000000051EC: D8EC4880 14000005
	ds_read_b64 v[22:23], v5 offset:23200                      // 0000000051F4: D8EC5AA0 16000005
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]// 0000000051FC: D3E1007C 0DF241BA
	ds_read_b64 v[24:25], v5 offset:27840                      // 000000005204: D8EC6CC0 18000005
	ds_read_b64 v[26:27], v5 offset:32480                      // 00000000520C: D8EC7EE0 1A000005
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]// 000000005214: D3E10080 0E0241BC
	ds_read_b128 a[144:147], v4 offset:18560                   // 00000000521C: DBFE4880 90000004
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]// 000000005224: D3E10084 0E1241BE
	ds_read_b128 a[148:151], v4 offset:18624                   // 00000000522C: DBFE48C0 94000004
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]// 000000005234: D3E10088 0E2241C0
	ds_read_b128 a[152:155], v4 offset:18816                   // 00000000523C: DBFE4980 98000004
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]// 000000005244: D3E1008C 0E3241C2
	ds_read_b128 a[156:159], v4 offset:18880                   // 00000000524C: DBFE49C0 9C000004
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]// 000000005254: D3E10090 0E4241C4
	ds_read_b128 a[160:163], v4 offset:19072                   // 00000000525C: DBFE4A80 A0000004
	v_mfma_f32_16x16x16_bf16 v[148:151], a[198:199], v[32:33], v[148:151]// 000000005264: D3E10094 0E5241C6
	ds_read_b128 a[164:167], v4 offset:19136                   // 00000000526C: DBFE4AC0 A4000004
	v_mfma_f32_16x16x16_bf16 v[152:155], a[200:201], v[32:33], v[152:155]// 000000005274: D3E10098 0E6241C8
	ds_read_b128 a[168:171], v4 offset:19328                   // 00000000527C: DBFE4B80 A8000004
	v_mfma_f32_16x16x16_bf16 v[156:159], a[202:203], v[32:33], v[156:159]// 000000005284: D3E1009C 0E7241CA
	ds_read_b128 a[172:175], v4 offset:19392                   // 00000000528C: DBFE4BC0 AC000004
	v_mfma_f32_16x16x16_bf16 v[160:163], a[204:205], v[32:33], v[160:163]// 000000005294: D3E100A0 0E8241CC
	s_waitcnt lgkmcnt(8)                                       // 00000000529C: BF8CC87F
	v_perm_b32 v168, v22, v20, s53                             // 0000000052A0: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 0000000052A8: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 0000000052B0: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 0000000052B8: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[164:167], a[206:207], v[32:33], v[164:167]// 0000000052C0: D3E100A4 0E9241CE
	ds_write_b128 v6, v[168:171] offset:37120                  // 0000000052C8: D9BE9100 0000A806
	v_perm_b32 v168, v23, v21, s53                             // 0000000052D0: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 0000000052D8: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 0000000052E0: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 0000000052E8: D1ED00AB 00D2331B
	ds_write_b128 v6, v[168:171] offset:38144                  // 0000000052F0: D9BE9500 0000A806
	ds_read_b64 v[20:21], v5 offset:19584                      // 0000000052F8: D8EC4C80 14000005
	ds_read_b64 v[22:23], v5 offset:24224                      // 000000005300: D8EC5EA0 16000005
	ds_read_b64 v[24:25], v5 offset:28864                      // 000000005308: D8EC70C0 18000005
	ds_read_b64 v[26:27], v5 offset:33504                      // 000000005310: D8EC82E0 1A000005
	s_nop 0                                                    // 000000005318: BF800000
	s_addk_i32 s70, 0x1                                        // 00000000531C: B7460001
	s_cmp_lt_i32 s70, s71                                      // 000000005320: BF044746
	s_cbranch_scc0 label_097C                                  // 000000005324: BF84FD72
	s_waitcnt lgkmcnt(4)                                       // 000000005328: BF8CC47F
	s_waitcnt vmcnt(0)                                         // 00000000532C: BF8C0F70
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0   // 000000005330: D3E10020 1A020190
	buffer_load_dword v11, v8, s[24:27], 0 offen               // 000000005338: E0501000 80060B08
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]// 000000005340: D3E10020 1C820592
	ds_read_b128 a[176:179], v4 offset:19584                   // 000000005348: DBFE4C80 B0000004
	ds_read_b128 a[180:183], v4 offset:19648                   // 000000005350: DBFE4CC0 B4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]// 000000005358: D3E10020 1C820994
	v_cvt_pk_f32_fp8_sdwa v[180:181], v178 src0_sel:WORD_0     // 000000005360: 7F68ACF9 000406B2
	v_cvt_pk_f32_fp8_sdwa v[182:183], v178 src0_sel:WORD_1     // 000000005368: 7F6CACF9 000506B2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]// 000000005370: D3E10020 1C820D96
	v_cvt_pk_f32_fp8_sdwa v[186:187], v184 src0_sel:WORD_0     // 000000005378: 7F74ACF9 000406B8
	v_cvt_pk_f32_fp8_sdwa v[188:189], v184 src0_sel:WORD_1     // 000000005380: 7F78ACF9 000506B8
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]// 000000005388: D3E10020 1C821198
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]// 000000005390: D3E10020 1C82159A
	ds_read_b128 a[184:187], v4 offset:19840                   // 000000005398: DBFE4D80 B8000004
	ds_read_b128 a[188:191], v4 offset:19904                   // 0000000053A0: DBFE4DC0 BC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]// 0000000053A8: D3E10020 1C82199C
	v_cvt_pk_f32_fp8_sdwa v[192:193], v190 src0_sel:WORD_0     // 0000000053B0: 7F80ACF9 000406BE
	v_cvt_pk_f32_fp8_sdwa v[194:195], v190 src0_sel:WORD_1     // 0000000053B8: 7F84ACF9 000506BE
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]// 0000000053C0: D3E10020 1C821D9E
	s_waitcnt lgkmcnt(4)                                       // 0000000053C8: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]// 0000000053CC: D3E10020 1C8221A0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]// 0000000053D4: D3E10020 1C8225A2
	ds_read_b128 a[192:195], v4 offset:20096                   // 0000000053DC: DBFE4E80 C0000004
	ds_read_b128 a[196:199], v4 offset:20160                   // 0000000053E4: DBFE4EC0 C4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]// 0000000053EC: D3E10020 1C8229A4
	v_cvt_pk_f32_fp8_sdwa v[198:199], v196 src0_sel:WORD_0     // 0000000053F4: 7F8CACF9 000406C4
	v_cvt_pk_f32_fp8_sdwa v[200:201], v196 src0_sel:WORD_1     // 0000000053FC: 7F90ACF9 000506C4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]// 000000005404: D3E10020 1C822DA6
	v_cvt_pk_f32_fp8_sdwa v[204:205], v202 src0_sel:WORD_0     // 00000000540C: 7F98ACF9 000406CA
	v_cvt_pk_f32_fp8_sdwa v[206:207], v202 src0_sel:WORD_1     // 000000005414: 7F9CACF9 000506CA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]// 00000000541C: D3E10020 1C8231A8
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]// 000000005424: D3E10020 1C8235AA
	ds_read_b128 a[200:203], v4 offset:20352                   // 00000000542C: DBFE4F80 C8000004
	ds_read_b128 a[204:207], v4 offset:20416                   // 000000005434: DBFE4FC0 CC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]// 00000000543C: D3E10020 1C8239AC
	v_cvt_pk_f32_fp8_sdwa v[210:211], v208 src0_sel:WORD_0     // 000000005444: 7FA4ACF9 000406D0
	v_cvt_pk_f32_fp8_sdwa v[212:213], v208 src0_sel:WORD_1     // 00000000544C: 7FA8ACF9 000506D0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]// 000000005454: D3E10020 1C823DAE
	s_waitcnt lgkmcnt(4)                                       // 00000000545C: BF8CC47F
	s_barrier                                                  // 000000005460: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]// 000000005464: D3E10020 1C8241B0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]// 00000000546C: D3E10020 1C8245B2
	ds_read_b128 a[208:211], v4 offset:20608                   // 000000005474: DBFE5080 D0000004
	ds_read_b128 a[212:215], v4 offset:20672                   // 00000000547C: DBFE50C0 D4000004
	v_cvt_pk_f32_fp8_sdwa v[216:217], v214 src0_sel:WORD_0     // 000000005484: 7FB0ACF9 000406D6
	v_cvt_pk_f32_fp8_sdwa v[218:219], v214 src0_sel:WORD_1     // 00000000548C: 7FB4ACF9 000506D6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]// 000000005494: D3E10020 1C8249B4
	buffer_load_dword v179, v18, s[20:23], 0 offen             // 00000000549C: E0501000 8005B312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]// 0000000054A4: D3E10020 1C824DB6
	v_perm_b32 v168, v22, v20, s53                             // 0000000054AC: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 0000000054B4: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 0000000054BC: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 0000000054C4: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]// 0000000054CC: D3E10020 1C8251B8
	buffer_load_dword v185, v18, s[20:23], 0 offen offset:64   // 0000000054D4: E0501040 8005B912
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]// 0000000054DC: D3E10020 1C8255BA
	ds_write_b128 v6, v[168:171] offset:45312                  // 0000000054E4: D9BEB100 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]// 0000000054EC: D3E10020 1C8259BC
	buffer_load_dword v191, v18, s[20:23], 0 offen offset:128  // 0000000054F4: E0501080 8005BF12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]// 0000000054FC: D3E10020 1C825DBE
	v_perm_b32 v168, v23, v21, s53                             // 000000005504: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 00000000550C: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 000000005514: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 00000000551C: D1ED00AB 00D2331B
	s_waitcnt lgkmcnt(1)                                       // 000000005524: BF8CC17F
	s_barrier                                                  // 000000005528: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]// 00000000552C: D3E10020 1C8261C0
	buffer_load_dword v197, v18, s[20:23], 0 offen offset:192  // 000000005534: E05010C0 8005C512
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]// 00000000553C: D3E10020 1C8265C2
	ds_write_b128 v6, v[168:171] offset:46336                  // 000000005544: D9BEB500 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]// 00000000554C: D3E10020 1C8269C4
	v_cvt_pk_f32_fp8_sdwa v[222:223], v220 src0_sel:WORD_0     // 000000005554: 7FBCACF9 000406DC
	v_cvt_pk_f32_fp8_sdwa v[224:225], v220 src0_sel:WORD_1     // 00000000555C: 7FC0ACF9 000506DC
	buffer_load_dword v203, v18, s[20:23], 0 offen offset:256  // 000000005564: E0501100 8005CB12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]// 00000000556C: D3E10020 1C826DC6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]// 000000005574: D3E10020 1C8271C8
	v_cvt_pk_f32_fp8_sdwa v[228:229], v226 src0_sel:WORD_0     // 00000000557C: 7FC8ACF9 000406E2
	v_cvt_pk_f32_fp8_sdwa v[230:231], v226 src0_sel:WORD_1     // 000000005584: 7FCCACF9 000506E2
	buffer_load_dword v209, v18, s[20:23], 0 offen offset:320  // 00000000558C: E0501140 8005D112
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]// 000000005594: D3E10020 1C8275CA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]// 00000000559C: D3E10020 1C8279CC
	v_perm_b32 v180, v181, v180, s52                           // 0000000055A4: D1ED00B4 00D369B5
	v_perm_b32 v181, v183, v182, s52                           // 0000000055AC: D1ED00B5 00D36DB7
	buffer_load_dword v215, v18, s[20:23], 0 offen offset:384  // 0000000055B4: E0501180 8005D712
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]// 0000000055BC: D3E10020 1C827DCE
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]// 0000000055C4: D3E10020 1C8281D0
	v_perm_b32 v186, v187, v186, s52                           // 0000000055CC: D1ED00BA 00D375BB
	v_perm_b32 v187, v189, v188, s52                           // 0000000055D4: D1ED00BB 00D379BD
	buffer_load_dword v221, v18, s[20:23], 0 offen offset:448  // 0000000055DC: E05011C0 8005DD12
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]// 0000000055E4: D3E10020 1C8285D2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]// 0000000055EC: D3E10020 1C8289D4
	buffer_load_dword v227, v18, s[20:23], 0 offen offset:512  // 0000000055F4: E0501200 8005E312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]// 0000000055FC: D3E10020 1C828DD6
	v_add_u32_e32 v8, s73, v8                                  // 000000005604: 68101049
	s_cmp_le_i32 s83, s82                                      // 000000005608: BF055253
	s_cbranch_scc1 label_0CE8                                  // 00000000560C: BF850024
	v_mov_b32_e32 v25, 0xff800000                              // 000000005610: 7E3202FF FF800000
	s_add_u32 s57, s82, 0                                      // 000000005618: 80398052
	v_mov_b32_e32 v24, s57                                     // 00000000561C: 7E300239
	v_add_u32_e32 v24, s7, v24                                 // 000000005620: 68303007
	s_sub_u32 s56, s83, 15                                     // 000000005624: 80B88F53
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000005628: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 00000000562C: 0C282884
	v_add_u32_e32 v20, s56, v20                                // 000000005630: 68282838
	v_add_u32_e32 v21, 1, v20                                  // 000000005634: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 000000005638: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 00000000563C: 682E2883
	v_cmp_le_u32_e64 s[38:39], v20, v24                        // 000000005640: D0CB0026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 000000005648: 682828C0
	s_nop 0                                                    // 00000000564C: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 000000005650: D1000020 009A4119
	v_cmp_le_u32_e64 s[38:39], v21, v24                        // 000000005658: D0CB0026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 000000005660: 682A2AC0
	s_nop 0                                                    // 000000005664: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 000000005668: D1000021 009A4319
	v_cmp_le_u32_e64 s[38:39], v22, v24                        // 000000005670: D0CB0026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 000000005678: 682C2CC0
	s_nop 0                                                    // 00000000567C: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 000000005680: D1000022 009A4519
	v_cmp_le_u32_e64 s[38:39], v23, v24                        // 000000005688: D0CB0026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 000000005690: 682E2EC0
	s_nop 0                                                    // 000000005694: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000005698: D1000023 009A4719

00000000000056a0 <label_0CE8>:
	s_waitcnt lgkmcnt(0)                                       // 0000000056A0: BF8CC07F
	s_barrier                                                  // 0000000056A4: BF8A0000
	v_max3_f32 v24, v32, v33, v32                              // 0000000056A8: D1D30018 04824320
	v_max3_f32 v24, v34, v35, v24                              // 0000000056B0: D1D30018 04624722
	ds_write_b32 v3, v24 offset:53504                          // 0000000056B8: D81AD100 00001803
	v_perm_b32 v192, v193, v192, s52                           // 0000000056C0: D1ED00C0 00D381C1
	v_perm_b32 v193, v195, v194, s52                           // 0000000056C8: D1ED00C1 00D385C3
	v_perm_b32 v198, v199, v198, s52                           // 0000000056D0: D1ED00C6 00D38DC7
	v_perm_b32 v199, v201, v200, s52                           // 0000000056D8: D1ED00C7 00D391C9
	s_waitcnt lgkmcnt(0)                                       // 0000000056E0: BF8CC07F
	ds_read_b32 v20, v2 offset:53504                           // 0000000056E4: D86CD100 14000002
	ds_read_b32 v21, v2 offset:53568                           // 0000000056EC: D86CD140 15000002
	ds_read_b32 v22, v2 offset:53632                           // 0000000056F4: D86CD180 16000002
	ds_read_b32 v23, v2 offset:53696                           // 0000000056FC: D86CD1C0 17000002
	v_perm_b32 v204, v205, v204, s52                           // 000000005704: D1ED00CC 00D399CD
	v_perm_b32 v205, v207, v206, s52                           // 00000000570C: D1ED00CD 00D39DCF
	v_perm_b32 v210, v211, v210, s52                           // 000000005714: D1ED00D2 00D3A5D3
	v_perm_b32 v211, v213, v212, s52                           // 00000000571C: D1ED00D3 00D3A9D5
	v_perm_b32 v216, v217, v216, s52                           // 000000005724: D1ED00D8 00D3B1D9
	v_perm_b32 v217, v219, v218, s52                           // 00000000572C: D1ED00D9 00D3B5DB
	s_waitcnt lgkmcnt(0)                                       // 000000005734: BF8CC07F
	v_max3_f32 v24, v20, v21, v24                              // 000000005738: D1D30018 04622B14
	v_max3_f32 v24, v22, v23, v24                              // 000000005740: D1D30018 04622F16
	v_perm_b32 v222, v223, v222, s52                           // 000000005748: D1ED00DE 00D3BDDF
	v_perm_b32 v223, v225, v224, s52                           // 000000005750: D1ED00DF 00D3C1E1
	v_perm_b32 v228, v229, v228, s52                           // 000000005758: D1ED00E4 00D3C9E5
	v_perm_b32 v229, v231, v230, s52                           // 000000005760: D1ED00E5 00D3CDE7
	ds_write_b64 v176, v[180:181]                              // 000000005768: D89A0000 0000B4B0
	ds_read_b128 a[144:147], v7 offset:37120                   // 000000005770: DBFE9100 90000007
	ds_read_b128 a[148:151], v7 offset:38144                   // 000000005778: DBFE9500 94000007
	ds_write_b64 v176, v[186:187] offset:256                   // 000000005780: D89A0100 0000BAB0
	ds_read_b128 a[152:155], v7 offset:39168                   // 000000005788: DBFE9900 98000007
	ds_read_b128 a[156:159], v7 offset:40192                   // 000000005790: DBFE9D00 9C000007
	ds_write_b64 v176, v[192:193] offset:512                   // 000000005798: D89A0200 0000C0B0
	ds_read_b128 a[160:163], v7 offset:41216                   // 0000000057A0: DBFEA100 A0000007
	ds_read_b128 a[164:167], v7 offset:42240                   // 0000000057A8: DBFEA500 A4000007
	ds_write_b64 v176, v[198:199] offset:768                   // 0000000057B0: D89A0300 0000C6B0
	ds_read_b128 a[168:171], v7 offset:43264                   // 0000000057B8: DBFEA900 A8000007
	ds_read_b128 a[172:175], v7 offset:44288                   // 0000000057C0: DBFEAD00 AC000007
	v_mov_b32_e32 v25, 0xff7fffff                              // 0000000057C8: 7E3202FF FF7FFFFF
	v_cmp_eq_u32_e64 s[38:39], v25, v12                        // 0000000057D0: D0CA0026 00021919
	v_max_f32_e32 v20, v24, v12                                // 0000000057D8: 16281918
	v_sub_f32_e32 v16, v12, v20                                // 0000000057DC: 0420290C
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]                    // 0000000057E0: D1000010 00990110
	v_mov_b32_e32 v12, v20                                     // 0000000057E8: 7E180314
	v_mul_f32_e32 v21, s5, v20                                 // 0000000057EC: 0A2A2805
	v_mul_f32_e32 v16, s5, v16                                 // 0000000057F0: 0A202005
	v_exp_f32_e32 v16, v16                                     // 0000000057F4: 7E204110
	v_fma_f32 v32, v32, s5, -v21                               // 0000000057F8: D1CB0020 84540B20
	v_fma_f32 v33, v33, s5, -v21                               // 000000005800: D1CB0021 84540B21
	v_fma_f32 v34, v34, s5, -v21                               // 000000005808: D1CB0022 84540B22
	v_fma_f32 v35, v35, s5, -v21                               // 000000005810: D1CB0023 84540B23
	v_exp_f32_e32 v32, v32                                     // 000000005818: 7E404120
	v_exp_f32_e32 v33, v33                                     // 00000000581C: 7E424121
	v_exp_f32_e32 v34, v34                                     // 000000005820: 7E444122
	v_exp_f32_e32 v35, v35                                     // 000000005824: 7E464123
	v_mul_f32_e32 v14, v16, v14                                // 000000005828: 0A1C1D10
	v_mov_b32_e32 v22, v32                                     // 00000000582C: 7E2C0320
	v_add_f32_e32 v22, v33, v22                                // 000000005830: 022C2D21
	v_add_f32_e32 v22, v34, v22                                // 000000005834: 022C2D22
	v_add_f32_e32 v22, v35, v22                                // 000000005838: 022C2D23
	v_add_f32_e32 v14, v22, v14                                // 00000000583C: 021C1D16
	v_mov_b32_e32 v29, 0xffff0000                              // 000000005840: 7E3A02FF FFFF0000
	v_mov_b32_e32 v30, 0x7fff0000                              // 000000005848: 7E3C02FF 7FFF0000
	v_mov_b32_e32 v31, 0x7fff                                  // 000000005850: 7E3E02FF 00007FFF
	v_cmp_u_f32_e64 s[38:39], v32, v32                         // 000000005858: D0480026 00024120
	v_add3_u32 v28, v32, v31, 1                                // 000000005860: D1FF001C 02063F20
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000005868: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v33, v33                         // 000000005870: D0480026 00024321
	v_add3_u32 v28, v33, v31, 1                                // 000000005878: D1FF001C 02063F21
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000005880: D1000015 009A3D1C
	v_perm_b32 v32, v21, v20, s52                              // 000000005888: D1ED0020 00D22915
	v_cmp_u_f32_e64 s[38:39], v34, v34                         // 000000005890: D0480026 00024522
	v_add3_u32 v28, v34, v31, 1                                // 000000005898: D1FF001C 02063F22
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000058A0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v35, v35                         // 0000000058A8: D0480026 00024723
	v_add3_u32 v28, v35, v31, 1                                // 0000000058B0: D1FF001C 02063F23
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000058B8: D1000015 009A3D1C
	v_perm_b32 v33, v21, v20, s52                              // 0000000058C0: D1ED0021 00D22915
	s_nop 2                                                    // 0000000058C8: BF800002
	s_add_u32 s83, s84, s83                                    // 0000000058CC: 80535354
	s_nop 0                                                    // 0000000058D0: BF800000
	v_mul_u32_u24_e32 v18, v10, v9                             // 0000000058D4: 1024130A
	v_add_u32_e32 v18, v18, v1                                 // 0000000058D8: 68240312
	s_mov_b32 m0, s35                                          // 0000000058DC: BEFC0023
	v_mov_b32_e32 v22, v16                                     // 0000000058E0: 7E2C0310
	v_mov_b32_e32 v23, v16                                     // 0000000058E4: 7E2E0310
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]                  // 0000000058E8: D3B14028 18025116
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]                  // 0000000058F0: D3B1402A 18025516
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]                  // 0000000058F8: D3B1402C 18025916
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]                  // 000000005900: D3B1402E 18025D16
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]                  // 000000005908: D3B14030 18026116
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]                  // 000000005910: D3B14032 18026516
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]                  // 000000005918: D3B14034 18026916
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]                  // 000000005920: D3B14036 18026D16
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]                  // 000000005928: D3B14038 18027116
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]                  // 000000005930: D3B1403A 18027516
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]                  // 000000005938: D3B1403C 18027916
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]                  // 000000005940: D3B1403E 18027D16
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]                  // 000000005948: D3B14040 18028116
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]                  // 000000005950: D3B14042 18028516
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]                  // 000000005958: D3B14044 18028916
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]                  // 000000005960: D3B14046 18028D16
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]                  // 000000005968: D3B14048 18029116
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]                  // 000000005970: D3B1404A 18029516
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]                  // 000000005978: D3B1404C 18029916
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]                  // 000000005980: D3B1404E 18029D16
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]                  // 000000005988: D3B14050 1802A116
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]                  // 000000005990: D3B14052 1802A516
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]                  // 000000005998: D3B14054 1802A916
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]                  // 0000000059A0: D3B14056 1802AD16
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]                  // 0000000059A8: D3B14058 1802B116
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]                  // 0000000059B0: D3B1405A 1802B516
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]                  // 0000000059B8: D3B1405C 1802B916
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]                  // 0000000059C0: D3B1405E 1802BD16
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]                  // 0000000059C8: D3B14060 1802C116
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]                  // 0000000059D0: D3B14062 1802C516
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]              // 0000000059D8: D3B14064 1802C916
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]              // 0000000059E0: D3B14066 1802CD16
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]              // 0000000059E8: D3B14068 1802D116
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]              // 0000000059F0: D3B1406A 1802D516
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]              // 0000000059F8: D3B1406C 1802D916
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]              // 000000005A00: D3B1406E 1802DD16
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]              // 000000005A08: D3B14070 1802E116
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]              // 000000005A10: D3B14072 1802E516
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]              // 000000005A18: D3B14074 1802E916
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]              // 000000005A20: D3B14076 1802ED16
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]              // 000000005A28: D3B14078 1802F116
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]              // 000000005A30: D3B1407A 1802F516
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]              // 000000005A38: D3B1407C 1802F916
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]              // 000000005A40: D3B1407E 1802FD16
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]              // 000000005A48: D3B14080 18030116
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]              // 000000005A50: D3B14082 18030516
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]              // 000000005A58: D3B14084 18030916
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]              // 000000005A60: D3B14086 18030D16
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]              // 000000005A68: D3B14088 18031116
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]              // 000000005A70: D3B1408A 18031516
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]              // 000000005A78: D3B1408C 18031916
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]              // 000000005A80: D3B1408E 18031D16
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]              // 000000005A88: D3B14090 18032116
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]              // 000000005A90: D3B14092 18032516
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]              // 000000005A98: D3B14094 18032916
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]              // 000000005AA0: D3B14096 18032D16
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]              // 000000005AA8: D3B14098 18033116
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]              // 000000005AB0: D3B1409A 18033516
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]              // 000000005AB8: D3B1409C 18033916
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]              // 000000005AC0: D3B1409E 18033D16
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]              // 000000005AC8: D3B140A0 18034116
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]              // 000000005AD0: D3B140A2 18034516
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]              // 000000005AD8: D3B140A4 18034916
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]              // 000000005AE0: D3B140A6 18034D16
	s_waitcnt lgkmcnt(0)                                       // 000000005AE8: BF8CC07F
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]// 000000005AEC: D3E10028 0CA24190
	ds_write_b64 v176, v[192:193] offset:512                   // 000000005AF4: D89A0200 0000C0B0
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]// 000000005AFC: D3E1002C 0CB24192
	ds_read_b128 a[176:179], v7 offset:45312                   // 000000005B04: DBFEB100 B0000007
	ds_read_b128 a[180:183], v7 offset:46336                   // 000000005B0C: DBFEB500 B4000007
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]// 000000005B14: D3E10030 0CC24194
	ds_write_b64 v176, v[198:199] offset:768                   // 000000005B1C: D89A0300 0000C6B0
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]// 000000005B24: D3E10034 0CD24196
	ds_write_b64 v176, v[204:205] offset:1024                  // 000000005B2C: D89A0400 0000CCB0
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]// 000000005B34: D3E10038 0CE24198
	ds_write_b64 v176, v[210:211] offset:1280                  // 000000005B3C: D89A0500 0000D2B0
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]// 000000005B44: D3E1003C 0CF2419A
	ds_read_b128 a[184:187], v7 offset:47360                   // 000000005B4C: DBFEB900 B8000007
	ds_read_b128 a[188:191], v7 offset:48384                   // 000000005B54: DBFEBD00 BC000007
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]// 000000005B5C: D3E10040 0D02419C
	ds_write_b64 v176, v[216:217] offset:1536                  // 000000005B64: D89A0600 0000D8B0
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]// 000000005B6C: D3E10044 0D12419E
	ds_write_b64 v176, v[222:223] offset:1792                  // 000000005B74: D89A0700 0000DEB0
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]// 000000005B7C: D3E10048 0D2241A0
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]// 000000005B84: D3E1004C 0D3241A2
	ds_read_b128 a[192:195], v7 offset:49408                   // 000000005B8C: DBFEC100 C0000007
	ds_read_b128 a[196:199], v7 offset:50432                   // 000000005B94: DBFEC500 C4000007
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]// 000000005B9C: D3E10050 0D4241A4
	ds_write_b64 v176, v[228:229] offset:2048                  // 000000005BA4: D89A0800 0000E4B0
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]// 000000005BAC: D3E10054 0D5241A6
	s_waitcnt lgkmcnt(4)                                       // 000000005BB4: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]// 000000005BB8: D3E10058 0D6241A8
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]// 000000005BC0: D3E1005C 0D7241AA
	ds_read_b128 a[200:203], v7 offset:51456                   // 000000005BC8: DBFEC900 C8000007
	ds_read_b128 a[204:207], v7 offset:52480                   // 000000005BD0: DBFECD00 CC000007
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]// 000000005BD8: D3E10060 0D8241AC
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]// 000000005BE0: D3E10064 0D9241AE
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]// 000000005BE8: D3E10068 0DA241B0
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]// 000000005BF0: D3E1006C 0DB241B2
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]// 000000005BF8: D3E10070 0DC241B4
	s_waitcnt vmcnt(9) lgkmcnt(9)                              // 000000005C00: BF8C0979
	s_barrier                                                  // 000000005C04: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]// 000000005C08: D3E10074 0DD241B6
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]// 000000005C10: D3E10078 0DE241B8
	ds_read_b64 v[20:21], v5                                   // 000000005C18: D8EC0000 14000005
	ds_read_b64 v[22:23], v5 offset:4640                       // 000000005C20: D8EC1220 16000005
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]// 000000005C28: D3E1007C 0DF241BA
	ds_read_b64 v[24:25], v5 offset:9280                       // 000000005C30: D8EC2440 18000005
	ds_read_b64 v[26:27], v5 offset:13920                      // 000000005C38: D8EC3660 1A000005
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]// 000000005C40: D3E10080 0E0241BC
	ds_read_b128 a[144:147], v4                                // 000000005C48: DBFE0000 90000004
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]// 000000005C50: D3E10084 0E1241BE
	ds_read_b128 a[148:151], v4 offset:64                      // 000000005C58: DBFE0040 94000004
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]// 000000005C60: D3E10088 0E2241C0
	ds_read_b128 a[152:155], v4 offset:256                     // 000000005C68: DBFE0100 98000004
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]// 000000005C70: D3E1008C 0E3241C2
	ds_read_b128 a[156:159], v4 offset:320                     // 000000005C78: DBFE0140 9C000004
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]// 000000005C80: D3E10090 0E4241C4
	ds_read_b128 a[160:163], v4 offset:512                     // 000000005C88: DBFE0200 A0000004
	v_mfma_f32_16x16x16_bf16 v[148:151], a[198:199], v[32:33], v[148:151]// 000000005C90: D3E10094 0E5241C6
	ds_read_b128 a[164:167], v4 offset:576                     // 000000005C98: DBFE0240 A4000004
	v_mfma_f32_16x16x16_bf16 v[152:155], a[200:201], v[32:33], v[152:155]// 000000005CA0: D3E10098 0E6241C8
	ds_read_b128 a[168:171], v4 offset:768                     // 000000005CA8: DBFE0300 A8000004
	v_mfma_f32_16x16x16_bf16 v[156:159], a[202:203], v[32:33], v[156:159]// 000000005CB0: D3E1009C 0E7241CA
	ds_read_b128 a[172:175], v4 offset:832                     // 000000005CB8: DBFE0340 AC000004
	v_mfma_f32_16x16x16_bf16 v[160:163], a[204:205], v[32:33], v[160:163]// 000000005CC0: D3E100A0 0E8241CC
	s_waitcnt lgkmcnt(8)                                       // 000000005CC8: BF8CC87F
	v_perm_b32 v168, v22, v20, s53                             // 000000005CCC: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 000000005CD4: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000005CDC: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 000000005CE4: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[164:167], a[206:207], v[32:33], v[164:167]// 000000005CEC: D3E100A4 0E9241CE
	ds_write_b128 v6, v[168:171] offset:37120                  // 000000005CF4: D9BE9100 0000A806
	v_perm_b32 v168, v23, v21, s53                             // 000000005CFC: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 000000005D04: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 000000005D0C: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 000000005D14: D1ED00AB 00D2331B
	ds_write_b128 v6, v[168:171] offset:38144                  // 000000005D1C: D9BE9500 0000A806
	ds_read_b64 v[20:21], v5 offset:1024                       // 000000005D24: D8EC0400 14000005
	ds_read_b64 v[22:23], v5 offset:5664                       // 000000005D2C: D8EC1620 16000005
	ds_read_b64 v[24:25], v5 offset:10304                      // 000000005D34: D8EC2840 18000005
	ds_read_b64 v[26:27], v5 offset:14944                      // 000000005D3C: D8EC3A60 1A000005
	s_nop 0                                                    // 000000005D44: BF800000
	s_addk_i32 s70, 0x1                                        // 000000005D48: B7460001
	s_cmp_lt_i32 s70, s71                                      // 000000005D4C: BF044746
	s_cbranch_scc0 label_097C                                  // 000000005D50: BF84FAE7
	s_branch label_097F                                        // 000000005D54: BF82FAE9

0000000000005d58 <label_0E96>:
	s_cmp_eq_i32 s48, 0                                        // 000000005D58: BF008030
	s_cbranch_scc1 label_12AF                                  // 000000005D5C: BF850417

0000000000005d60 <label_0E98>:
	s_and_b32 s56, s71, 1                                      // 000000005D60: 86388147
	s_cmp_eq_i32 s56, 1                                        // 000000005D64: BF008138
	s_cbranch_scc1 label_10A5                                  // 000000005D68: BF85020A
	s_waitcnt lgkmcnt(4)                                       // 000000005D6C: BF8CC47F
	s_waitcnt vmcnt(0)                                         // 000000005D70: BF8C0F70
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0   // 000000005D74: D3E10020 1A020190
	ds_read_b128 a[176:179], v4 offset:1024                    // 000000005D7C: DBFE0400 B0000004
	ds_read_b128 a[180:183], v4 offset:1088                    // 000000005D84: DBFE0440 B4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]// 000000005D8C: D3E10020 1C820592
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]// 000000005D94: D3E10020 1C820994
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]// 000000005D9C: D3E10020 1C820D96
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]// 000000005DA4: D3E10020 1C821198
	ds_read_b128 a[184:187], v4 offset:1280                    // 000000005DAC: DBFE0500 B8000004
	ds_read_b128 a[188:191], v4 offset:1344                    // 000000005DB4: DBFE0540 BC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]// 000000005DBC: D3E10020 1C82159A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]// 000000005DC4: D3E10020 1C82199C
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]// 000000005DCC: D3E10020 1C821D9E
	s_waitcnt lgkmcnt(4)                                       // 000000005DD4: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]// 000000005DD8: D3E10020 1C8221A0
	ds_read_b128 a[192:195], v4 offset:1536                    // 000000005DE0: DBFE0600 C0000004
	ds_read_b128 a[196:199], v4 offset:1600                    // 000000005DE8: DBFE0640 C4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]// 000000005DF0: D3E10020 1C8225A2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]// 000000005DF8: D3E10020 1C8229A4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]// 000000005E00: D3E10020 1C822DA6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]// 000000005E08: D3E10020 1C8231A8
	ds_read_b128 a[200:203], v4 offset:1792                    // 000000005E10: DBFE0700 C8000004
	ds_read_b128 a[204:207], v4 offset:1856                    // 000000005E18: DBFE0740 CC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]// 000000005E20: D3E10020 1C8235AA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]// 000000005E28: D3E10020 1C8239AC
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]// 000000005E30: D3E10020 1C823DAE
	s_waitcnt lgkmcnt(4)                                       // 000000005E38: BF8CC47F
	s_barrier                                                  // 000000005E3C: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]// 000000005E40: D3E10020 1C8241B0
	ds_read_b128 a[208:211], v4 offset:2048                    // 000000005E48: DBFE0800 D0000004
	ds_read_b128 a[212:215], v4 offset:2112                    // 000000005E50: DBFE0840 D4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]// 000000005E58: D3E10020 1C8245B2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]// 000000005E60: D3E10020 1C8249B4
	v_perm_b32 v168, v22, v20, s53                             // 000000005E68: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 000000005E70: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 000000005E78: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 000000005E80: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]// 000000005E88: D3E10020 1C824DB6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]// 000000005E90: D3E10020 1C8251B8
	ds_write_b128 v6, v[168:171] offset:45312                  // 000000005E98: D9BEB100 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]// 000000005EA0: D3E10020 1C8255BA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]// 000000005EA8: D3E10020 1C8259BC
	v_perm_b32 v168, v23, v21, s53                             // 000000005EB0: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 000000005EB8: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 000000005EC0: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 000000005EC8: D1ED00AB 00D2331B
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]// 000000005ED0: D3E10020 1C825DBE
	s_waitcnt lgkmcnt(1)                                       // 000000005ED8: BF8CC17F
	s_barrier                                                  // 000000005EDC: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]// 000000005EE0: D3E10020 1C8261C0
	ds_write_b128 v6, v[168:171] offset:46336                  // 000000005EE8: D9BEB500 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]// 000000005EF0: D3E10020 1C8265C2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]// 000000005EF8: D3E10020 1C8269C4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]// 000000005F00: D3E10020 1C826DC6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]// 000000005F08: D3E10020 1C8271C8
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]// 000000005F10: D3E10020 1C8275CA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]// 000000005F18: D3E10020 1C8279CC
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]// 000000005F20: D3E10020 1C827DCE
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]// 000000005F28: D3E10020 1C8281D0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]// 000000005F30: D3E10020 1C8285D2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]// 000000005F38: D3E10020 1C8289D4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]// 000000005F40: D3E10020 1C828DD6
	s_cmp_le_i32 s83, s82                                      // 000000005F48: BF055253
	s_cbranch_scc1 label_0F38                                  // 000000005F4C: BF850024
	v_mov_b32_e32 v25, 0xff800000                              // 000000005F50: 7E3202FF FF800000
	s_add_u32 s57, s82, 0                                      // 000000005F58: 80398052
	v_mov_b32_e32 v24, s57                                     // 000000005F5C: 7E300239
	v_add_u32_e32 v24, s7, v24                                 // 000000005F60: 68303007
	s_sub_u32 s56, s83, 15                                     // 000000005F64: 80B88F53
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000005F68: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 000000005F6C: 0C282884
	v_add_u32_e32 v20, s56, v20                                // 000000005F70: 68282838
	v_add_u32_e32 v21, 1, v20                                  // 000000005F74: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 000000005F78: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 000000005F7C: 682E2883
	v_cmp_le_u32_e64 s[38:39], v20, v24                        // 000000005F80: D0CB0026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 000000005F88: 682828C0
	s_nop 0                                                    // 000000005F8C: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 000000005F90: D1000020 009A4119
	v_cmp_le_u32_e64 s[38:39], v21, v24                        // 000000005F98: D0CB0026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 000000005FA0: 682A2AC0
	s_nop 0                                                    // 000000005FA4: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 000000005FA8: D1000021 009A4319
	v_cmp_le_u32_e64 s[38:39], v22, v24                        // 000000005FB0: D0CB0026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 000000005FB8: 682C2CC0
	s_nop 0                                                    // 000000005FBC: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 000000005FC0: D1000022 009A4519
	v_cmp_le_u32_e64 s[38:39], v23, v24                        // 000000005FC8: D0CB0026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 000000005FD0: 682E2EC0
	s_nop 0                                                    // 000000005FD4: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000005FD8: D1000023 009A4719

0000000000005fe0 <label_0F38>:
	s_waitcnt lgkmcnt(0)                                       // 000000005FE0: BF8CC07F
	s_barrier                                                  // 000000005FE4: BF8A0000
	v_mov_b32_e32 v25, 0xff800000                              // 000000005FE8: 7E3202FF FF800000
	s_and_b32 s56, s48, 0xff                                   // 000000005FF0: 8638FF30 000000FF
	v_mov_b32_e32 v24, s56                                     // 000000005FF8: 7E300238
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000005FFC: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 000000006000: 0C282884
	v_add_u32_e32 v21, 1, v20                                  // 000000006004: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 000000006008: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 00000000600C: 682E2883
	v_cmp_lt_u32_e64 s[38:39], v20, v24                        // 000000006010: D0C90026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 000000006018: 682828C0
	s_nop 0                                                    // 00000000601C: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 000000006020: D1000020 009A4119
	v_cmp_lt_u32_e64 s[38:39], v21, v24                        // 000000006028: D0C90026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 000000006030: 682A2AC0
	s_nop 0                                                    // 000000006034: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 000000006038: D1000021 009A4319
	v_cmp_lt_u32_e64 s[38:39], v22, v24                        // 000000006040: D0C90026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 000000006048: 682C2CC0
	s_nop 0                                                    // 00000000604C: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 000000006050: D1000022 009A4519
	v_cmp_lt_u32_e64 s[38:39], v23, v24                        // 000000006058: D0C90026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 000000006060: 682E2EC0
	s_nop 0                                                    // 000000006064: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000006068: D1000023 009A4719
	v_max3_f32 v24, v32, v33, v32                              // 000000006070: D1D30018 04824320
	v_max3_f32 v24, v34, v35, v24                              // 000000006078: D1D30018 04624722
	ds_write_b32 v3, v24 offset:53504                          // 000000006080: D81AD100 00001803
	s_waitcnt lgkmcnt(0)                                       // 000000006088: BF8CC07F
	ds_read_b32 v20, v2 offset:53504                           // 00000000608C: D86CD100 14000002
	ds_read_b32 v21, v2 offset:53568                           // 000000006094: D86CD140 15000002
	ds_read_b32 v22, v2 offset:53632                           // 00000000609C: D86CD180 16000002
	ds_read_b32 v23, v2 offset:53696                           // 0000000060A4: D86CD1C0 17000002
	s_waitcnt lgkmcnt(0)                                       // 0000000060AC: BF8CC07F
	v_max3_f32 v24, v20, v21, v24                              // 0000000060B0: D1D30018 04622B14
	v_max3_f32 v24, v22, v23, v24                              // 0000000060B8: D1D30018 04622F16
	ds_read_b128 a[144:147], v7 offset:37120                   // 0000000060C0: DBFE9100 90000007
	ds_read_b128 a[148:151], v7 offset:38144                   // 0000000060C8: DBFE9500 94000007
	ds_read_b128 a[152:155], v7 offset:39168                   // 0000000060D0: DBFE9900 98000007
	ds_read_b128 a[156:159], v7 offset:40192                   // 0000000060D8: DBFE9D00 9C000007
	ds_read_b128 a[160:163], v7 offset:41216                   // 0000000060E0: DBFEA100 A0000007
	ds_read_b128 a[164:167], v7 offset:42240                   // 0000000060E8: DBFEA500 A4000007
	ds_read_b128 a[168:171], v7 offset:43264                   // 0000000060F0: DBFEA900 A8000007
	ds_read_b128 a[172:175], v7 offset:44288                   // 0000000060F8: DBFEAD00 AC000007
	v_mov_b32_e32 v25, 0xff7fffff                              // 000000006100: 7E3202FF FF7FFFFF
	v_cmp_eq_u32_e64 s[38:39], v25, v12                        // 000000006108: D0CA0026 00021919
	v_max_f32_e32 v20, v24, v12                                // 000000006110: 16281918
	v_sub_f32_e32 v16, v12, v20                                // 000000006114: 0420290C
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]                    // 000000006118: D1000010 00990110
	v_mov_b32_e32 v12, v20                                     // 000000006120: 7E180314
	v_mul_f32_e32 v21, s5, v20                                 // 000000006124: 0A2A2805
	v_mul_f32_e32 v16, s5, v16                                 // 000000006128: 0A202005
	v_exp_f32_e32 v16, v16                                     // 00000000612C: 7E204110
	v_fma_f32 v32, v32, s5, -v21                               // 000000006130: D1CB0020 84540B20
	v_fma_f32 v33, v33, s5, -v21                               // 000000006138: D1CB0021 84540B21
	v_fma_f32 v34, v34, s5, -v21                               // 000000006140: D1CB0022 84540B22
	v_fma_f32 v35, v35, s5, -v21                               // 000000006148: D1CB0023 84540B23
	v_exp_f32_e32 v32, v32                                     // 000000006150: 7E404120
	v_exp_f32_e32 v33, v33                                     // 000000006154: 7E424121
	v_exp_f32_e32 v34, v34                                     // 000000006158: 7E444122
	v_exp_f32_e32 v35, v35                                     // 00000000615C: 7E464123
	v_mul_f32_e32 v14, v16, v14                                // 000000006160: 0A1C1D10
	v_mov_b32_e32 v22, v32                                     // 000000006164: 7E2C0320
	v_add_f32_e32 v22, v33, v22                                // 000000006168: 022C2D21
	v_add_f32_e32 v22, v34, v22                                // 00000000616C: 022C2D22
	v_add_f32_e32 v22, v35, v22                                // 000000006170: 022C2D23
	v_add_f32_e32 v14, v22, v14                                // 000000006174: 021C1D16
	v_mov_b32_e32 v29, 0xffff0000                              // 000000006178: 7E3A02FF FFFF0000
	v_mov_b32_e32 v30, 0x7fff0000                              // 000000006180: 7E3C02FF 7FFF0000
	v_mov_b32_e32 v31, 0x7fff                                  // 000000006188: 7E3E02FF 00007FFF
	v_cmp_u_f32_e64 s[38:39], v32, v32                         // 000000006190: D0480026 00024120
	v_add3_u32 v28, v32, v31, 1                                // 000000006198: D1FF001C 02063F20
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000061A0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v33, v33                         // 0000000061A8: D0480026 00024321
	v_add3_u32 v28, v33, v31, 1                                // 0000000061B0: D1FF001C 02063F21
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000061B8: D1000015 009A3D1C
	v_perm_b32 v32, v21, v20, s52                              // 0000000061C0: D1ED0020 00D22915
	v_cmp_u_f32_e64 s[38:39], v34, v34                         // 0000000061C8: D0480026 00024522
	v_add3_u32 v28, v34, v31, 1                                // 0000000061D0: D1FF001C 02063F22
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000061D8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v35, v35                         // 0000000061E0: D0480026 00024723
	v_add3_u32 v28, v35, v31, 1                                // 0000000061E8: D1FF001C 02063F23
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000061F0: D1000015 009A3D1C
	v_perm_b32 v33, v21, v20, s52                              // 0000000061F8: D1ED0021 00D22915
	s_nop 2                                                    // 000000006200: BF800002
	s_add_u32 s83, s84, s83                                    // 000000006204: 80535354
	s_nop 0                                                    // 000000006208: BF800000
	v_mov_b32_e32 v22, v16                                     // 00000000620C: 7E2C0310
	v_mov_b32_e32 v23, v16                                     // 000000006210: 7E2E0310
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]                  // 000000006214: D3B14028 18025116
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]                  // 00000000621C: D3B1402A 18025516
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]                  // 000000006224: D3B1402C 18025916
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]                  // 00000000622C: D3B1402E 18025D16
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]                  // 000000006234: D3B14030 18026116
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]                  // 00000000623C: D3B14032 18026516
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]                  // 000000006244: D3B14034 18026916
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]                  // 00000000624C: D3B14036 18026D16
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]                  // 000000006254: D3B14038 18027116
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]                  // 00000000625C: D3B1403A 18027516
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]                  // 000000006264: D3B1403C 18027916
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]                  // 00000000626C: D3B1403E 18027D16
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]                  // 000000006274: D3B14040 18028116
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]                  // 00000000627C: D3B14042 18028516
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]                  // 000000006284: D3B14044 18028916
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]                  // 00000000628C: D3B14046 18028D16
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]                  // 000000006294: D3B14048 18029116
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]                  // 00000000629C: D3B1404A 18029516
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]                  // 0000000062A4: D3B1404C 18029916
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]                  // 0000000062AC: D3B1404E 18029D16
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]                  // 0000000062B4: D3B14050 1802A116
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]                  // 0000000062BC: D3B14052 1802A516
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]                  // 0000000062C4: D3B14054 1802A916
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]                  // 0000000062CC: D3B14056 1802AD16
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]                  // 0000000062D4: D3B14058 1802B116
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]                  // 0000000062DC: D3B1405A 1802B516
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]                  // 0000000062E4: D3B1405C 1802B916
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]                  // 0000000062EC: D3B1405E 1802BD16
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]                  // 0000000062F4: D3B14060 1802C116
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]                  // 0000000062FC: D3B14062 1802C516
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]              // 000000006304: D3B14064 1802C916
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]              // 00000000630C: D3B14066 1802CD16
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]              // 000000006314: D3B14068 1802D116
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]              // 00000000631C: D3B1406A 1802D516
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]              // 000000006324: D3B1406C 1802D916
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]              // 00000000632C: D3B1406E 1802DD16
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]              // 000000006334: D3B14070 1802E116
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]              // 00000000633C: D3B14072 1802E516
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]              // 000000006344: D3B14074 1802E916
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]              // 00000000634C: D3B14076 1802ED16
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]              // 000000006354: D3B14078 1802F116
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]              // 00000000635C: D3B1407A 1802F516
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]              // 000000006364: D3B1407C 1802F916
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]              // 00000000636C: D3B1407E 1802FD16
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]              // 000000006374: D3B14080 18030116
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]              // 00000000637C: D3B14082 18030516
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]              // 000000006384: D3B14084 18030916
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]              // 00000000638C: D3B14086 18030D16
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]              // 000000006394: D3B14088 18031116
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]              // 00000000639C: D3B1408A 18031516
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]              // 0000000063A4: D3B1408C 18031916
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]              // 0000000063AC: D3B1408E 18031D16
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]              // 0000000063B4: D3B14090 18032116
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]              // 0000000063BC: D3B14092 18032516
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]              // 0000000063C4: D3B14094 18032916
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]              // 0000000063CC: D3B14096 18032D16
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]              // 0000000063D4: D3B14098 18033116
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]              // 0000000063DC: D3B1409A 18033516
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]              // 0000000063E4: D3B1409C 18033916
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]              // 0000000063EC: D3B1409E 18033D16
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]              // 0000000063F4: D3B140A0 18034116
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]              // 0000000063FC: D3B140A2 18034516
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]              // 000000006404: D3B140A4 18034916
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]              // 00000000640C: D3B140A6 18034D16
	s_waitcnt lgkmcnt(0)                                       // 000000006414: BF8CC07F
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]// 000000006418: D3E10028 0CA24190
	ds_read_b128 a[176:179], v7 offset:45312                   // 000000006420: DBFEB100 B0000007
	ds_read_b128 a[180:183], v7 offset:46336                   // 000000006428: DBFEB500 B4000007
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]// 000000006430: D3E1002C 0CB24192
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]// 000000006438: D3E10030 0CC24194
	ds_write_b64 v176, v[192:193] offset:19072                 // 000000006440: D89A4A80 0000C0B0
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]// 000000006448: D3E10034 0CD24196
	ds_write_b64 v176, v[198:199] offset:19328                 // 000000006450: D89A4B80 0000C6B0
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]// 000000006458: D3E10038 0CE24198
	ds_read_b128 a[184:187], v7 offset:47360                   // 000000006460: DBFEB900 B8000007
	ds_read_b128 a[188:191], v7 offset:48384                   // 000000006468: DBFEBD00 BC000007
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]// 000000006470: D3E1003C 0CF2419A
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]// 000000006478: D3E10040 0D02419C
	ds_write_b64 v176, v[204:205] offset:19584                 // 000000006480: D89A4C80 0000CCB0
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]// 000000006488: D3E10044 0D12419E
	ds_write_b64 v176, v[210:211] offset:19840                 // 000000006490: D89A4D80 0000D2B0
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]// 000000006498: D3E10048 0D2241A0
	ds_read_b128 a[192:195], v7 offset:49408                   // 0000000064A0: DBFEC100 C0000007
	ds_read_b128 a[196:199], v7 offset:50432                   // 0000000064A8: DBFEC500 C4000007
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]// 0000000064B0: D3E1004C 0D3241A2
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]// 0000000064B8: D3E10050 0D4241A4
	ds_write_b64 v176, v[216:217] offset:20096                 // 0000000064C0: D89A4E80 0000D8B0
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]// 0000000064C8: D3E10054 0D5241A6
	s_waitcnt lgkmcnt(4)                                       // 0000000064D0: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]// 0000000064D4: D3E10058 0D6241A8
	ds_read_b128 a[200:203], v7 offset:51456                   // 0000000064DC: DBFEC900 C8000007
	ds_read_b128 a[204:207], v7 offset:52480                   // 0000000064E4: DBFECD00 CC000007
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]// 0000000064EC: D3E1005C 0D7241AA
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]// 0000000064F4: D3E10060 0D8241AC
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]// 0000000064FC: D3E10064 0D9241AE
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]// 000000006504: D3E10068 0DA241B0
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]// 00000000650C: D3E1006C 0DB241B2
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]// 000000006514: D3E10070 0DC241B4
	s_waitcnt vmcnt(9) lgkmcnt(9)                              // 00000000651C: BF8C0979
	s_barrier                                                  // 000000006520: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]// 000000006524: D3E10074 0DD241B6
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]// 00000000652C: D3E10078 0DE241B8
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]// 000000006534: D3E1007C 0DF241BA
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]// 00000000653C: D3E10080 0E0241BC
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]// 000000006544: D3E10084 0E1241BE
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]// 00000000654C: D3E10088 0E2241C0
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]// 000000006554: D3E1008C 0E3241C2
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]// 00000000655C: D3E10090 0E4241C4
	v_mfma_f32_16x16x16_bf16 v[148:151], a[198:199], v[32:33], v[148:151]// 000000006564: D3E10094 0E5241C6
	v_mfma_f32_16x16x16_bf16 v[152:155], a[200:201], v[32:33], v[152:155]// 00000000656C: D3E10098 0E6241C8
	v_mfma_f32_16x16x16_bf16 v[156:159], a[202:203], v[32:33], v[156:159]// 000000006574: D3E1009C 0E7241CA
	v_mfma_f32_16x16x16_bf16 v[160:163], a[204:205], v[32:33], v[160:163]// 00000000657C: D3E100A0 0E8241CC
	v_mfma_f32_16x16x16_bf16 v[164:167], a[206:207], v[32:33], v[164:167]// 000000006584: D3E100A4 0E9241CE
	s_nop 8                                                    // 00000000658C: BF800008
	s_branch label_12AF                                        // 000000006590: BF82020A

0000000000006594 <label_10A5>:
	s_waitcnt lgkmcnt(4)                                       // 000000006594: BF8CC47F
	s_waitcnt vmcnt(0)                                         // 000000006598: BF8C0F70
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0   // 00000000659C: D3E10020 1A020190
	ds_read_b128 a[176:179], v4 offset:19584                   // 0000000065A4: DBFE4C80 B0000004
	ds_read_b128 a[180:183], v4 offset:19648                   // 0000000065AC: DBFE4CC0 B4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]// 0000000065B4: D3E10020 1C820592
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]// 0000000065BC: D3E10020 1C820994
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]// 0000000065C4: D3E10020 1C820D96
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]// 0000000065CC: D3E10020 1C821198
	ds_read_b128 a[184:187], v4 offset:19840                   // 0000000065D4: DBFE4D80 B8000004
	ds_read_b128 a[188:191], v4 offset:19904                   // 0000000065DC: DBFE4DC0 BC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]// 0000000065E4: D3E10020 1C82159A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]// 0000000065EC: D3E10020 1C82199C
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]// 0000000065F4: D3E10020 1C821D9E
	s_waitcnt lgkmcnt(4)                                       // 0000000065FC: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]// 000000006600: D3E10020 1C8221A0
	ds_read_b128 a[192:195], v4 offset:20096                   // 000000006608: DBFE4E80 C0000004
	ds_read_b128 a[196:199], v4 offset:20160                   // 000000006610: DBFE4EC0 C4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]// 000000006618: D3E10020 1C8225A2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]// 000000006620: D3E10020 1C8229A4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]// 000000006628: D3E10020 1C822DA6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]// 000000006630: D3E10020 1C8231A8
	ds_read_b128 a[200:203], v4 offset:20352                   // 000000006638: DBFE4F80 C8000004
	ds_read_b128 a[204:207], v4 offset:20416                   // 000000006640: DBFE4FC0 CC000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]// 000000006648: D3E10020 1C8235AA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]// 000000006650: D3E10020 1C8239AC
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]// 000000006658: D3E10020 1C823DAE
	s_waitcnt lgkmcnt(4)                                       // 000000006660: BF8CC47F
	s_barrier                                                  // 000000006664: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]// 000000006668: D3E10020 1C8241B0
	ds_read_b128 a[208:211], v4 offset:20608                   // 000000006670: DBFE5080 D0000004
	ds_read_b128 a[212:215], v4 offset:20672                   // 000000006678: DBFE50C0 D4000004
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]// 000000006680: D3E10020 1C8245B2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]// 000000006688: D3E10020 1C8249B4
	v_perm_b32 v168, v22, v20, s53                             // 000000006690: D1ED00A8 00D62916
	v_perm_b32 v170, v22, v20, s52                             // 000000006698: D1ED00AA 00D22916
	v_perm_b32 v169, v26, v24, s53                             // 0000000066A0: D1ED00A9 00D6311A
	v_perm_b32 v171, v26, v24, s52                             // 0000000066A8: D1ED00AB 00D2311A
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]// 0000000066B0: D3E10020 1C824DB6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]// 0000000066B8: D3E10020 1C8251B8
	ds_write_b128 v6, v[168:171] offset:45312                  // 0000000066C0: D9BEB100 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]// 0000000066C8: D3E10020 1C8255BA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]// 0000000066D0: D3E10020 1C8259BC
	v_perm_b32 v168, v23, v21, s53                             // 0000000066D8: D1ED00A8 00D62B17
	v_perm_b32 v170, v23, v21, s52                             // 0000000066E0: D1ED00AA 00D22B17
	v_perm_b32 v169, v27, v25, s53                             // 0000000066E8: D1ED00A9 00D6331B
	v_perm_b32 v171, v27, v25, s52                             // 0000000066F0: D1ED00AB 00D2331B
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]// 0000000066F8: D3E10020 1C825DBE
	s_waitcnt lgkmcnt(1)                                       // 000000006700: BF8CC17F
	s_barrier                                                  // 000000006704: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]// 000000006708: D3E10020 1C8261C0
	ds_write_b128 v6, v[168:171] offset:46336                  // 000000006710: D9BEB500 0000A806
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]// 000000006718: D3E10020 1C8265C2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]// 000000006720: D3E10020 1C8269C4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]// 000000006728: D3E10020 1C826DC6
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]// 000000006730: D3E10020 1C8271C8
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]// 000000006738: D3E10020 1C8275CA
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]// 000000006740: D3E10020 1C8279CC
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]// 000000006748: D3E10020 1C827DCE
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]// 000000006750: D3E10020 1C8281D0
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]// 000000006758: D3E10020 1C8285D2
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]// 000000006760: D3E10020 1C8289D4
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]// 000000006768: D3E10020 1C828DD6
	s_cmp_le_i32 s83, s82                                      // 000000006770: BF055253
	s_cbranch_scc1 label_1142                                  // 000000006774: BF850024
	v_mov_b32_e32 v25, 0xff800000                              // 000000006778: 7E3202FF FF800000
	s_add_u32 s57, s82, 0                                      // 000000006780: 80398052
	v_mov_b32_e32 v24, s57                                     // 000000006784: 7E300239
	v_add_u32_e32 v24, s7, v24                                 // 000000006788: 68303007
	s_sub_u32 s56, s83, 15                                     // 00000000678C: 80B88F53
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000006790: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 000000006794: 0C282884
	v_add_u32_e32 v20, s56, v20                                // 000000006798: 68282838
	v_add_u32_e32 v21, 1, v20                                  // 00000000679C: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 0000000067A0: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 0000000067A4: 682E2883
	v_cmp_le_u32_e64 s[38:39], v20, v24                        // 0000000067A8: D0CB0026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 0000000067B0: 682828C0
	s_nop 0                                                    // 0000000067B4: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 0000000067B8: D1000020 009A4119
	v_cmp_le_u32_e64 s[38:39], v21, v24                        // 0000000067C0: D0CB0026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 0000000067C8: 682A2AC0
	s_nop 0                                                    // 0000000067CC: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 0000000067D0: D1000021 009A4319
	v_cmp_le_u32_e64 s[38:39], v22, v24                        // 0000000067D8: D0CB0026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 0000000067E0: 682C2CC0
	s_nop 0                                                    // 0000000067E4: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 0000000067E8: D1000022 009A4519
	v_cmp_le_u32_e64 s[38:39], v23, v24                        // 0000000067F0: D0CB0026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 0000000067F8: 682E2EC0
	s_nop 0                                                    // 0000000067FC: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000006800: D1000023 009A4719

0000000000006808 <label_1142>:
	s_waitcnt lgkmcnt(0)                                       // 000000006808: BF8CC07F
	s_barrier                                                  // 00000000680C: BF8A0000
	v_mov_b32_e32 v25, 0xff800000                              // 000000006810: 7E3202FF FF800000
	s_and_b32 s56, s48, 0xff                                   // 000000006818: 8638FF30 000000FF
	v_mov_b32_e32 v24, s56                                     // 000000006820: 7E300238
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000006824: 20280084
	v_mul_i32_i24_e32 v20, 4, v20                              // 000000006828: 0C282884
	v_add_u32_e32 v21, 1, v20                                  // 00000000682C: 682A2881
	v_add_u32_e32 v22, 2, v20                                  // 000000006830: 682C2882
	v_add_u32_e32 v23, 3, v20                                  // 000000006834: 682E2883
	v_cmp_lt_u32_e64 s[38:39], v20, v24                        // 000000006838: D0C90026 00023114
	v_add_u32_e32 v20, 64, v20                                 // 000000006840: 682828C0
	s_nop 0                                                    // 000000006844: BF800000
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]                  // 000000006848: D1000020 009A4119
	v_cmp_lt_u32_e64 s[38:39], v21, v24                        // 000000006850: D0C90026 00023115
	v_add_u32_e32 v21, 64, v21                                 // 000000006858: 682A2AC0
	s_nop 0                                                    // 00000000685C: BF800000
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]                  // 000000006860: D1000021 009A4319
	v_cmp_lt_u32_e64 s[38:39], v22, v24                        // 000000006868: D0C90026 00023116
	v_add_u32_e32 v22, 64, v22                                 // 000000006870: 682C2CC0
	s_nop 0                                                    // 000000006874: BF800000
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]                  // 000000006878: D1000022 009A4519
	v_cmp_lt_u32_e64 s[38:39], v23, v24                        // 000000006880: D0C90026 00023117
	v_add_u32_e32 v23, 64, v23                                 // 000000006888: 682E2EC0
	s_nop 0                                                    // 00000000688C: BF800000
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]                  // 000000006890: D1000023 009A4719
	v_max3_f32 v24, v32, v33, v32                              // 000000006898: D1D30018 04824320
	v_max3_f32 v24, v34, v35, v24                              // 0000000068A0: D1D30018 04624722
	ds_write_b32 v3, v24 offset:53504                          // 0000000068A8: D81AD100 00001803
	s_waitcnt lgkmcnt(0)                                       // 0000000068B0: BF8CC07F
	ds_read_b32 v20, v2 offset:53504                           // 0000000068B4: D86CD100 14000002
	ds_read_b32 v21, v2 offset:53568                           // 0000000068BC: D86CD140 15000002
	ds_read_b32 v22, v2 offset:53632                           // 0000000068C4: D86CD180 16000002
	ds_read_b32 v23, v2 offset:53696                           // 0000000068CC: D86CD1C0 17000002
	s_waitcnt lgkmcnt(0)                                       // 0000000068D4: BF8CC07F
	v_max3_f32 v24, v20, v21, v24                              // 0000000068D8: D1D30018 04622B14
	v_max3_f32 v24, v22, v23, v24                              // 0000000068E0: D1D30018 04622F16
	ds_read_b128 a[144:147], v7 offset:37120                   // 0000000068E8: DBFE9100 90000007
	ds_read_b128 a[148:151], v7 offset:38144                   // 0000000068F0: DBFE9500 94000007
	ds_read_b128 a[152:155], v7 offset:39168                   // 0000000068F8: DBFE9900 98000007
	ds_read_b128 a[156:159], v7 offset:40192                   // 000000006900: DBFE9D00 9C000007
	ds_read_b128 a[160:163], v7 offset:41216                   // 000000006908: DBFEA100 A0000007
	ds_read_b128 a[164:167], v7 offset:42240                   // 000000006910: DBFEA500 A4000007
	ds_read_b128 a[168:171], v7 offset:43264                   // 000000006918: DBFEA900 A8000007
	ds_read_b128 a[172:175], v7 offset:44288                   // 000000006920: DBFEAD00 AC000007
	v_mov_b32_e32 v25, 0xff7fffff                              // 000000006928: 7E3202FF FF7FFFFF
	v_cmp_eq_u32_e64 s[38:39], v25, v12                        // 000000006930: D0CA0026 00021919
	v_max_f32_e32 v20, v24, v12                                // 000000006938: 16281918
	v_sub_f32_e32 v16, v12, v20                                // 00000000693C: 0420290C
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]                    // 000000006940: D1000010 00990110
	v_mov_b32_e32 v12, v20                                     // 000000006948: 7E180314
	v_mul_f32_e32 v21, s5, v20                                 // 00000000694C: 0A2A2805
	v_mul_f32_e32 v16, s5, v16                                 // 000000006950: 0A202005
	v_exp_f32_e32 v16, v16                                     // 000000006954: 7E204110
	v_fma_f32 v32, v32, s5, -v21                               // 000000006958: D1CB0020 84540B20
	v_fma_f32 v33, v33, s5, -v21                               // 000000006960: D1CB0021 84540B21
	v_fma_f32 v34, v34, s5, -v21                               // 000000006968: D1CB0022 84540B22
	v_fma_f32 v35, v35, s5, -v21                               // 000000006970: D1CB0023 84540B23
	v_exp_f32_e32 v32, v32                                     // 000000006978: 7E404120
	v_exp_f32_e32 v33, v33                                     // 00000000697C: 7E424121
	v_exp_f32_e32 v34, v34                                     // 000000006980: 7E444122
	v_exp_f32_e32 v35, v35                                     // 000000006984: 7E464123
	v_mul_f32_e32 v14, v16, v14                                // 000000006988: 0A1C1D10
	v_mov_b32_e32 v22, v32                                     // 00000000698C: 7E2C0320
	v_add_f32_e32 v22, v33, v22                                // 000000006990: 022C2D21
	v_add_f32_e32 v22, v34, v22                                // 000000006994: 022C2D22
	v_add_f32_e32 v22, v35, v22                                // 000000006998: 022C2D23
	v_add_f32_e32 v14, v22, v14                                // 00000000699C: 021C1D16
	v_mov_b32_e32 v29, 0xffff0000                              // 0000000069A0: 7E3A02FF FFFF0000
	v_mov_b32_e32 v30, 0x7fff0000                              // 0000000069A8: 7E3C02FF 7FFF0000
	v_mov_b32_e32 v31, 0x7fff                                  // 0000000069B0: 7E3E02FF 00007FFF
	v_cmp_u_f32_e64 s[38:39], v32, v32                         // 0000000069B8: D0480026 00024120
	v_add3_u32 v28, v32, v31, 1                                // 0000000069C0: D1FF001C 02063F20
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000069C8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v33, v33                         // 0000000069D0: D0480026 00024321
	v_add3_u32 v28, v33, v31, 1                                // 0000000069D8: D1FF001C 02063F21
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000069E0: D1000015 009A3D1C
	v_perm_b32 v32, v21, v20, s52                              // 0000000069E8: D1ED0020 00D22915
	v_cmp_u_f32_e64 s[38:39], v34, v34                         // 0000000069F0: D0480026 00024522
	v_add3_u32 v28, v34, v31, 1                                // 0000000069F8: D1FF001C 02063F22
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000006A00: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v35, v35                         // 000000006A08: D0480026 00024723
	v_add3_u32 v28, v35, v31, 1                                // 000000006A10: D1FF001C 02063F23
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000006A18: D1000015 009A3D1C
	v_perm_b32 v33, v21, v20, s52                              // 000000006A20: D1ED0021 00D22915
	s_nop 2                                                    // 000000006A28: BF800002
	s_add_u32 s83, s84, s83                                    // 000000006A2C: 80535354
	s_nop 0                                                    // 000000006A30: BF800000
	v_mov_b32_e32 v22, v16                                     // 000000006A34: 7E2C0310
	v_mov_b32_e32 v23, v16                                     // 000000006A38: 7E2E0310
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]                  // 000000006A3C: D3B14028 18025116
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]                  // 000000006A44: D3B1402A 18025516
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]                  // 000000006A4C: D3B1402C 18025916
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]                  // 000000006A54: D3B1402E 18025D16
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]                  // 000000006A5C: D3B14030 18026116
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]                  // 000000006A64: D3B14032 18026516
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]                  // 000000006A6C: D3B14034 18026916
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]                  // 000000006A74: D3B14036 18026D16
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]                  // 000000006A7C: D3B14038 18027116
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]                  // 000000006A84: D3B1403A 18027516
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]                  // 000000006A8C: D3B1403C 18027916
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]                  // 000000006A94: D3B1403E 18027D16
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]                  // 000000006A9C: D3B14040 18028116
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]                  // 000000006AA4: D3B14042 18028516
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]                  // 000000006AAC: D3B14044 18028916
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]                  // 000000006AB4: D3B14046 18028D16
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]                  // 000000006ABC: D3B14048 18029116
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]                  // 000000006AC4: D3B1404A 18029516
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]                  // 000000006ACC: D3B1404C 18029916
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]                  // 000000006AD4: D3B1404E 18029D16
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]                  // 000000006ADC: D3B14050 1802A116
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]                  // 000000006AE4: D3B14052 1802A516
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]                  // 000000006AEC: D3B14054 1802A916
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]                  // 000000006AF4: D3B14056 1802AD16
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]                  // 000000006AFC: D3B14058 1802B116
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]                  // 000000006B04: D3B1405A 1802B516
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]                  // 000000006B0C: D3B1405C 1802B916
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]                  // 000000006B14: D3B1405E 1802BD16
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]                  // 000000006B1C: D3B14060 1802C116
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]                  // 000000006B24: D3B14062 1802C516
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]              // 000000006B2C: D3B14064 1802C916
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]              // 000000006B34: D3B14066 1802CD16
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]              // 000000006B3C: D3B14068 1802D116
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]              // 000000006B44: D3B1406A 1802D516
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]              // 000000006B4C: D3B1406C 1802D916
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]              // 000000006B54: D3B1406E 1802DD16
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]              // 000000006B5C: D3B14070 1802E116
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]              // 000000006B64: D3B14072 1802E516
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]              // 000000006B6C: D3B14074 1802E916
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]              // 000000006B74: D3B14076 1802ED16
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]              // 000000006B7C: D3B14078 1802F116
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]              // 000000006B84: D3B1407A 1802F516
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]              // 000000006B8C: D3B1407C 1802F916
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]              // 000000006B94: D3B1407E 1802FD16
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]              // 000000006B9C: D3B14080 18030116
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]              // 000000006BA4: D3B14082 18030516
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]              // 000000006BAC: D3B14084 18030916
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]              // 000000006BB4: D3B14086 18030D16
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]              // 000000006BBC: D3B14088 18031116
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]              // 000000006BC4: D3B1408A 18031516
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]              // 000000006BCC: D3B1408C 18031916
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]              // 000000006BD4: D3B1408E 18031D16
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]              // 000000006BDC: D3B14090 18032116
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]              // 000000006BE4: D3B14092 18032516
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]              // 000000006BEC: D3B14094 18032916
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]              // 000000006BF4: D3B14096 18032D16
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]              // 000000006BFC: D3B14098 18033116
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]              // 000000006C04: D3B1409A 18033516
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]              // 000000006C0C: D3B1409C 18033916
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]              // 000000006C14: D3B1409E 18033D16
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]              // 000000006C1C: D3B140A0 18034116
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]              // 000000006C24: D3B140A2 18034516
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]              // 000000006C2C: D3B140A4 18034916
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]              // 000000006C34: D3B140A6 18034D16
	s_waitcnt lgkmcnt(0)                                       // 000000006C3C: BF8CC07F
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]// 000000006C40: D3E10028 0CA24190
	ds_read_b128 a[176:179], v7 offset:45312                   // 000000006C48: DBFEB100 B0000007
	ds_read_b128 a[180:183], v7 offset:46336                   // 000000006C50: DBFEB500 B4000007
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]// 000000006C58: D3E1002C 0CB24192
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]// 000000006C60: D3E10030 0CC24194
	ds_write_b64 v176, v[192:193] offset:512                   // 000000006C68: D89A0200 0000C0B0
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]// 000000006C70: D3E10034 0CD24196
	ds_write_b64 v176, v[198:199] offset:768                   // 000000006C78: D89A0300 0000C6B0
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]// 000000006C80: D3E10038 0CE24198
	ds_read_b128 a[184:187], v7 offset:47360                   // 000000006C88: DBFEB900 B8000007
	ds_read_b128 a[188:191], v7 offset:48384                   // 000000006C90: DBFEBD00 BC000007
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]// 000000006C98: D3E1003C 0CF2419A
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]// 000000006CA0: D3E10040 0D02419C
	ds_write_b64 v176, v[204:205] offset:1024                  // 000000006CA8: D89A0400 0000CCB0
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]// 000000006CB0: D3E10044 0D12419E
	ds_write_b64 v176, v[210:211] offset:1280                  // 000000006CB8: D89A0500 0000D2B0
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]// 000000006CC0: D3E10048 0D2241A0
	ds_read_b128 a[192:195], v7 offset:49408                   // 000000006CC8: DBFEC100 C0000007
	ds_read_b128 a[196:199], v7 offset:50432                   // 000000006CD0: DBFEC500 C4000007
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]// 000000006CD8: D3E1004C 0D3241A2
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]// 000000006CE0: D3E10050 0D4241A4
	ds_write_b64 v176, v[216:217] offset:1536                  // 000000006CE8: D89A0600 0000D8B0
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]// 000000006CF0: D3E10054 0D5241A6
	s_waitcnt lgkmcnt(4)                                       // 000000006CF8: BF8CC47F
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]// 000000006CFC: D3E10058 0D6241A8
	ds_read_b128 a[200:203], v7 offset:51456                   // 000000006D04: DBFEC900 C8000007
	ds_read_b128 a[204:207], v7 offset:52480                   // 000000006D0C: DBFECD00 CC000007
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]// 000000006D14: D3E1005C 0D7241AA
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]// 000000006D1C: D3E10060 0D8241AC
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]// 000000006D24: D3E10064 0D9241AE
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]// 000000006D2C: D3E10068 0DA241B0
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]// 000000006D34: D3E1006C 0DB241B2
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]// 000000006D3C: D3E10070 0DC241B4
	s_waitcnt vmcnt(9) lgkmcnt(9)                              // 000000006D44: BF8C0979
	s_barrier                                                  // 000000006D48: BF8A0000
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]// 000000006D4C: D3E10074 0DD241B6
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]// 000000006D54: D3E10078 0DE241B8
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]// 000000006D5C: D3E1007C 0DF241BA
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]// 000000006D64: D3E10080 0E0241BC
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]// 000000006D6C: D3E10084 0E1241BE
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]// 000000006D74: D3E10088 0E2241C0
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]// 000000006D7C: D3E1008C 0E3241C2
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]// 000000006D84: D3E10090 0E4241C4
	v_mfma_f32_16x16x16_bf16 v[148:151], a[198:199], v[32:33], v[148:151]// 000000006D8C: D3E10094 0E5241C6
	v_mfma_f32_16x16x16_bf16 v[152:155], a[200:201], v[32:33], v[152:155]// 000000006D94: D3E10098 0E6241C8
	v_mfma_f32_16x16x16_bf16 v[156:159], a[202:203], v[32:33], v[156:159]// 000000006D9C: D3E1009C 0E7241CA
	v_mfma_f32_16x16x16_bf16 v[160:163], a[204:205], v[32:33], v[160:163]// 000000006DA4: D3E100A0 0E8241CC
	v_mfma_f32_16x16x16_bf16 v[164:167], a[206:207], v[32:33], v[164:167]// 000000006DAC: D3E100A4 0E9241CE
	s_nop 8                                                    // 000000006DB4: BF800008
	s_branch label_12AF                                        // 000000006DB8: BF820000

0000000000006dbc <label_12AF>:
	ds_write_b32 v3, v14 offset:55552                          // 000000006DBC: D81AD900 00000E03
	ds_write_b32 v3, v15 offset:56576                          // 000000006DC4: D81ADD00 00000F03
	s_waitcnt lgkmcnt(0)                                       // 000000006DCC: BF8CC07F
	ds_read_b32 v20, v2 offset:55552                           // 000000006DD0: D86CD900 14000002
	ds_read_b32 v21, v2 offset:55616                           // 000000006DD8: D86CD940 15000002
	ds_read_b32 v22, v2 offset:55680                           // 000000006DE0: D86CD980 16000002
	ds_read_b32 v23, v2 offset:55744                           // 000000006DE8: D86CD9C0 17000002
	ds_read_b32 v24, v2 offset:56576                           // 000000006DF0: D86CDD00 18000002
	ds_read_b32 v25, v2 offset:56640                           // 000000006DF8: D86CDD40 19000002
	ds_read_b32 v26, v2 offset:56704                           // 000000006E00: D86CDD80 1A000002
	ds_read_b32 v27, v2 offset:56768                           // 000000006E08: D86CDDC0 1B000002
	s_waitcnt lgkmcnt(0)                                       // 000000006E10: BF8CC07F
	v_mov_b32_e32 v14, 0                                       // 000000006E14: 7E1C0280
	v_mov_b32_e32 v15, 0                                       // 000000006E18: 7E1E0280
	v_add_f32_e32 v14, v20, v14                                // 000000006E1C: 021C1D14
	v_add_f32_e32 v15, v24, v15                                // 000000006E20: 021E1F18
	v_add_f32_e32 v14, v21, v14                                // 000000006E24: 021C1D15
	v_add_f32_e32 v15, v25, v15                                // 000000006E28: 021E1F19
	v_add_f32_e32 v14, v22, v14                                // 000000006E2C: 021C1D16
	v_add_f32_e32 v15, v26, v15                                // 000000006E30: 021E1F1A
	v_add_f32_e32 v14, v23, v14                                // 000000006E34: 021C1D17
	v_add_f32_e32 v15, v27, v15                                // 000000006E38: 021E1F1B
	v_mov_b32_e32 v20, 0                                       // 000000006E3C: 7E280280
	v_cmp_eq_u32_e64 s[38:39], v20, v14                        // 000000006E40: D0CA0026 00021D14
	v_cmp_eq_u32_e64 s[40:41], v20, v15                        // 000000006E48: D0CA0028 00021F14
	v_mul_f32_e64 v20, v12, s64                                // 000000006E50: D1050014 0000810C
	v_mul_f32_e64 v22, v13, s64                                // 000000006E58: D1050016 0000810D
	v_log_f32_e32 v21, v14                                     // 000000006E60: 7E2A430E
	v_log_f32_e32 v23, v15                                     // 000000006E64: 7E2E430F
	v_cndmask_b32_e64 v14, v14, 1.0, s[38:39]                  // 000000006E68: D100000E 0099E50E
	v_cndmask_b32_e64 v15, v15, 1.0, s[40:41]                  // 000000006E70: D100000F 00A1E50F
	s_nop 1                                                    // 000000006E78: BF800001
	v_rcp_f32_e32 v14, v14                                     // 000000006E7C: 7E1C450E
	v_rcp_f32_e32 v15, v15                                     // 000000006E80: 7E1E450F
	s_nop 1                                                    // 000000006E84: BF800001
	v_fma_f32 v24, v21, s63, v20                               // 000000006E88: D1CB0018 04507F15
	v_fma_f32 v25, v23, s63, v22                               // 000000006E90: D1CB0019 04587F17
	v_mul_f32_e32 v40, v14, v40                                // 000000006E98: 0A50510E
	v_mul_f32_e32 v41, v14, v41                                // 000000006E9C: 0A52530E
	v_mul_f32_e32 v42, v14, v42                                // 000000006EA0: 0A54550E
	v_mul_f32_e32 v43, v14, v43                                // 000000006EA4: 0A56570E
	v_mul_f32_e32 v44, v14, v44                                // 000000006EA8: 0A58590E
	v_mul_f32_e32 v45, v14, v45                                // 000000006EAC: 0A5A5B0E
	v_mul_f32_e32 v46, v14, v46                                // 000000006EB0: 0A5C5D0E
	v_mul_f32_e32 v47, v14, v47                                // 000000006EB4: 0A5E5F0E
	v_mul_f32_e32 v48, v14, v48                                // 000000006EB8: 0A60610E
	v_mul_f32_e32 v49, v14, v49                                // 000000006EBC: 0A62630E
	v_mul_f32_e32 v50, v14, v50                                // 000000006EC0: 0A64650E
	v_mul_f32_e32 v51, v14, v51                                // 000000006EC4: 0A66670E
	v_mul_f32_e32 v52, v14, v52                                // 000000006EC8: 0A68690E
	v_mul_f32_e32 v53, v14, v53                                // 000000006ECC: 0A6A6B0E
	v_mul_f32_e32 v54, v14, v54                                // 000000006ED0: 0A6C6D0E
	v_mul_f32_e32 v55, v14, v55                                // 000000006ED4: 0A6E6F0E
	v_mul_f32_e32 v56, v14, v56                                // 000000006ED8: 0A70710E
	v_mul_f32_e32 v57, v14, v57                                // 000000006EDC: 0A72730E
	v_mul_f32_e32 v58, v14, v58                                // 000000006EE0: 0A74750E
	v_mul_f32_e32 v59, v14, v59                                // 000000006EE4: 0A76770E
	v_mul_f32_e32 v60, v14, v60                                // 000000006EE8: 0A78790E
	v_mul_f32_e32 v61, v14, v61                                // 000000006EEC: 0A7A7B0E
	v_mul_f32_e32 v62, v14, v62                                // 000000006EF0: 0A7C7D0E
	v_mul_f32_e32 v63, v14, v63                                // 000000006EF4: 0A7E7F0E
	v_mul_f32_e32 v64, v14, v64                                // 000000006EF8: 0A80810E
	v_mul_f32_e32 v65, v14, v65                                // 000000006EFC: 0A82830E
	v_mul_f32_e32 v66, v14, v66                                // 000000006F00: 0A84850E
	v_mul_f32_e32 v67, v14, v67                                // 000000006F04: 0A86870E
	v_mul_f32_e32 v68, v14, v68                                // 000000006F08: 0A88890E
	v_mul_f32_e32 v69, v14, v69                                // 000000006F0C: 0A8A8B0E
	v_mul_f32_e32 v70, v14, v70                                // 000000006F10: 0A8C8D0E
	v_mul_f32_e32 v71, v14, v71                                // 000000006F14: 0A8E8F0E
	v_mul_f32_e32 v72, v14, v72                                // 000000006F18: 0A90910E
	v_mul_f32_e32 v73, v14, v73                                // 000000006F1C: 0A92930E
	v_mul_f32_e32 v74, v14, v74                                // 000000006F20: 0A94950E
	v_mul_f32_e32 v75, v14, v75                                // 000000006F24: 0A96970E
	v_mul_f32_e32 v76, v14, v76                                // 000000006F28: 0A98990E
	v_mul_f32_e32 v77, v14, v77                                // 000000006F2C: 0A9A9B0E
	v_mul_f32_e32 v78, v14, v78                                // 000000006F30: 0A9C9D0E
	v_mul_f32_e32 v79, v14, v79                                // 000000006F34: 0A9E9F0E
	v_mul_f32_e32 v80, v14, v80                                // 000000006F38: 0AA0A10E
	v_mul_f32_e32 v81, v14, v81                                // 000000006F3C: 0AA2A30E
	v_mul_f32_e32 v82, v14, v82                                // 000000006F40: 0AA4A50E
	v_mul_f32_e32 v83, v14, v83                                // 000000006F44: 0AA6A70E
	v_mul_f32_e32 v84, v14, v84                                // 000000006F48: 0AA8A90E
	v_mul_f32_e32 v85, v14, v85                                // 000000006F4C: 0AAAAB0E
	v_mul_f32_e32 v86, v14, v86                                // 000000006F50: 0AACAD0E
	v_mul_f32_e32 v87, v14, v87                                // 000000006F54: 0AAEAF0E
	v_mul_f32_e32 v88, v14, v88                                // 000000006F58: 0AB0B10E
	v_mul_f32_e32 v89, v14, v89                                // 000000006F5C: 0AB2B30E
	v_mul_f32_e32 v90, v14, v90                                // 000000006F60: 0AB4B50E
	v_mul_f32_e32 v91, v14, v91                                // 000000006F64: 0AB6B70E
	v_mul_f32_e32 v92, v14, v92                                // 000000006F68: 0AB8B90E
	v_mul_f32_e32 v93, v14, v93                                // 000000006F6C: 0ABABB0E
	v_mul_f32_e32 v94, v14, v94                                // 000000006F70: 0ABCBD0E
	v_mul_f32_e32 v95, v14, v95                                // 000000006F74: 0ABEBF0E
	v_mul_f32_e32 v96, v14, v96                                // 000000006F78: 0AC0C10E
	v_mul_f32_e32 v97, v14, v97                                // 000000006F7C: 0AC2C30E
	v_mul_f32_e32 v98, v14, v98                                // 000000006F80: 0AC4C50E
	v_mul_f32_e32 v99, v14, v99                                // 000000006F84: 0AC6C70E
	v_mul_f32_e32 v100, v14, v100                              // 000000006F88: 0AC8C90E
	v_mul_f32_e32 v101, v14, v101                              // 000000006F8C: 0ACACB0E
	v_mul_f32_e32 v102, v14, v102                              // 000000006F90: 0ACCCD0E
	v_mul_f32_e32 v103, v14, v103                              // 000000006F94: 0ACECF0E
	v_mul_f32_e32 v104, v14, v104                              // 000000006F98: 0AD0D10E
	v_mul_f32_e32 v105, v14, v105                              // 000000006F9C: 0AD2D30E
	v_mul_f32_e32 v106, v14, v106                              // 000000006FA0: 0AD4D50E
	v_mul_f32_e32 v107, v14, v107                              // 000000006FA4: 0AD6D70E
	v_mul_f32_e32 v108, v14, v108                              // 000000006FA8: 0AD8D90E
	v_mul_f32_e32 v109, v14, v109                              // 000000006FAC: 0ADADB0E
	v_mul_f32_e32 v110, v14, v110                              // 000000006FB0: 0ADCDD0E
	v_mul_f32_e32 v111, v14, v111                              // 000000006FB4: 0ADEDF0E
	v_mul_f32_e32 v112, v14, v112                              // 000000006FB8: 0AE0E10E
	v_mul_f32_e32 v113, v14, v113                              // 000000006FBC: 0AE2E30E
	v_mul_f32_e32 v114, v14, v114                              // 000000006FC0: 0AE4E50E
	v_mul_f32_e32 v115, v14, v115                              // 000000006FC4: 0AE6E70E
	v_mul_f32_e32 v116, v14, v116                              // 000000006FC8: 0AE8E90E
	v_mul_f32_e32 v117, v14, v117                              // 000000006FCC: 0AEAEB0E
	v_mul_f32_e32 v118, v14, v118                              // 000000006FD0: 0AECED0E
	v_mul_f32_e32 v119, v14, v119                              // 000000006FD4: 0AEEEF0E
	v_mul_f32_e32 v120, v14, v120                              // 000000006FD8: 0AF0F10E
	v_mul_f32_e32 v121, v14, v121                              // 000000006FDC: 0AF2F30E
	v_mul_f32_e32 v122, v14, v122                              // 000000006FE0: 0AF4F50E
	v_mul_f32_e32 v123, v14, v123                              // 000000006FE4: 0AF6F70E
	v_mul_f32_e32 v124, v14, v124                              // 000000006FE8: 0AF8F90E
	v_mul_f32_e32 v125, v14, v125                              // 000000006FEC: 0AFAFB0E
	v_mul_f32_e32 v126, v14, v126                              // 000000006FF0: 0AFCFD0E
	v_mul_f32_e32 v127, v14, v127                              // 000000006FF4: 0AFEFF0E
	v_mul_f32_e32 v128, v14, v128                              // 000000006FF8: 0B01010E
	v_mul_f32_e32 v129, v14, v129                              // 000000006FFC: 0B03030E
	v_mul_f32_e32 v130, v14, v130                              // 000000007000: 0B05050E
	v_mul_f32_e32 v131, v14, v131                              // 000000007004: 0B07070E
	v_mul_f32_e32 v132, v14, v132                              // 000000007008: 0B09090E
	v_mul_f32_e32 v133, v14, v133                              // 00000000700C: 0B0B0B0E
	v_mul_f32_e32 v134, v14, v134                              // 000000007010: 0B0D0D0E
	v_mul_f32_e32 v135, v14, v135                              // 000000007014: 0B0F0F0E
	v_mul_f32_e32 v136, v14, v136                              // 000000007018: 0B11110E
	v_mul_f32_e32 v137, v14, v137                              // 00000000701C: 0B13130E
	v_mul_f32_e32 v138, v14, v138                              // 000000007020: 0B15150E
	v_mul_f32_e32 v139, v14, v139                              // 000000007024: 0B17170E
	v_mul_f32_e32 v140, v14, v140                              // 000000007028: 0B19190E
	v_mul_f32_e32 v141, v14, v141                              // 00000000702C: 0B1B1B0E
	v_mul_f32_e32 v142, v14, v142                              // 000000007030: 0B1D1D0E
	v_mul_f32_e32 v143, v14, v143                              // 000000007034: 0B1F1F0E
	v_mul_f32_e32 v144, v14, v144                              // 000000007038: 0B21210E
	v_mul_f32_e32 v145, v14, v145                              // 00000000703C: 0B23230E
	v_mul_f32_e32 v146, v14, v146                              // 000000007040: 0B25250E
	v_mul_f32_e32 v147, v14, v147                              // 000000007044: 0B27270E
	v_mul_f32_e32 v148, v14, v148                              // 000000007048: 0B29290E
	v_mul_f32_e32 v149, v14, v149                              // 00000000704C: 0B2B2B0E
	v_mul_f32_e32 v150, v14, v150                              // 000000007050: 0B2D2D0E
	v_mul_f32_e32 v151, v14, v151                              // 000000007054: 0B2F2F0E
	v_mul_f32_e32 v152, v14, v152                              // 000000007058: 0B31310E
	v_mul_f32_e32 v153, v14, v153                              // 00000000705C: 0B33330E
	v_mul_f32_e32 v154, v14, v154                              // 000000007060: 0B35350E
	v_mul_f32_e32 v155, v14, v155                              // 000000007064: 0B37370E
	v_mul_f32_e32 v156, v14, v156                              // 000000007068: 0B39390E
	v_mul_f32_e32 v157, v14, v157                              // 00000000706C: 0B3B3B0E
	v_mul_f32_e32 v158, v14, v158                              // 000000007070: 0B3D3D0E
	v_mul_f32_e32 v159, v14, v159                              // 000000007074: 0B3F3F0E
	v_mul_f32_e32 v160, v14, v160                              // 000000007078: 0B41410E
	v_mul_f32_e32 v161, v14, v161                              // 00000000707C: 0B43430E
	v_mul_f32_e32 v162, v14, v162                              // 000000007080: 0B45450E
	v_mul_f32_e32 v163, v14, v163                              // 000000007084: 0B47470E
	v_mul_f32_e32 v164, v14, v164                              // 000000007088: 0B49490E
	v_mul_f32_e32 v165, v14, v165                              // 00000000708C: 0B4B4B0E
	v_mul_f32_e32 v166, v14, v166                              // 000000007090: 0B4D4D0E
	v_mul_f32_e32 v167, v14, v167                              // 000000007094: 0B4F4F0E
	s_cmp_lt_i32 s87, 0                                        // 000000007098: BF048057
	s_cbranch_scc0 label_185B                                  // 00000000709C: BF8404F3
	s_mov_b32 s75, 0x4000                                      // 0000000070A0: BECB00FF 00004000
	s_mul_i32 s56, s75, s78                                    // 0000000070A8: 92384E4B
	s_add_u32 s88, s56, s88                                    // 0000000070AC: 80585838
	s_addc_u32 s89, 0, s89                                     // 0000000070B0: 82595980
	s_sub_u32 s56, s81, s80                                    // 0000000070B4: 80B85051
	s_mul_i32 s56, s56, s75                                    // 0000000070B8: 92384B38
	s_mov_b32 s90, s56                                         // 0000000070BC: BEDA0038
	v_and_b32_e32 v20, 7, v0                                   // 0000000070C0: 26280087
	v_lshlrev_b32_e32 v18, 4, v20                              // 0000000070C4: 24242884
	v_lshrrev_b32_e32 v20, 3, v0                               // 0000000070C8: 20280083
	v_mul_i32_i24_e32 v20, 0x400, v20                          // 0000000070CC: 0C2828FF 00000400
	s_mul_i32 s57, s75, s7                                     // 0000000070D4: 9239074B
	v_add_u32_e32 v20, s57, v20                                // 0000000070D8: 68282839
	v_add_u32_e32 v18, v18, v20                                // 0000000070DC: 68242912
	v_mov_b32_e32 v19, v18                                     // 0000000070E0: 7E260312
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000070E4: BF8C0070
	s_barrier                                                  // 0000000070E8: BF8A0000
	s_mul_i32 s75, 0x400, s65                                  // 0000000070EC: 924B41FF 00000400
	s_mul_i32 s76, s67, s75                                    // 0000000070F4: 924C4B43
	v_lshrrev_b32_e32 v20, 4, v0                               // 0000000070F8: 20280084
	v_mul_i32_i24_e32 v5, 0x48, v20                            // 0000000070FC: 0C0A28FF 00000048
	v_and_b32_e32 v20, 15, v0                                  // 000000007104: 2628008F
	v_mul_i32_i24_e32 v20, 2, v20                              // 000000007108: 0C282882
	v_add_u32_e32 v5, v20, v5                                  // 00000000710C: 680A0B14
	s_mul_i32 s56, s7, 0x480                                   // 000000007110: 9238FF07 00000480
	v_add_u32_e32 v5, s56, v5                                  // 000000007118: 680A0A38
	v_lshlrev_b32_e32 v5, 2, v5                                // 00000000711C: 240A0A82
	v_lshrrev_b32_e32 v20, 3, v0                               // 000000007120: 20280083
	v_mul_i32_i24_e32 v4, 2, v20                               // 000000007124: 0C082882
	v_and_b32_e32 v20, 7, v0                                   // 000000007128: 26280087
	v_mul_i32_i24_e32 v20, 36, v20                             // 00000000712C: 0C2828A4
	v_add_u32_e32 v4, v20, v4                                  // 000000007130: 68080914
	s_mul_i32 s56, s7, 0x480                                   // 000000007134: 9238FF07 00000480
	v_add_u32_e32 v4, s56, v4                                  // 00000000713C: 68080838
	v_lshlrev_b32_e32 v4, 2, v4                                // 000000007140: 24080882
	v_mov_b32_e32 v29, 0xffff0000                              // 000000007144: 7E3A02FF FFFF0000
	v_mov_b32_e32 v30, 0x7fff0000                              // 00000000714C: 7E3C02FF 7FFF0000
	v_mov_b32_e32 v31, 0x7fff                                  // 000000007154: 7E3E02FF 00007FFF
	s_mul_i32 s56, 0, s76                                      // 00000000715C: 92384C80
	v_add_u32_e64 v19, v19, s56                                // 000000007160: D1340013 00007113
	v_mul_f32_e32 v24, s43, v40                                // 000000007168: 0A30502B
	v_mul_f32_e32 v25, s43, v44                                // 00000000716C: 0A32582B
	v_mul_f32_e32 v26, s43, v48                                // 000000007170: 0A34602B
	v_mul_f32_e32 v27, s43, v52                                // 000000007174: 0A36682B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007178: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007180: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007188: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007190: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007198: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000071A0: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 0000000071A8: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 0000000071B0: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 0000000071B8: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000071C0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000071C8: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000071D0: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000071D8: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000071E0: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25]                                  // 0000000071E8: D89A0000 00001805
	v_mul_f32_e32 v24, s43, v41                                // 0000000071F0: 0A30522B
	v_mul_f32_e32 v25, s43, v45                                // 0000000071F4: 0A325A2B
	v_mul_f32_e32 v26, s43, v49                                // 0000000071F8: 0A34622B
	v_mul_f32_e32 v27, s43, v53                                // 0000000071FC: 0A366A2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007200: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007208: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007210: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007218: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007220: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007228: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007230: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007238: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007240: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007248: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007250: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007258: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007260: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007268: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1152                      // 000000007270: D89A0480 00001805
	v_mul_f32_e32 v24, s43, v42                                // 000000007278: 0A30542B
	v_mul_f32_e32 v25, s43, v46                                // 00000000727C: 0A325C2B
	v_mul_f32_e32 v26, s43, v50                                // 000000007280: 0A34642B
	v_mul_f32_e32 v27, s43, v54                                // 000000007284: 0A366C2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007288: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007290: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007298: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000072A0: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 0000000072A8: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000072B0: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 0000000072B8: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 0000000072C0: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 0000000072C8: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000072D0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000072D8: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000072E0: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000072E8: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000072F0: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:144                       // 0000000072F8: D89A0090 00001805
	v_mul_f32_e32 v24, s43, v43                                // 000000007300: 0A30562B
	v_mul_f32_e32 v25, s43, v47                                // 000000007304: 0A325E2B
	v_mul_f32_e32 v26, s43, v51                                // 000000007308: 0A34662B
	v_mul_f32_e32 v27, s43, v55                                // 00000000730C: 0A366E2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007310: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007318: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007320: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007328: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007330: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007338: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007340: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007348: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007350: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007358: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007360: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007368: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007370: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007378: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1296                      // 000000007380: D89A0510 00001805
	v_mul_f32_e32 v24, s43, v56                                // 000000007388: 0A30702B
	v_mul_f32_e32 v25, s43, v60                                // 00000000738C: 0A32782B
	v_mul_f32_e32 v26, s43, v64                                // 000000007390: 0A34802B
	v_mul_f32_e32 v27, s43, v68                                // 000000007394: 0A36882B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007398: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000073A0: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000073A8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000073B0: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 0000000073B8: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000073C0: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 0000000073C8: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 0000000073D0: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 0000000073D8: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000073E0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000073E8: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000073F0: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000073F8: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007400: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2304                      // 000000007408: D89A0900 00001805
	v_mul_f32_e32 v24, s43, v57                                // 000000007410: 0A30722B
	v_mul_f32_e32 v25, s43, v61                                // 000000007414: 0A327A2B
	v_mul_f32_e32 v26, s43, v65                                // 000000007418: 0A34822B
	v_mul_f32_e32 v27, s43, v69                                // 00000000741C: 0A368A2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007420: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007428: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007430: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007438: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007440: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007448: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007450: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007458: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007460: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007468: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007470: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007478: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007480: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007488: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3456                      // 000000007490: D89A0D80 00001805
	v_mul_f32_e32 v24, s43, v58                                // 000000007498: 0A30742B
	v_mul_f32_e32 v25, s43, v62                                // 00000000749C: 0A327C2B
	v_mul_f32_e32 v26, s43, v66                                // 0000000074A0: 0A34842B
	v_mul_f32_e32 v27, s43, v70                                // 0000000074A4: 0A368C2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000074A8: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000074B0: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000074B8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000074C0: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 0000000074C8: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000074D0: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 0000000074D8: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 0000000074E0: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 0000000074E8: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000074F0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000074F8: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007500: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007508: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007510: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2448                      // 000000007518: D89A0990 00001805
	v_mul_f32_e32 v24, s43, v59                                // 000000007520: 0A30762B
	v_mul_f32_e32 v25, s43, v63                                // 000000007524: 0A327E2B
	v_mul_f32_e32 v26, s43, v67                                // 000000007528: 0A34862B
	v_mul_f32_e32 v27, s43, v71                                // 00000000752C: 0A368E2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007530: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007538: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007540: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007548: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007550: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007558: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007560: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007568: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007570: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007578: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007580: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007588: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007590: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007598: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3600                      // 0000000075A0: D89A0E10 00001805
	s_waitcnt lgkmcnt(4)                                       // 0000000075A8: BF8CC47F
	ds_read_b64 v[40:41], v4                                   // 0000000075AC: D8EC0000 28000004
	ds_read_b64 v[44:45], v4 offset:64                         // 0000000075B4: D8EC0040 2C000004
	ds_read_b64 v[42:43], v4 offset:1152                       // 0000000075BC: D8EC0480 2A000004
	ds_read_b64 v[46:47], v4 offset:1216                       // 0000000075C4: D8EC04C0 2E000004
	s_waitcnt lgkmcnt(4)                                       // 0000000075CC: BF8CC47F
	ds_read_b64 v[48:49], v4 offset:2304                       // 0000000075D0: D8EC0900 30000004
	ds_read_b64 v[52:53], v4 offset:2368                       // 0000000075D8: D8EC0940 34000004
	ds_read_b64 v[50:51], v4 offset:3456                       // 0000000075E0: D8EC0D80 32000004
	ds_read_b64 v[54:55], v4 offset:3520                       // 0000000075E8: D8EC0DC0 36000004
	s_waitcnt lgkmcnt(0)                                       // 0000000075F0: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 0000000075F4: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[88:91], 0 offen      // 0000000075F8: E07C1000 80162812
	buffer_store_dwordx4 v[48:51], v18, s[88:91], 0 offen offset:128// 000000007600: E07C1080 80163012
	v_add_u32_e32 v18, 0x2000, v18                             // 000000007608: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[88:91], 0 offen      // 000000007610: E07C1000 80162C12
	buffer_store_dwordx4 v[52:55], v18, s[88:91], 0 offen offset:128// 000000007618: E07C1080 80163412
	v_add_u32_e32 v18, 0x2000, v18                             // 000000007620: 682424FF 00002000
	v_mul_f32_e32 v24, s43, v72                                // 000000007628: 0A30902B
	v_mul_f32_e32 v25, s43, v76                                // 00000000762C: 0A32982B
	v_mul_f32_e32 v26, s43, v80                                // 000000007630: 0A34A02B
	v_mul_f32_e32 v27, s43, v84                                // 000000007634: 0A36A82B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007638: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007640: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007648: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007650: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007658: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007660: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007668: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007670: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007678: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007680: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007688: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007690: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007698: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000076A0: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25]                                  // 0000000076A8: D89A0000 00001805
	v_mul_f32_e32 v24, s43, v73                                // 0000000076B0: 0A30922B
	v_mul_f32_e32 v25, s43, v77                                // 0000000076B4: 0A329A2B
	v_mul_f32_e32 v26, s43, v81                                // 0000000076B8: 0A34A22B
	v_mul_f32_e32 v27, s43, v85                                // 0000000076BC: 0A36AA2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000076C0: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000076C8: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000076D0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000076D8: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 0000000076E0: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000076E8: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 0000000076F0: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 0000000076F8: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007700: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007708: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007710: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007718: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007720: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007728: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1152                      // 000000007730: D89A0480 00001805
	v_mul_f32_e32 v24, s43, v74                                // 000000007738: 0A30942B
	v_mul_f32_e32 v25, s43, v78                                // 00000000773C: 0A329C2B
	v_mul_f32_e32 v26, s43, v82                                // 000000007740: 0A34A42B
	v_mul_f32_e32 v27, s43, v86                                // 000000007744: 0A36AC2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007748: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007750: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007758: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007760: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007768: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007770: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007778: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007780: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007788: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007790: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007798: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000077A0: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000077A8: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000077B0: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:144                       // 0000000077B8: D89A0090 00001805
	v_mul_f32_e32 v24, s43, v75                                // 0000000077C0: 0A30962B
	v_mul_f32_e32 v25, s43, v79                                // 0000000077C4: 0A329E2B
	v_mul_f32_e32 v26, s43, v83                                // 0000000077C8: 0A34A62B
	v_mul_f32_e32 v27, s43, v87                                // 0000000077CC: 0A36AE2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000077D0: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000077D8: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000077E0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000077E8: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 0000000077F0: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000077F8: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007800: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007808: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007810: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007818: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007820: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007828: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007830: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007838: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1296                      // 000000007840: D89A0510 00001805
	v_mul_f32_e32 v24, s43, v88                                // 000000007848: 0A30B02B
	v_mul_f32_e32 v25, s43, v92                                // 00000000784C: 0A32B82B
	v_mul_f32_e32 v26, s43, v96                                // 000000007850: 0A34C02B
	v_mul_f32_e32 v27, s43, v100                               // 000000007854: 0A36C82B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007858: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007860: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007868: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007870: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007878: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007880: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007888: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007890: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007898: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000078A0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000078A8: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000078B0: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000078B8: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000078C0: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2304                      // 0000000078C8: D89A0900 00001805
	v_mul_f32_e32 v24, s43, v89                                // 0000000078D0: 0A30B22B
	v_mul_f32_e32 v25, s43, v93                                // 0000000078D4: 0A32BA2B
	v_mul_f32_e32 v26, s43, v97                                // 0000000078D8: 0A34C22B
	v_mul_f32_e32 v27, s43, v101                               // 0000000078DC: 0A36CA2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000078E0: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000078E8: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000078F0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000078F8: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007900: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007908: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007910: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007918: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007920: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007928: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007930: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007938: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007940: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007948: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3456                      // 000000007950: D89A0D80 00001805
	v_mul_f32_e32 v24, s43, v90                                // 000000007958: 0A30B42B
	v_mul_f32_e32 v25, s43, v94                                // 00000000795C: 0A32BC2B
	v_mul_f32_e32 v26, s43, v98                                // 000000007960: 0A34C42B
	v_mul_f32_e32 v27, s43, v102                               // 000000007964: 0A36CC2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007968: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007970: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007978: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007980: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007988: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007990: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007998: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 0000000079A0: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 0000000079A8: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000079B0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000079B8: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000079C0: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000079C8: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000079D0: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2448                      // 0000000079D8: D89A0990 00001805
	v_mul_f32_e32 v24, s43, v91                                // 0000000079E0: 0A30B62B
	v_mul_f32_e32 v25, s43, v95                                // 0000000079E4: 0A32BE2B
	v_mul_f32_e32 v26, s43, v99                                // 0000000079E8: 0A34C62B
	v_mul_f32_e32 v27, s43, v103                               // 0000000079EC: 0A36CE2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000079F0: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000079F8: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007A00: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007A08: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007A10: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007A18: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007A20: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007A28: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007A30: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007A38: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007A40: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007A48: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007A50: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007A58: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3600                      // 000000007A60: D89A0E10 00001805
	s_waitcnt lgkmcnt(4)                                       // 000000007A68: BF8CC47F
	ds_read_b64 v[40:41], v4                                   // 000000007A6C: D8EC0000 28000004
	ds_read_b64 v[44:45], v4 offset:64                         // 000000007A74: D8EC0040 2C000004
	ds_read_b64 v[42:43], v4 offset:1152                       // 000000007A7C: D8EC0480 2A000004
	ds_read_b64 v[46:47], v4 offset:1216                       // 000000007A84: D8EC04C0 2E000004
	s_waitcnt lgkmcnt(4)                                       // 000000007A8C: BF8CC47F
	ds_read_b64 v[48:49], v4 offset:2304                       // 000000007A90: D8EC0900 30000004
	ds_read_b64 v[52:53], v4 offset:2368                       // 000000007A98: D8EC0940 34000004
	ds_read_b64 v[50:51], v4 offset:3456                       // 000000007AA0: D8EC0D80 32000004
	ds_read_b64 v[54:55], v4 offset:3520                       // 000000007AA8: D8EC0DC0 36000004
	s_waitcnt lgkmcnt(0)                                       // 000000007AB0: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 000000007AB4: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[88:91], 0 offen offset:256// 000000007AB8: E07C1100 80162812
	buffer_store_dwordx4 v[48:51], v18, s[88:91], 0 offen offset:384// 000000007AC0: E07C1180 80163012
	v_add_u32_e32 v18, 0x2000, v18                             // 000000007AC8: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[88:91], 0 offen offset:256// 000000007AD0: E07C1100 80162C12
	buffer_store_dwordx4 v[52:55], v18, s[88:91], 0 offen offset:384// 000000007AD8: E07C1180 80163412
	v_add_u32_e32 v18, 0x2000, v18                             // 000000007AE0: 682424FF 00002000
	v_mul_f32_e32 v24, s43, v104                               // 000000007AE8: 0A30D02B
	v_mul_f32_e32 v25, s43, v108                               // 000000007AEC: 0A32D82B
	v_mul_f32_e32 v26, s43, v112                               // 000000007AF0: 0A34E02B
	v_mul_f32_e32 v27, s43, v116                               // 000000007AF4: 0A36E82B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007AF8: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007B00: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007B08: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007B10: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007B18: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007B20: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007B28: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007B30: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007B38: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007B40: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007B48: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007B50: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007B58: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007B60: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25]                                  // 000000007B68: D89A0000 00001805
	v_mul_f32_e32 v24, s43, v105                               // 000000007B70: 0A30D22B
	v_mul_f32_e32 v25, s43, v109                               // 000000007B74: 0A32DA2B
	v_mul_f32_e32 v26, s43, v113                               // 000000007B78: 0A34E22B
	v_mul_f32_e32 v27, s43, v117                               // 000000007B7C: 0A36EA2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007B80: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007B88: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007B90: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007B98: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007BA0: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007BA8: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007BB0: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007BB8: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007BC0: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007BC8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007BD0: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007BD8: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007BE0: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007BE8: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1152                      // 000000007BF0: D89A0480 00001805
	v_mul_f32_e32 v24, s43, v106                               // 000000007BF8: 0A30D42B
	v_mul_f32_e32 v25, s43, v110                               // 000000007BFC: 0A32DC2B
	v_mul_f32_e32 v26, s43, v114                               // 000000007C00: 0A34E42B
	v_mul_f32_e32 v27, s43, v118                               // 000000007C04: 0A36EC2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007C08: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007C10: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007C18: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007C20: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007C28: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007C30: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007C38: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007C40: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007C48: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007C50: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007C58: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007C60: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007C68: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007C70: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:144                       // 000000007C78: D89A0090 00001805
	v_mul_f32_e32 v24, s43, v107                               // 000000007C80: 0A30D62B
	v_mul_f32_e32 v25, s43, v111                               // 000000007C84: 0A32DE2B
	v_mul_f32_e32 v26, s43, v115                               // 000000007C88: 0A34E62B
	v_mul_f32_e32 v27, s43, v119                               // 000000007C8C: 0A36EE2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007C90: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007C98: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007CA0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007CA8: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007CB0: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007CB8: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007CC0: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007CC8: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007CD0: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007CD8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007CE0: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007CE8: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007CF0: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007CF8: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1296                      // 000000007D00: D89A0510 00001805
	v_mul_f32_e32 v24, s43, v120                               // 000000007D08: 0A30F02B
	v_mul_f32_e32 v25, s43, v124                               // 000000007D0C: 0A32F82B
	v_mul_f32_e32 v26, s43, v128                               // 000000007D10: 0A35002B
	v_mul_f32_e32 v27, s43, v132                               // 000000007D14: 0A37082B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007D18: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007D20: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007D28: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007D30: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007D38: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007D40: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007D48: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007D50: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007D58: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007D60: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007D68: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007D70: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007D78: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007D80: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2304                      // 000000007D88: D89A0900 00001805
	v_mul_f32_e32 v24, s43, v121                               // 000000007D90: 0A30F22B
	v_mul_f32_e32 v25, s43, v125                               // 000000007D94: 0A32FA2B
	v_mul_f32_e32 v26, s43, v129                               // 000000007D98: 0A35022B
	v_mul_f32_e32 v27, s43, v133                               // 000000007D9C: 0A370A2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007DA0: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007DA8: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007DB0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007DB8: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007DC0: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007DC8: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007DD0: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007DD8: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007DE0: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007DE8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007DF0: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007DF8: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007E00: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007E08: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3456                      // 000000007E10: D89A0D80 00001805
	v_mul_f32_e32 v24, s43, v122                               // 000000007E18: 0A30F42B
	v_mul_f32_e32 v25, s43, v126                               // 000000007E1C: 0A32FC2B
	v_mul_f32_e32 v26, s43, v130                               // 000000007E20: 0A35042B
	v_mul_f32_e32 v27, s43, v134                               // 000000007E24: 0A370C2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007E28: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007E30: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007E38: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007E40: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007E48: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007E50: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007E58: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007E60: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007E68: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007E70: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007E78: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007E80: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007E88: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007E90: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2448                      // 000000007E98: D89A0990 00001805
	v_mul_f32_e32 v24, s43, v123                               // 000000007EA0: 0A30F62B
	v_mul_f32_e32 v25, s43, v127                               // 000000007EA4: 0A32FE2B
	v_mul_f32_e32 v26, s43, v131                               // 000000007EA8: 0A35062B
	v_mul_f32_e32 v27, s43, v135                               // 000000007EAC: 0A370E2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007EB0: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007EB8: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007EC0: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007EC8: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007ED0: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007ED8: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007EE0: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007EE8: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007EF0: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007EF8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000007F00: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000007F08: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007F10: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000007F18: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3600                      // 000000007F20: D89A0E10 00001805
	s_waitcnt lgkmcnt(4)                                       // 000000007F28: BF8CC47F
	ds_read_b64 v[40:41], v4                                   // 000000007F2C: D8EC0000 28000004
	ds_read_b64 v[44:45], v4 offset:64                         // 000000007F34: D8EC0040 2C000004
	ds_read_b64 v[42:43], v4 offset:1152                       // 000000007F3C: D8EC0480 2A000004
	ds_read_b64 v[46:47], v4 offset:1216                       // 000000007F44: D8EC04C0 2E000004
	s_waitcnt lgkmcnt(4)                                       // 000000007F4C: BF8CC47F
	ds_read_b64 v[48:49], v4 offset:2304                       // 000000007F50: D8EC0900 30000004
	ds_read_b64 v[52:53], v4 offset:2368                       // 000000007F58: D8EC0940 34000004
	ds_read_b64 v[50:51], v4 offset:3456                       // 000000007F60: D8EC0D80 32000004
	ds_read_b64 v[54:55], v4 offset:3520                       // 000000007F68: D8EC0DC0 36000004
	s_waitcnt lgkmcnt(0)                                       // 000000007F70: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 000000007F74: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[88:91], 0 offen offset:512// 000000007F78: E07C1200 80162812
	buffer_store_dwordx4 v[48:51], v18, s[88:91], 0 offen offset:640// 000000007F80: E07C1280 80163012
	v_add_u32_e32 v18, 0x2000, v18                             // 000000007F88: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[88:91], 0 offen offset:512// 000000007F90: E07C1200 80162C12
	buffer_store_dwordx4 v[52:55], v18, s[88:91], 0 offen offset:640// 000000007F98: E07C1280 80163412
	v_add_u32_e32 v18, 0x2000, v18                             // 000000007FA0: 682424FF 00002000
	v_mul_f32_e32 v24, s43, v136                               // 000000007FA8: 0A31102B
	v_mul_f32_e32 v25, s43, v140                               // 000000007FAC: 0A33182B
	v_mul_f32_e32 v26, s43, v144                               // 000000007FB0: 0A35202B
	v_mul_f32_e32 v27, s43, v148                               // 000000007FB4: 0A37282B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000007FB8: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000007FC0: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000007FC8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000007FD0: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000007FD8: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000007FE0: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000007FE8: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000007FF0: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000007FF8: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008000: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000008008: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000008010: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008018: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000008020: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25]                                  // 000000008028: D89A0000 00001805
	v_mul_f32_e32 v24, s43, v137                               // 000000008030: 0A31122B
	v_mul_f32_e32 v25, s43, v141                               // 000000008034: 0A331A2B
	v_mul_f32_e32 v26, s43, v145                               // 000000008038: 0A35222B
	v_mul_f32_e32 v27, s43, v149                               // 00000000803C: 0A372A2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000008040: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000008048: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008050: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000008058: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000008060: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008068: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000008070: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000008078: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000008080: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008088: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000008090: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000008098: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000080A0: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000080A8: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1152                      // 0000000080B0: D89A0480 00001805
	v_mul_f32_e32 v24, s43, v138                               // 0000000080B8: 0A31142B
	v_mul_f32_e32 v25, s43, v142                               // 0000000080BC: 0A331C2B
	v_mul_f32_e32 v26, s43, v146                               // 0000000080C0: 0A35242B
	v_mul_f32_e32 v27, s43, v150                               // 0000000080C4: 0A372C2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000080C8: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000080D0: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000080D8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000080E0: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 0000000080E8: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000080F0: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 0000000080F8: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000008100: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000008108: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008110: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000008118: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000008120: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008128: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000008130: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:144                       // 000000008138: D89A0090 00001805
	v_mul_f32_e32 v24, s43, v139                               // 000000008140: 0A31162B
	v_mul_f32_e32 v25, s43, v143                               // 000000008144: 0A331E2B
	v_mul_f32_e32 v26, s43, v147                               // 000000008148: 0A35262B
	v_mul_f32_e32 v27, s43, v151                               // 00000000814C: 0A372E2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000008150: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000008158: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008160: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000008168: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000008170: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008178: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000008180: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000008188: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000008190: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008198: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000081A0: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000081A8: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000081B0: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000081B8: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:1296                      // 0000000081C0: D89A0510 00001805
	v_mul_f32_e32 v24, s43, v152                               // 0000000081C8: 0A31302B
	v_mul_f32_e32 v25, s43, v156                               // 0000000081CC: 0A33382B
	v_mul_f32_e32 v26, s43, v160                               // 0000000081D0: 0A35402B
	v_mul_f32_e32 v27, s43, v164                               // 0000000081D4: 0A37482B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000081D8: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000081E0: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000081E8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 0000000081F0: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 0000000081F8: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008200: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000008208: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000008210: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000008218: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008220: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000008228: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000008230: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008238: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000008240: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2304                      // 000000008248: D89A0900 00001805
	v_mul_f32_e32 v24, s43, v153                               // 000000008250: 0A31322B
	v_mul_f32_e32 v25, s43, v157                               // 000000008254: 0A333A2B
	v_mul_f32_e32 v26, s43, v161                               // 000000008258: 0A35422B
	v_mul_f32_e32 v27, s43, v165                               // 00000000825C: 0A374A2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000008260: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000008268: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008270: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000008278: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000008280: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008288: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000008290: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000008298: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 0000000082A0: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000082A8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000082B0: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000082B8: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000082C0: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000082C8: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3456                      // 0000000082D0: D89A0D80 00001805
	v_mul_f32_e32 v24, s43, v154                               // 0000000082D8: 0A31342B
	v_mul_f32_e32 v25, s43, v158                               // 0000000082DC: 0A333C2B
	v_mul_f32_e32 v26, s43, v162                               // 0000000082E0: 0A35442B
	v_mul_f32_e32 v27, s43, v166                               // 0000000082E4: 0A374C2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 0000000082E8: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 0000000082F0: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000082F8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000008300: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000008308: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008310: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 000000008318: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 000000008320: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 000000008328: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008330: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 000000008338: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 000000008340: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008348: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 000000008350: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:2448                      // 000000008358: D89A0990 00001805
	v_mul_f32_e32 v24, s43, v155                               // 000000008360: 0A31362B
	v_mul_f32_e32 v25, s43, v159                               // 000000008364: 0A333E2B
	v_mul_f32_e32 v26, s43, v163                               // 000000008368: 0A35462B
	v_mul_f32_e32 v27, s43, v167                               // 00000000836C: 0A374E2B
	v_cmp_u_f32_e64 s[38:39], v24, v24                         // 000000008370: D0480026 00023118
	v_add3_u32 v28, v24, v31, 1                                // 000000008378: D1FF001C 02063F18
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 000000008380: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v25, v25                         // 000000008388: D0480026 00023319
	v_add3_u32 v28, v25, v31, 1                                // 000000008390: D1FF001C 02063F19
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 000000008398: D1000015 009A3D1C
	v_perm_b32 v24, v21, v20, s52                              // 0000000083A0: D1ED0018 00D22915
	v_cmp_u_f32_e64 s[38:39], v26, v26                         // 0000000083A8: D0480026 0002351A
	v_add3_u32 v28, v26, v31, 1                                // 0000000083B0: D1FF001C 02063F1A
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]                  // 0000000083B8: D1000014 009A3D1C
	v_cmp_u_f32_e64 s[38:39], v27, v27                         // 0000000083C0: D0480026 0002371B
	v_add3_u32 v28, v27, v31, 1                                // 0000000083C8: D1FF001C 02063F1B
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]                  // 0000000083D0: D1000015 009A3D1C
	v_perm_b32 v25, v21, v20, s52                              // 0000000083D8: D1ED0019 00D22915
	ds_write_b64 v5, v[24:25] offset:3600                      // 0000000083E0: D89A0E10 00001805
	s_waitcnt lgkmcnt(4)                                       // 0000000083E8: BF8CC47F
	ds_read_b64 v[40:41], v4                                   // 0000000083EC: D8EC0000 28000004
	ds_read_b64 v[44:45], v4 offset:64                         // 0000000083F4: D8EC0040 2C000004
	ds_read_b64 v[42:43], v4 offset:1152                       // 0000000083FC: D8EC0480 2A000004
	ds_read_b64 v[46:47], v4 offset:1216                       // 000000008404: D8EC04C0 2E000004
	s_waitcnt lgkmcnt(4)                                       // 00000000840C: BF8CC47F
	ds_read_b64 v[48:49], v4 offset:2304                       // 000000008410: D8EC0900 30000004
	ds_read_b64 v[52:53], v4 offset:2368                       // 000000008418: D8EC0940 34000004
	ds_read_b64 v[50:51], v4 offset:3456                       // 000000008420: D8EC0D80 32000004
	ds_read_b64 v[54:55], v4 offset:3520                       // 000000008428: D8EC0DC0 36000004
	s_waitcnt lgkmcnt(0)                                       // 000000008430: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 000000008434: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[88:91], 0 offen offset:768// 000000008438: E07C1300 80162812
	buffer_store_dwordx4 v[48:51], v18, s[88:91], 0 offen offset:896// 000000008440: E07C1380 80163012
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008448: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[88:91], 0 offen offset:768// 000000008450: E07C1300 80162C12
	buffer_store_dwordx4 v[52:55], v18, s[88:91], 0 offen offset:896// 000000008458: E07C1380 80163412
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008460: 682424FF 00002000
	s_branch label_1A09                                        // 000000008468: BF8201AE

000000000000846c <label_185B>:
	s_mov_b32 s75, 0x8000                                      // 00000000846C: BECB00FF 00008000
	s_mul_i32 s76, s87, s75                                    // 000000008474: 924C4B57
	s_mov_b32 s56, s76                                         // 000000008478: BEB8004C
	s_add_u32 s8, s56, s8                                      // 00000000847C: 80080838
	s_addc_u32 s9, 0, s9                                       // 000000008480: 82090980
	s_sub_u32 s56, s81, s80                                    // 000000008484: 80B85051
	s_mul_i32 s56, s56, s75                                    // 000000008488: 92384B38
	s_mov_b32 s10, s56                                         // 00000000848C: BE8A0038
	v_and_b32_e32 v20, 15, v0                                  // 000000008490: 2628008F
	v_lshlrev_b32_e32 v18, 4, v20                              // 000000008494: 24242884
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000008498: 20280084
	v_mul_i32_i24_e32 v20, 0x800, v20                          // 00000000849C: 0C2828FF 00000800
	v_add_u32_e32 v18, v18, v20                                // 0000000084A4: 68242912
	s_mul_i32 s56, s7, s75                                     // 0000000084A8: 92384B07
	v_add_u32_e64 v18, v18, s56                                // 0000000084AC: D1340012 00007112
	v_mov_b32_e32 v19, v18                                     // 0000000084B4: 7E260312
	s_mov_b32 s58, 64                                          // 0000000084B8: BEBA00C0
	s_mul_i32 s56, s58, s87                                    // 0000000084BC: 9238573A
	s_add_u32 s12, s56, s12                                    // 0000000084C0: 800C0C38
	s_addc_u32 s13, 0, s13                                     // 0000000084C4: 820D0D80
	s_sub_u32 s56, s81, s80                                    // 0000000084C8: 80B85051
	s_mul_i32 s56, s56, s58                                    // 0000000084CC: 92383A38
	s_mov_b32 s14, s56                                         // 0000000084D0: BE8E0038
	v_and_b32_e32 v26, 15, v0                                  // 0000000084D4: 2634008F
	v_lshlrev_b32_e32 v26, 2, v26                              // 0000000084D8: 24343482
	s_mul_i32 s56, s7, s57                                     // 0000000084DC: 92383907
	s_mul_i32 s57, s58, s7                                     // 0000000084E0: 9239073A
	v_add_u32_e64 v26, v26, s57                                // 0000000084E4: D134001A 0000731A
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000084EC: BF8C0070
	s_barrier                                                  // 0000000084F0: BF8A0000
	v_lshlrev_b32_e32 v5, 2, v0                                // 0000000084F4: 240A0082
	s_mul_i32 s56, s7, 0x840                                   // 0000000084F8: 9238FF07 00000840
	v_add_u32_e32 v5, s56, v5                                  // 000000008500: 680A0A38
	v_lshlrev_b32_e32 v5, 2, v5                                // 000000008504: 240A0A82
	v_lshrrev_b32_e32 v20, 4, v0                               // 000000008508: 20280084
	v_mul_i32_i24_e32 v4, 4, v20                               // 00000000850C: 0C082884
	v_and_b32_e32 v20, 3, v0                                   // 000000008510: 26280083
	v_mul_i32_i24_e32 v20, 0x108, v20                          // 000000008514: 0C2828FF 00000108
	v_add_u32_e32 v4, v20, v4                                  // 00000000851C: 68080914
	v_and_b32_e32 v20, 15, v0                                  // 000000008520: 2628008F
	v_lshrrev_b32_e32 v20, 2, v20                              // 000000008524: 20282882
	v_mul_i32_i24_e32 v20, 64, v20                             // 000000008528: 0C2828C0
	v_add_u32_e32 v4, v20, v4                                  // 00000000852C: 68080914
	s_mul_i32 s56, s7, 0x840                                   // 000000008530: 9238FF07 00000840
	v_add_u32_e32 v4, s56, v4                                  // 000000008538: 68080838
	v_lshlrev_b32_e32 v4, 2, v4                                // 00000000853C: 24080882
	s_mul_i32 s56, 0, s76                                      // 000000008540: 92384C80
	v_add_u32_e64 v19, v19, s56                                // 000000008544: D1340013 00007113
	v_mul_f32_e32 v20, s43, v40                                // 00000000854C: 0A28502B
	v_mul_f32_e32 v21, s43, v44                                // 000000008550: 0A2A582B
	v_mul_f32_e32 v22, s43, v48                                // 000000008554: 0A2C602B
	v_mul_f32_e32 v23, s43, v52                                // 000000008558: 0A2E682B
	ds_write_b128 v5, v[20:23]                                 // 00000000855C: D9BE0000 00001405
	v_mul_f32_e32 v20, s43, v41                                // 000000008564: 0A28522B
	v_mul_f32_e32 v21, s43, v45                                // 000000008568: 0A2A5A2B
	v_mul_f32_e32 v22, s43, v49                                // 00000000856C: 0A2C622B
	v_mul_f32_e32 v23, s43, v53                                // 000000008570: 0A2E6A2B
	ds_write_b128 v5, v[20:23] offset:1056                     // 000000008574: D9BE0420 00001405
	v_mul_f32_e32 v20, s43, v42                                // 00000000857C: 0A28542B
	v_mul_f32_e32 v21, s43, v46                                // 000000008580: 0A2A5C2B
	v_mul_f32_e32 v22, s43, v50                                // 000000008584: 0A2C642B
	v_mul_f32_e32 v23, s43, v54                                // 000000008588: 0A2E6C2B
	ds_write_b128 v5, v[20:23] offset:2112                     // 00000000858C: D9BE0840 00001405
	v_mul_f32_e32 v20, s43, v43                                // 000000008594: 0A28562B
	v_mul_f32_e32 v21, s43, v47                                // 000000008598: 0A2A5E2B
	v_mul_f32_e32 v22, s43, v51                                // 00000000859C: 0A2C662B
	v_mul_f32_e32 v23, s43, v55                                // 0000000085A0: 0A2E6E2B
	ds_write_b128 v5, v[20:23] offset:3168                     // 0000000085A4: D9BE0C60 00001405
	v_mul_f32_e32 v20, s43, v56                                // 0000000085AC: 0A28702B
	v_mul_f32_e32 v21, s43, v60                                // 0000000085B0: 0A2A782B
	v_mul_f32_e32 v22, s43, v64                                // 0000000085B4: 0A2C802B
	v_mul_f32_e32 v23, s43, v68                                // 0000000085B8: 0A2E882B
	ds_write_b128 v5, v[20:23] offset:4224                     // 0000000085BC: D9BE1080 00001405
	v_mul_f32_e32 v20, s43, v57                                // 0000000085C4: 0A28722B
	v_mul_f32_e32 v21, s43, v61                                // 0000000085C8: 0A2A7A2B
	v_mul_f32_e32 v22, s43, v65                                // 0000000085CC: 0A2C822B
	v_mul_f32_e32 v23, s43, v69                                // 0000000085D0: 0A2E8A2B
	ds_write_b128 v5, v[20:23] offset:5280                     // 0000000085D4: D9BE14A0 00001405
	v_mul_f32_e32 v20, s43, v58                                // 0000000085DC: 0A28742B
	v_mul_f32_e32 v21, s43, v62                                // 0000000085E0: 0A2A7C2B
	v_mul_f32_e32 v22, s43, v66                                // 0000000085E4: 0A2C842B
	v_mul_f32_e32 v23, s43, v70                                // 0000000085E8: 0A2E8C2B
	ds_write_b128 v5, v[20:23] offset:6336                     // 0000000085EC: D9BE18C0 00001405
	v_mul_f32_e32 v20, s43, v59                                // 0000000085F4: 0A28762B
	v_mul_f32_e32 v21, s43, v63                                // 0000000085F8: 0A2A7E2B
	v_mul_f32_e32 v22, s43, v67                                // 0000000085FC: 0A2C862B
	v_mul_f32_e32 v23, s43, v71                                // 000000008600: 0A2E8E2B
	ds_write_b128 v5, v[20:23] offset:7392                     // 000000008604: D9BE1CE0 00001405
	s_waitcnt lgkmcnt(4)                                       // 00000000860C: BF8CC47F
	ds_read_b128 v[40:43], v4                                  // 000000008610: D9FE0000 28000004
	ds_read_b128 v[44:47], v4 offset:64                        // 000000008618: D9FE0040 2C000004
	ds_read_b128 v[48:51], v4 offset:128                       // 000000008620: D9FE0080 30000004
	ds_read_b128 v[52:55], v4 offset:192                       // 000000008628: D9FE00C0 34000004
	s_waitcnt lgkmcnt(4)                                       // 000000008630: BF8CC47F
	ds_read_b128 v[56:59], v4 offset:4224                      // 000000008634: D9FE1080 38000004
	ds_read_b128 v[60:63], v4 offset:4288                      // 00000000863C: D9FE10C0 3C000004
	ds_read_b128 v[64:67], v4 offset:4352                      // 000000008644: D9FE1100 40000004
	ds_read_b128 v[68:71], v4 offset:4416                      // 00000000864C: D9FE1140 44000004
	s_waitcnt lgkmcnt(0)                                       // 000000008654: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 000000008658: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen       // 00000000865C: E07C1000 80022812
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:256// 000000008664: E07C1100 80023812
	v_add_u32_e32 v18, 0x2000, v18                             // 00000000866C: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen       // 000000008674: E07C1000 80022C12
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:256// 00000000867C: E07C1100 80023C12
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008684: 682424FF 00002000
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen       // 00000000868C: E07C1000 80023012
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:256// 000000008694: E07C1100 80024012
	v_add_u32_e32 v18, 0x2000, v18                             // 00000000869C: 682424FF 00002000
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen       // 0000000086A4: E07C1000 80023412
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:256// 0000000086AC: E07C1100 80024412
	v_add_u32_e32 v18, 0x2000, v18                             // 0000000086B4: 682424FF 00002000
	v_mul_f32_e32 v20, s43, v72                                // 0000000086BC: 0A28902B
	v_mul_f32_e32 v21, s43, v76                                // 0000000086C0: 0A2A982B
	v_mul_f32_e32 v22, s43, v80                                // 0000000086C4: 0A2CA02B
	v_mul_f32_e32 v23, s43, v84                                // 0000000086C8: 0A2EA82B
	ds_write_b128 v5, v[20:23]                                 // 0000000086CC: D9BE0000 00001405
	v_mul_f32_e32 v20, s43, v73                                // 0000000086D4: 0A28922B
	v_mul_f32_e32 v21, s43, v77                                // 0000000086D8: 0A2A9A2B
	v_mul_f32_e32 v22, s43, v81                                // 0000000086DC: 0A2CA22B
	v_mul_f32_e32 v23, s43, v85                                // 0000000086E0: 0A2EAA2B
	ds_write_b128 v5, v[20:23] offset:1056                     // 0000000086E4: D9BE0420 00001405
	v_mul_f32_e32 v20, s43, v74                                // 0000000086EC: 0A28942B
	v_mul_f32_e32 v21, s43, v78                                // 0000000086F0: 0A2A9C2B
	v_mul_f32_e32 v22, s43, v82                                // 0000000086F4: 0A2CA42B
	v_mul_f32_e32 v23, s43, v86                                // 0000000086F8: 0A2EAC2B
	ds_write_b128 v5, v[20:23] offset:2112                     // 0000000086FC: D9BE0840 00001405
	v_mul_f32_e32 v20, s43, v75                                // 000000008704: 0A28962B
	v_mul_f32_e32 v21, s43, v79                                // 000000008708: 0A2A9E2B
	v_mul_f32_e32 v22, s43, v83                                // 00000000870C: 0A2CA62B
	v_mul_f32_e32 v23, s43, v87                                // 000000008710: 0A2EAE2B
	ds_write_b128 v5, v[20:23] offset:3168                     // 000000008714: D9BE0C60 00001405
	v_mul_f32_e32 v20, s43, v88                                // 00000000871C: 0A28B02B
	v_mul_f32_e32 v21, s43, v92                                // 000000008720: 0A2AB82B
	v_mul_f32_e32 v22, s43, v96                                // 000000008724: 0A2CC02B
	v_mul_f32_e32 v23, s43, v100                               // 000000008728: 0A2EC82B
	ds_write_b128 v5, v[20:23] offset:4224                     // 00000000872C: D9BE1080 00001405
	v_mul_f32_e32 v20, s43, v89                                // 000000008734: 0A28B22B
	v_mul_f32_e32 v21, s43, v93                                // 000000008738: 0A2ABA2B
	v_mul_f32_e32 v22, s43, v97                                // 00000000873C: 0A2CC22B
	v_mul_f32_e32 v23, s43, v101                               // 000000008740: 0A2ECA2B
	ds_write_b128 v5, v[20:23] offset:5280                     // 000000008744: D9BE14A0 00001405
	v_mul_f32_e32 v20, s43, v90                                // 00000000874C: 0A28B42B
	v_mul_f32_e32 v21, s43, v94                                // 000000008750: 0A2ABC2B
	v_mul_f32_e32 v22, s43, v98                                // 000000008754: 0A2CC42B
	v_mul_f32_e32 v23, s43, v102                               // 000000008758: 0A2ECC2B
	ds_write_b128 v5, v[20:23] offset:6336                     // 00000000875C: D9BE18C0 00001405
	v_mul_f32_e32 v20, s43, v91                                // 000000008764: 0A28B62B
	v_mul_f32_e32 v21, s43, v95                                // 000000008768: 0A2ABE2B
	v_mul_f32_e32 v22, s43, v99                                // 00000000876C: 0A2CC62B
	v_mul_f32_e32 v23, s43, v103                               // 000000008770: 0A2ECE2B
	ds_write_b128 v5, v[20:23] offset:7392                     // 000000008774: D9BE1CE0 00001405
	s_waitcnt lgkmcnt(4)                                       // 00000000877C: BF8CC47F
	ds_read_b128 v[40:43], v4                                  // 000000008780: D9FE0000 28000004
	ds_read_b128 v[44:47], v4 offset:64                        // 000000008788: D9FE0040 2C000004
	ds_read_b128 v[48:51], v4 offset:128                       // 000000008790: D9FE0080 30000004
	ds_read_b128 v[52:55], v4 offset:192                       // 000000008798: D9FE00C0 34000004
	s_waitcnt lgkmcnt(4)                                       // 0000000087A0: BF8CC47F
	ds_read_b128 v[56:59], v4 offset:4224                      // 0000000087A4: D9FE1080 38000004
	ds_read_b128 v[60:63], v4 offset:4288                      // 0000000087AC: D9FE10C0 3C000004
	ds_read_b128 v[64:67], v4 offset:4352                      // 0000000087B4: D9FE1100 40000004
	ds_read_b128 v[68:71], v4 offset:4416                      // 0000000087BC: D9FE1140 44000004
	s_waitcnt lgkmcnt(0)                                       // 0000000087C4: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 0000000087C8: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:512// 0000000087CC: E07C1200 80022812
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:768// 0000000087D4: E07C1300 80023812
	v_add_u32_e32 v18, 0x2000, v18                             // 0000000087DC: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:512// 0000000087E4: E07C1200 80022C12
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:768// 0000000087EC: E07C1300 80023C12
	v_add_u32_e32 v18, 0x2000, v18                             // 0000000087F4: 682424FF 00002000
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:512// 0000000087FC: E07C1200 80023012
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:768// 000000008804: E07C1300 80024012
	v_add_u32_e32 v18, 0x2000, v18                             // 00000000880C: 682424FF 00002000
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:512// 000000008814: E07C1200 80023412
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:768// 00000000881C: E07C1300 80024412
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008824: 682424FF 00002000
	v_mul_f32_e32 v20, s43, v104                               // 00000000882C: 0A28D02B
	v_mul_f32_e32 v21, s43, v108                               // 000000008830: 0A2AD82B
	v_mul_f32_e32 v22, s43, v112                               // 000000008834: 0A2CE02B
	v_mul_f32_e32 v23, s43, v116                               // 000000008838: 0A2EE82B
	ds_write_b128 v5, v[20:23]                                 // 00000000883C: D9BE0000 00001405
	v_mul_f32_e32 v20, s43, v105                               // 000000008844: 0A28D22B
	v_mul_f32_e32 v21, s43, v109                               // 000000008848: 0A2ADA2B
	v_mul_f32_e32 v22, s43, v113                               // 00000000884C: 0A2CE22B
	v_mul_f32_e32 v23, s43, v117                               // 000000008850: 0A2EEA2B
	ds_write_b128 v5, v[20:23] offset:1056                     // 000000008854: D9BE0420 00001405
	v_mul_f32_e32 v20, s43, v106                               // 00000000885C: 0A28D42B
	v_mul_f32_e32 v21, s43, v110                               // 000000008860: 0A2ADC2B
	v_mul_f32_e32 v22, s43, v114                               // 000000008864: 0A2CE42B
	v_mul_f32_e32 v23, s43, v118                               // 000000008868: 0A2EEC2B
	ds_write_b128 v5, v[20:23] offset:2112                     // 00000000886C: D9BE0840 00001405
	v_mul_f32_e32 v20, s43, v107                               // 000000008874: 0A28D62B
	v_mul_f32_e32 v21, s43, v111                               // 000000008878: 0A2ADE2B
	v_mul_f32_e32 v22, s43, v115                               // 00000000887C: 0A2CE62B
	v_mul_f32_e32 v23, s43, v119                               // 000000008880: 0A2EEE2B
	ds_write_b128 v5, v[20:23] offset:3168                     // 000000008884: D9BE0C60 00001405
	v_mul_f32_e32 v20, s43, v120                               // 00000000888C: 0A28F02B
	v_mul_f32_e32 v21, s43, v124                               // 000000008890: 0A2AF82B
	v_mul_f32_e32 v22, s43, v128                               // 000000008894: 0A2D002B
	v_mul_f32_e32 v23, s43, v132                               // 000000008898: 0A2F082B
	ds_write_b128 v5, v[20:23] offset:4224                     // 00000000889C: D9BE1080 00001405
	v_mul_f32_e32 v20, s43, v121                               // 0000000088A4: 0A28F22B
	v_mul_f32_e32 v21, s43, v125                               // 0000000088A8: 0A2AFA2B
	v_mul_f32_e32 v22, s43, v129                               // 0000000088AC: 0A2D022B
	v_mul_f32_e32 v23, s43, v133                               // 0000000088B0: 0A2F0A2B
	ds_write_b128 v5, v[20:23] offset:5280                     // 0000000088B4: D9BE14A0 00001405
	v_mul_f32_e32 v20, s43, v122                               // 0000000088BC: 0A28F42B
	v_mul_f32_e32 v21, s43, v126                               // 0000000088C0: 0A2AFC2B
	v_mul_f32_e32 v22, s43, v130                               // 0000000088C4: 0A2D042B
	v_mul_f32_e32 v23, s43, v134                               // 0000000088C8: 0A2F0C2B
	ds_write_b128 v5, v[20:23] offset:6336                     // 0000000088CC: D9BE18C0 00001405
	v_mul_f32_e32 v20, s43, v123                               // 0000000088D4: 0A28F62B
	v_mul_f32_e32 v21, s43, v127                               // 0000000088D8: 0A2AFE2B
	v_mul_f32_e32 v22, s43, v131                               // 0000000088DC: 0A2D062B
	v_mul_f32_e32 v23, s43, v135                               // 0000000088E0: 0A2F0E2B
	ds_write_b128 v5, v[20:23] offset:7392                     // 0000000088E4: D9BE1CE0 00001405
	s_waitcnt lgkmcnt(4)                                       // 0000000088EC: BF8CC47F
	ds_read_b128 v[40:43], v4                                  // 0000000088F0: D9FE0000 28000004
	ds_read_b128 v[44:47], v4 offset:64                        // 0000000088F8: D9FE0040 2C000004
	ds_read_b128 v[48:51], v4 offset:128                       // 000000008900: D9FE0080 30000004
	ds_read_b128 v[52:55], v4 offset:192                       // 000000008908: D9FE00C0 34000004
	s_waitcnt lgkmcnt(4)                                       // 000000008910: BF8CC47F
	ds_read_b128 v[56:59], v4 offset:4224                      // 000000008914: D9FE1080 38000004
	ds_read_b128 v[60:63], v4 offset:4288                      // 00000000891C: D9FE10C0 3C000004
	ds_read_b128 v[64:67], v4 offset:4352                      // 000000008924: D9FE1100 40000004
	ds_read_b128 v[68:71], v4 offset:4416                      // 00000000892C: D9FE1140 44000004
	s_waitcnt lgkmcnt(0)                                       // 000000008934: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 000000008938: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:1024// 00000000893C: E07C1400 80022812
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:1280// 000000008944: E07C1500 80023812
	v_add_u32_e32 v18, 0x2000, v18                             // 00000000894C: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:1024// 000000008954: E07C1400 80022C12
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:1280// 00000000895C: E07C1500 80023C12
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008964: 682424FF 00002000
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:1024// 00000000896C: E07C1400 80023012
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:1280// 000000008974: E07C1500 80024012
	v_add_u32_e32 v18, 0x2000, v18                             // 00000000897C: 682424FF 00002000
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:1024// 000000008984: E07C1400 80023412
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:1280// 00000000898C: E07C1500 80024412
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008994: 682424FF 00002000
	v_mul_f32_e32 v20, s43, v136                               // 00000000899C: 0A29102B
	v_mul_f32_e32 v21, s43, v140                               // 0000000089A0: 0A2B182B
	v_mul_f32_e32 v22, s43, v144                               // 0000000089A4: 0A2D202B
	v_mul_f32_e32 v23, s43, v148                               // 0000000089A8: 0A2F282B
	ds_write_b128 v5, v[20:23]                                 // 0000000089AC: D9BE0000 00001405
	v_mul_f32_e32 v20, s43, v137                               // 0000000089B4: 0A29122B
	v_mul_f32_e32 v21, s43, v141                               // 0000000089B8: 0A2B1A2B
	v_mul_f32_e32 v22, s43, v145                               // 0000000089BC: 0A2D222B
	v_mul_f32_e32 v23, s43, v149                               // 0000000089C0: 0A2F2A2B
	ds_write_b128 v5, v[20:23] offset:1056                     // 0000000089C4: D9BE0420 00001405
	v_mul_f32_e32 v20, s43, v138                               // 0000000089CC: 0A29142B
	v_mul_f32_e32 v21, s43, v142                               // 0000000089D0: 0A2B1C2B
	v_mul_f32_e32 v22, s43, v146                               // 0000000089D4: 0A2D242B
	v_mul_f32_e32 v23, s43, v150                               // 0000000089D8: 0A2F2C2B
	ds_write_b128 v5, v[20:23] offset:2112                     // 0000000089DC: D9BE0840 00001405
	v_mul_f32_e32 v20, s43, v139                               // 0000000089E4: 0A29162B
	v_mul_f32_e32 v21, s43, v143                               // 0000000089E8: 0A2B1E2B
	v_mul_f32_e32 v22, s43, v147                               // 0000000089EC: 0A2D262B
	v_mul_f32_e32 v23, s43, v151                               // 0000000089F0: 0A2F2E2B
	ds_write_b128 v5, v[20:23] offset:3168                     // 0000000089F4: D9BE0C60 00001405
	v_mul_f32_e32 v20, s43, v152                               // 0000000089FC: 0A29302B
	v_mul_f32_e32 v21, s43, v156                               // 000000008A00: 0A2B382B
	v_mul_f32_e32 v22, s43, v160                               // 000000008A04: 0A2D402B
	v_mul_f32_e32 v23, s43, v164                               // 000000008A08: 0A2F482B
	ds_write_b128 v5, v[20:23] offset:4224                     // 000000008A0C: D9BE1080 00001405
	v_mul_f32_e32 v20, s43, v153                               // 000000008A14: 0A29322B
	v_mul_f32_e32 v21, s43, v157                               // 000000008A18: 0A2B3A2B
	v_mul_f32_e32 v22, s43, v161                               // 000000008A1C: 0A2D422B
	v_mul_f32_e32 v23, s43, v165                               // 000000008A20: 0A2F4A2B
	ds_write_b128 v5, v[20:23] offset:5280                     // 000000008A24: D9BE14A0 00001405
	v_mul_f32_e32 v20, s43, v154                               // 000000008A2C: 0A29342B
	v_mul_f32_e32 v21, s43, v158                               // 000000008A30: 0A2B3C2B
	v_mul_f32_e32 v22, s43, v162                               // 000000008A34: 0A2D442B
	v_mul_f32_e32 v23, s43, v166                               // 000000008A38: 0A2F4C2B
	ds_write_b128 v5, v[20:23] offset:6336                     // 000000008A3C: D9BE18C0 00001405
	v_mul_f32_e32 v20, s43, v155                               // 000000008A44: 0A29362B
	v_mul_f32_e32 v21, s43, v159                               // 000000008A48: 0A2B3E2B
	v_mul_f32_e32 v22, s43, v163                               // 000000008A4C: 0A2D462B
	v_mul_f32_e32 v23, s43, v167                               // 000000008A50: 0A2F4E2B
	ds_write_b128 v5, v[20:23] offset:7392                     // 000000008A54: D9BE1CE0 00001405
	s_waitcnt lgkmcnt(4)                                       // 000000008A5C: BF8CC47F
	ds_read_b128 v[40:43], v4                                  // 000000008A60: D9FE0000 28000004
	ds_read_b128 v[44:47], v4 offset:64                        // 000000008A68: D9FE0040 2C000004
	ds_read_b128 v[48:51], v4 offset:128                       // 000000008A70: D9FE0080 30000004
	ds_read_b128 v[52:55], v4 offset:192                       // 000000008A78: D9FE00C0 34000004
	s_waitcnt lgkmcnt(4)                                       // 000000008A80: BF8CC47F
	ds_read_b128 v[56:59], v4 offset:4224                      // 000000008A84: D9FE1080 38000004
	ds_read_b128 v[60:63], v4 offset:4288                      // 000000008A8C: D9FE10C0 3C000004
	ds_read_b128 v[64:67], v4 offset:4352                      // 000000008A94: D9FE1100 40000004
	ds_read_b128 v[68:71], v4 offset:4416                      // 000000008A9C: D9FE1140 44000004
	s_waitcnt lgkmcnt(0)                                       // 000000008AA4: BF8CC07F
	v_mov_b32_e32 v18, v19                                     // 000000008AA8: 7E240313
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:1536// 000000008AAC: E07C1600 80022812
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:1792// 000000008AB4: E07C1700 80023812
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008ABC: 682424FF 00002000
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:1536// 000000008AC4: E07C1600 80022C12
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:1792// 000000008ACC: E07C1700 80023C12
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008AD4: 682424FF 00002000
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:1536// 000000008ADC: E07C1600 80023012
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:1792// 000000008AE4: E07C1700 80024012
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008AEC: 682424FF 00002000
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:1536// 000000008AF4: E07C1600 80023412
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:1792// 000000008AFC: E07C1700 80024412
	v_add_u32_e32 v18, 0x2000, v18                             // 000000008B04: 682424FF 00002000
	v_mov_b32_e32 v20, v24                                     // 000000008B0C: 7E280318
	buffer_store_dword v24, v26, s[12:15], 0 offen             // 000000008B10: E0701000 8003181A
	s_mul_i32 s56, 4, s77                                      // 000000008B18: 92384D84
	v_add_u32_e64 v26, v26, s56                                // 000000008B1C: D134001A 0000711A

0000000000008b24 <label_1A09>:
	s_mov_b32 s56, 32                                          // 000000008B24: BEB800A0
	s_addk_i32 s85, 0x1                                        // 000000008B28: B7550001
	s_cmp_lt_i32 s85, s86                                      // 000000008B2C: BF045655
	s_cbranch_scc1 label_0029                                  // 000000008B30: BF85E61C

0000000000008b34 <label_1A0D>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000008B34: BF8C0000
	s_endpgm                                                   // 000000008B38: BF810000
