
mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co:	file format elf64-amdgpu

Disassembly of section .text:

0000000000002300 <_ZN5aiter36mla_a8w8_qh16_qseqlen1_gqaratio16_psE>:
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
	s_mul_i32 s60, s2, 4                                       // 000000002368: 923C8402
	s_and_b32 s33, s33, 0xffff                                 // 00000000236C: 8621FF21 0000FFFF
	s_add_u32 s32, s60, s32                                    // 000000002374: 8020203C
	s_addc_u32 s33, 0, s33                                     // 000000002378: 82212180
	s_load_dword s89, s[32:33], 0x0                            // 00000000237C: C0021650 00000000
	s_load_dword s90, s[32:33], 0x4                            // 000000002384: C0021690 00000004
	s_and_b32 s31, s31, 0xffff                                 // 00000000238C: 861FFF1F 0000FFFF
	s_waitcnt lgkmcnt(0)                                       // 000000002394: BF8CC07F
	s_cmp_eq_i32 s89, s90                                      // 000000002398: BF005A59
	s_cbranch_scc1 label_34F0                                  // 00000000239C: BF850CF2
	s_mul_i32 s60, s89, 32                                     // 0000000023A0: 923CA059

00000000000023a4 <label_00A4>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 0000000023A4: BF8C0000
	s_barrier                                                  // 0000000023A8: BF8A0000
	s_add_u32 s30, s60, s30                                    // 0000000023AC: 801E1E3C
	s_addc_u32 s31, 0, s31                                     // 0000000023B0: 821F1F80
	s_load_dword s91, s[30:31], 0x4                            // 0000000023B4: C00216CF 00000004
	s_load_dword s82, s[30:31], 0x8                            // 0000000023BC: C002148F 00000008
	s_load_dword s83, s[30:31], 0xc                            // 0000000023C4: C00214CF 0000000C
	s_load_dword s47, s[30:31], 0x10                           // 0000000023CC: C0020BCF 00000010
	s_load_dword s46, s[30:31], 0x14                           // 0000000023D4: C0020B8F 00000014
	s_load_dword s81, s[30:31], 0x18                           // 0000000023DC: C002144F 00000018
	s_load_dwordx2 s[8:9], s[0:1], 0x0                         // 0000000023E4: C0060200 00000000
	s_load_dwordx2 s[12:13], s[0:1], 0x10                      // 0000000023EC: C0060300 00000010
	s_load_dwordx2 s[16:17], s[0:1], 0x20                      // 0000000023F4: C0060400 00000020
	s_load_dwordx2 s[20:21], s[0:1], 0x30                      // 0000000023FC: C0060500 00000030
	s_load_dwordx2 s[24:25], s[0:1], 0x50                      // 000000002404: C0060600 00000050
	s_load_dword s68, s[0:1], 0x70                             // 00000000240C: C0021100 00000070
	s_load_dword s69, s[0:1], 0x80                             // 000000002414: C0021140 00000080
	s_load_dword s71, s[0:1], 0x90                             // 00000000241C: C00211C0 00000090
	s_load_dword s70, s[0:1], 0xa0                             // 000000002424: C0021180 000000A0
	s_load_dword s72, s[0:1], 0xb0                             // 00000000242C: C0021200 000000B0
	s_load_dword s73, s[0:1], 0xc0                             // 000000002434: C0021240 000000C0
	s_load_dwordx2 s[92:93], s[0:1], 0xf0                      // 00000000243C: C0061700 000000F0
	s_load_dwordx2 s[40:41], s[0:1], 0x100                     // 000000002444: C0060A00 00000100
	s_load_dwordx2 s[42:43], s[0:1], 0x110                     // 00000000244C: C0060A80 00000110
	s_waitcnt lgkmcnt(0)                                       // 000000002454: BF8CC07F
	s_min_u32 s80, 16, s69                                     // 000000002458: 83D04590
	s_sub_u32 s85, s83, s82                                    // 00000000245C: 80D55253
	s_mul_i32 s78, 0x240, s69                                  // 000000002460: 924E45FF 00000240
	s_mul_i32 s60, 4, s69                                      // 000000002468: 923C4584
	s_mov_b32 s10, s79                                         // 00000000246C: BE8A004F
	s_mov_b32 s18, s78                                         // 000000002470: BE92004E
	s_mov_b32 s14, s60                                         // 000000002474: BE8E003C
	s_mov_b32 s22, -16                                         // 000000002478: BE9600D0
	s_mov_b32 s26, -16                                         // 00000000247C: BE9A00D0
	s_mov_b32 s11, 0x20000                                     // 000000002480: BE8B00FF 00020000
	s_mov_b32 s95, 0x20000                                     // 000000002488: BEDF00FF 00020000
	s_mov_b32 s19, 0x20000                                     // 000000002490: BE9300FF 00020000
	s_mov_b32 s15, 0x20000                                     // 000000002498: BE8F00FF 00020000
	s_mov_b32 s23, 0x20000                                     // 0000000024A0: BE9700FF 00020000
	s_mov_b32 s27, 0x20000                                     // 0000000024A8: BE9B00FF 00020000
	s_and_b32 s9, s9, 0xffff                                   // 0000000024B0: 8609FF09 0000FFFF
	s_and_b32 s93, s93, 0xffff                                 // 0000000024B8: 865DFF5D 0000FFFF
	s_and_b32 s17, s17, 0xffff                                 // 0000000024C0: 8611FF11 0000FFFF
	s_and_b32 s13, s13, 0xffff                                 // 0000000024C8: 860DFF0D 0000FFFF
	s_and_b32 s21, s21, 0xffff                                 // 0000000024D0: 8615FF15 0000FFFF
	s_and_b32 s25, s25, 0xffff                                 // 0000000024D8: 8619FF19 0000FFFF
	s_and_b32 s41, s41, 0xffff                                 // 0000000024E0: 8629FF29 0000FFFF
	s_and_b32 s43, s43, 0xffff                                 // 0000000024E8: 862BFF2B 0000FFFF
	s_or_b32 s9, s9, 0x40000                                   // 0000000024F0: 8709FF09 00040000
	s_or_b32 s93, s93, 0x40000                                 // 0000000024F8: 875DFF5D 00040000
	s_or_b32 s17, s17, 0x40000                                 // 000000002500: 8711FF11 00040000
	s_or_b32 s13, s13, 0x40000                                 // 000000002508: 870DFF0D 00040000
	s_or_b32 s21, s21, 0x40000                                 // 000000002510: 8715FF15 00040000
	s_or_b32 s25, s25, 0x40000                                 // 000000002518: 8719FF19 00040000
	s_mov_b32 s85, 1                                           // 000000002520: BED50081
	s_mov_b32 s71, 1                                           // 000000002524: BEC70081
	s_mov_b32 s84, 0                                           // 000000002528: BED40080
	s_waitcnt lgkmcnt(0)                                       // 00000000252C: BF8CC07F
	s_load_dword s64, s[40:41], 0x0                            // 000000002530: C0021014 00000000
	s_load_dword s65, s[42:43], 0x0                            // 000000002538: C0021055 00000000
	s_mov_b32 s73, 0                                           // 000000002540: BEC90080
	s_lshr_b32 s44, 0x80, s73                                  // 000000002544: 8F2C49FF 00000080
	s_mul_i32 s77, s44, 4                                      // 00000000254C: 924D842C
	s_mul_i32 s77, s77, s71                                    // 000000002550: 924D474D
	s_mul_i32 s45, s4, s44                                     // 000000002554: 922D2C04
	s_sub_u32 s50, s46, s47                                    // 000000002558: 80B22F2E
	s_cmp_le_u32 s50, s45                                      // 00000000255C: BF0B2D32
	s_cbranch_scc1 label_34F0                                  // 000000002560: BF850C81
	s_mul_i32 s60, s50, 4                                      // 000000002564: 923C8432
	s_mov_b32 s26, s60                                         // 000000002568: BE9A003C
	s_mul_i32 s60, s47, 4                                      // 00000000256C: 923C842F
	s_add_u32 s24, s60, s24                                    // 000000002570: 8018183C
	s_addc_u32 s25, 0, s25                                     // 000000002574: 82191980
	s_mov_b32 s74, 0                                           // 000000002578: BECA0080
	s_sub_u32 s75, s50, s45                                    // 00000000257C: 80CB2D32
	s_mul_i32 s37, s71, s44                                    // 000000002580: 92252C47
	s_mov_b32 s36, s75                                         // 000000002584: BEA4004B
	v_cvt_f32_u32_e32 v28, s37                                 // 000000002588: 7E380C25
	s_sub_i32 s60, 0, s37                                      // 00000000258C: 81BC2580
	v_rcp_iflag_f32_e32 v28, v28                               // 000000002590: 7E38471C
	s_nop 0                                                    // 000000002594: BF800000
	v_mul_f32_e32 v28, 0x4f7ffffe, v28                         // 000000002598: 0A3838FF 4F7FFFFE
	v_cvt_u32_f32_e32 v28, v28                                 // 0000000025A0: 7E380F1C
	v_mul_lo_u32 v29, s60, v28                                 // 0000000025A4: D285001D 0002383C
	v_mul_hi_u32 v29, v28, v29                                 // 0000000025AC: D286001D 00023B1C
	v_add_u32_e32 v28, v28, v29                                // 0000000025B4: 68383B1C
	v_mul_hi_u32 v28, s36, v28                                 // 0000000025B8: D286001C 00023824
	v_mul_lo_u32 v29, v28, s37                                 // 0000000025C0: D285001D 00004B1C
	v_sub_u32_e32 v31, s36, v29                                // 0000000025C8: 6A3E3A24
	v_add_u32_e32 v30, 1, v28                                  // 0000000025CC: 683C3881
	v_cmp_le_u32_e32 vcc, s37, v31                             // 0000000025D0: 7D963E25
	v_subrev_u32_e32 v29, s37, v31                             // 0000000025D4: 6C3A3E25
	s_nop 0                                                    // 0000000025D8: BF800000
	v_cndmask_b32_e32 v28, v28, v30, vcc                       // 0000000025DC: 00383D1C
	v_cndmask_b32_e32 v31, v31, v29, vcc                       // 0000000025E0: 003E3B1F
	v_add_u32_e32 v29, 1, v28                                  // 0000000025E4: 683A3881
	v_cmp_le_u32_e32 vcc, s37, v31                             // 0000000025E8: 7D963E25
	s_nop 1                                                    // 0000000025EC: BF800001
	v_cndmask_b32_e32 v31, v28, v29, vcc                       // 0000000025F0: 003E3B1C
	s_nop 3                                                    // 0000000025F4: BF800003
	v_readfirstlane_b32 s38, v31                               // 0000000025F8: 7E4C051F
	s_nop 3                                                    // 0000000025FC: BF800003
	s_mov_b32 s75, s38                                         // 000000002600: BECB0026
	s_mul_i32 s60, s75, s37                                    // 000000002604: 923C254B
	s_sub_u32 s60, s36, s60                                    // 000000002608: 80BC3C24
	s_mov_b32 s61, 0                                           // 00000000260C: BEBD0080
	s_cmp_lt_u32 s60, s44                                      // 000000002610: BF0A2C3C
	s_cselect_b32 s61, s61, 1                                  // 000000002614: 853D813D
	s_add_u32 s75, s61, s75                                    // 000000002618: 804B4B3D
	s_cmpk_eq_u32 s61, 0x1                                     // 00000000261C: B43D0001
	s_cselect_b32 s49, 0, s60                                  // 000000002620: 85313C80
	s_mov_b32 s48, s49                                         // 000000002624: BEB00031
	v_lshrrev_b32_e32 v28, 2, v0                               // 000000002628: 20380082
	s_mul_i32 s60, s7, 32                                      // 00000000262C: 923CA007
	v_add_u32_e64 v26, v28, s60                                // 000000002630: D134001A 0000791C
	s_mov_b32 s60, 16                                          // 000000002638: BEBC0090
	v_add_u32_e32 v27, s60, v26                                // 00000000263C: 6836343C
	v_lshlrev_b32_e32 v26, 2, v26                              // 000000002640: 24343482
	v_lshlrev_b32_e32 v27, 2, v27                              // 000000002644: 24363682
	buffer_load_dword v22, v26, s[24:27], 0 offen              // 000000002648: E0501000 8006161A
	buffer_load_dword v23, v27, s[24:27], 0 offen              // 000000002650: E0501000 8006171B
	v_add_u32_e32 v26, s77, v26                                // 000000002658: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 00000000265C: 6836364D
	buffer_load_dword v24, v26, s[24:27], 0 offen              // 000000002660: E0501000 8006181A
	buffer_load_dword v25, v27, s[24:27], 0 offen              // 000000002668: E0501000 8006191B
	v_add_u32_e32 v26, s77, v26                                // 000000002670: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 000000002674: 6836364D
	s_mul_i32 s60, 0x240, 16                                   // 000000002678: 923C90FF 00000240
	s_mul_i32 s60, s82, s60                                    // 000000002680: 923C3C52
	s_add_u32 s16, s60, s16                                    // 000000002684: 8010103C
	s_addc_u32 s17, 0, s17                                     // 000000002688: 82111180
	s_mul_i32 s60, s7, 0x400                                   // 00000000268C: 923CFF07 00000400
	s_add_u32 m0, 0, s60                                       // 000000002694: 807C3C80
	v_and_b32_e32 v28, 3, v0                                   // 000000002698: 26380083
	v_mov_b32_e32 v29, 0                                       // 00000000269C: 7E3A0280
	s_mov_b32 s60, 0                                           // 0000000026A0: BEBC0080
	s_mov_b32 s61, -1                                          // 0000000026A4: BEBD00C1
	v_cndmask_b32_e64 v30, v29, v28, s[60:61]                  // 0000000026A8: D100001E 00F2391D
	s_nop 2                                                    // 0000000026B0: BF800002
	v_mov_b32_dpp v30, v30 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 0000000026B4: 7E3C02FA FF004E1E
	v_cndmask_b32_e64 v31, v28, v30, s[60:61]                  // 0000000026BC: D100001F 00F23D1C
	v_lshlrev_b32_e32 v31, 4, v31                              // 0000000026C4: 243E3E84
	v_mov_b32_e32 v1, v31                                      // 0000000026C8: 7E02031F
	v_lshrrev_b32_e32 v28, 2, v0                               // 0000000026CC: 20380082
	s_mov_b32 s60, 0x240                                       // 0000000026D0: BEBC00FF 00000240
	v_mul_i32_i24_e64 v28, v28, s60                            // 0000000026D8: D106001C 0000791C
	s_mul_i32 s60, s7, 64                                      // 0000000026E0: 923CC007
	v_add_u32_e32 v29, v28, v31                                // 0000000026E4: 683A3F1C
	v_add_u32_e64 v29, v29, s60                                // 0000000026E8: D134001D 0000791D
	buffer_load_dwordx4 v29, s[16:19], 0 offen lds             // 0000000026F0: E05D1000 8004001D
	s_add_u32 m0, m0, 0x1000                                   // 0000000026F8: 807CFF7C 00001000
	v_add_u32_e32 v29, 0x100, v29                              // 000000002700: 683A3AFF 00000100
	buffer_load_dwordx4 v29, s[16:19], 0 offen lds             // 000000002708: E05D1000 8004001D
	s_add_u32 m0, m0, 0x1000                                   // 000000002710: 807CFF7C 00001000
	v_add_u32_e32 v29, 0x100, v29                              // 000000002718: 683A3AFF 00000100
	buffer_load_dwordx4 v29, s[16:19], 0 offen lds             // 000000002720: E05D1000 8004001D
	s_add_u32 m0, m0, 0x1000                                   // 000000002728: 807CFF7C 00001000
	v_add_u32_e32 v29, 0x100, v29                              // 000000002730: 683A3AFF 00000100
	s_mov_b32 s52, 0x7060302                                   // 000000002738: BEB400FF 07060302
	s_mov_b32 s53, 0x7060302                                   // 000000002740: BEB500FF 07060302
	s_mov_b32 s54, 0x5040100                                   // 000000002748: BEB600FF 05040100
	v_mov_b32_e32 v71, 0xffff0000                              // 000000002750: 7E8E02FF FFFF0000
	v_mov_b32_e32 v72, 0x7fff0000                              // 000000002758: 7E9002FF 7FFF0000
	v_mov_b32_e32 v73, 0x7fff                                  // 000000002760: 7E9202FF 00007FFF
	s_mul_i32 s51, s7, 4                                       // 000000002768: 92338407
	s_mov_b32 s6, 0x3fb8aa3b                                   // 00000000276C: BE8600FF 3FB8AA3B
	v_mov_b32_e32 v29, s6                                      // 000000002774: 7E3A0206
	v_mov_b32_e32 v28, s68                                     // 000000002778: 7E380244
	v_mul_f32_e32 v28, s6, v28                                 // 00000000277C: 0A383806
	v_rcp_f32_e32 v29, v29                                     // 000000002780: 7E3A451D
	v_mov_b32_e32 v2, 0xff800000                               // 000000002784: 7E0402FF FF800000
	v_mov_b32_e32 v18, 0                                       // 00000000278C: 7E240280
	v_mov_b32_e32 v4, 0                                        // 000000002790: 7E080280
	v_readfirstlane_b32 s100, v28                              // 000000002794: 7EC8051C
	v_readfirstlane_b32 s67, v29                               // 000000002798: 7E86051D
	s_waitcnt lgkmcnt(0)                                       // 00000000279C: BF8CC07F
	v_mov_b32_e32 v28, s64                                     // 0000000027A0: 7E380240
	v_mul_f32_e32 v28, s65, v28                                // 0000000027A4: 0A383841
	v_mul_f32_e32 v29, s100, v28                               // 0000000027A8: 0A3A3864
	v_mul_f32_e32 v31, s68, v28                                // 0000000027AC: 0A3E3844
	v_readfirstlane_b32 s100, v29                              // 0000000027B0: 7EC8051D
	v_readfirstlane_b32 s68, v31                               // 0000000027B4: 7E88051F
	v_accvgpr_write_b32 a36, 0                                 // 0000000027B8: D3D94024 18000080
	v_accvgpr_write_b32 a37, 0                                 // 0000000027C0: D3D94025 18000080
	v_accvgpr_write_b32 a38, 0                                 // 0000000027C8: D3D94026 18000080
	v_accvgpr_write_b32 a39, 0                                 // 0000000027D0: D3D94027 18000080
	v_accvgpr_write_b32 a76, 0                                 // 0000000027D8: D3D9404C 18000080
	v_accvgpr_write_b32 a77, 0                                 // 0000000027E0: D3D9404D 18000080
	v_accvgpr_write_b32 a78, 0                                 // 0000000027E8: D3D9404E 18000080
	v_accvgpr_write_b32 a79, 0                                 // 0000000027F0: D3D9404F 18000080
	v_accvgpr_write_b32 a116, 0                                // 0000000027F8: D3D94074 18000080
	v_accvgpr_write_b32 a117, 0                                // 000000002800: D3D94075 18000080
	v_accvgpr_write_b32 a118, 0                                // 000000002808: D3D94076 18000080
	v_accvgpr_write_b32 a119, 0                                // 000000002810: D3D94077 18000080
	s_mov_b32 s56, 0x4000                                      // 000000002818: BEB800FF 00004000
	s_mov_b32 s57, 0x6400                                      // 000000002820: BEB900FF 00006400
	s_mov_b32 s58, 0x16000                                     // 000000002828: BEBA00FF 00016000
	s_mov_b32 s59, 0x18400                                     // 000000002830: BEBB00FF 00018400
	s_mul_i32 s61, s7, 0x4800                                  // 000000002838: 923DFF07 00004800
	s_add_u32 s56, s61, s56                                    // 000000002840: 8038383D
	s_add_u32 s57, s61, s57                                    // 000000002844: 8039393D
	s_add_u32 s58, s61, s58                                    // 000000002848: 803A3A3D
	s_add_u32 s59, s61, s59                                    // 00000000284C: 803B3B3D
	v_lshlrev_b32_e32 v34, 3, v0                               // 000000002850: 24440083
	s_mov_b32 s60, 0x200                                       // 000000002854: BEBC00FF 00000200
	s_mul_i32 s60, s60, s7                                     // 00000000285C: 923C073C
	v_add_u32_e32 v34, s60, v34                                // 000000002860: 6844443C
	v_and_b32_e32 v28, 31, v0                                  // 000000002864: 2638009F
	v_lshlrev_b32_e32 v35, 3, v28                              // 000000002868: 24463883
	v_lshrrev_b32_e32 v28, 5, v0                               // 00000000286C: 20380085
	s_mov_b32 s60, 0x200                                       // 000000002870: BEBC00FF 00000200
	v_mul_i32_i24_e32 v28, s60, v28                            // 000000002878: 0C38383C
	v_add_u32_e32 v35, v28, v35                                // 00000000287C: 6846471C
	v_lshlrev_b32_e32 v36, 2, v0                               // 000000002880: 24480082
	s_mul_i32 s60, 0x100, s7                                   // 000000002884: 923C07FF 00000100
	v_add_u32_e32 v36, s60, v36                                // 00000000288C: 6848483C
	v_lshlrev_b32_e32 v37, 2, v0                               // 000000002890: 244A0082
	s_waitcnt vmcnt(3)                                         // 000000002894: BF8C0F73
	v_mul_u32_u24_e64 v32, v22, s72                            // 000000002898: D1080020 00009116
	v_mul_u32_u24_e64 v33, v23, s72                            // 0000000028A0: D1080021 00009117
	v_add_u32_e32 v32, v32, v1                                 // 0000000028A8: 68400320
	v_add_u32_e32 v33, v33, v1                                 // 0000000028AC: 68420321
	s_mov_b32 m0, s58                                          // 0000000028B0: BEFC003A
	buffer_load_dwordx4 v32, s[20:23], 0 offen lds             // 0000000028B4: E05D1000 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000028BC: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:64 lds   // 0000000028C4: E05D1040 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000028CC: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:128 lds  // 0000000028D4: E05D1080 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000028DC: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:192 lds  // 0000000028E4: E05D10C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000028EC: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:256 lds  // 0000000028F4: E05D1100 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000028FC: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:320 lds  // 000000002904: E05D1140 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 00000000290C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:384 lds  // 000000002914: E05D1180 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 00000000291C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:448 lds  // 000000002924: E05D11C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 00000000292C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:512 lds  // 000000002934: E05D1200 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 00000000293C: 817CFF7C 000003C0
	s_mov_b32 m0, s59                                          // 000000002944: BEFC003B
	buffer_load_dwordx4 v33, s[20:23], 0 offen lds             // 000000002948: E05D1000 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002950: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:64 lds   // 000000002958: E05D1040 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002960: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:128 lds  // 000000002968: E05D1080 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002970: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:192 lds  // 000000002978: E05D10C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002980: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:256 lds  // 000000002988: E05D1100 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002990: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:320 lds  // 000000002998: E05D1140 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000029A0: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:384 lds  // 0000000029A8: E05D1180 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000029B0: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:448 lds  // 0000000029B8: E05D11C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000029C0: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:512 lds  // 0000000029C8: E05D1200 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000029D0: 817CFF7C 000003C0
	buffer_load_dword v22, v26, s[24:27], 0 offen              // 0000000029D8: E0501000 8006161A
	buffer_load_dword v23, v27, s[24:27], 0 offen              // 0000000029E0: E0501000 8006171B
	v_add_u32_e32 v26, s77, v26                                // 0000000029E8: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 0000000029EC: 6836364D
	v_lshrrev_b32_e32 v28, 4, v0                               // 0000000029F0: 20380084
	v_lshlrev_b32_e32 v28, 2, v28                              // 0000000029F4: 24383882
	v_sub_u32_e32 v29, 12, v28                                 // 0000000029F8: 6A3A388C
	v_mov_b32_e32 v30, v29                                     // 0000000029FC: 7E3C031D
	s_nop 1                                                    // 000000002A00: BF800001
	v_permlane16_swap_b32_e32 v29, v30                         // 000000002A04: 7E3AB31E
	s_nop 1                                                    // 000000002A08: BF800001
	v_permlane16_swap_b32_e32 v30, v29                         // 000000002A0C: 7E3CB31D
	s_mov_b32 s60, 0xff00ff00                                  // 000000002A10: BEBC00FF FF00FF00
	s_mov_b32 s61, 0xff00ff00                                  // 000000002A18: BEBD00FF FF00FF00
	v_cndmask_b32_e64 v30, v28, v29, s[60:61]                  // 000000002A20: D100001E 00F23B1C
	v_and_b32_e32 v28, 15, v0                                  // 000000002A28: 2638008F
	v_lshlrev_b32_e32 v28, 4, v28                              // 000000002A2C: 24383884
	v_add_u32_e32 v4, v28, v30                                 // 000000002A30: 68083D1C
	v_lshlrev_b32_e32 v4, 2, v4                                // 000000002A34: 24080882
	s_waitcnt vmcnt(20)                                        // 000000002A38: BF8C4F74
	s_barrier                                                  // 000000002A3C: BF8A0000
	ds_read_b128 a[0:3], v4                                    // 000000002A40: DBFE0000 00000004
	ds_read_b128 a[4:7], v4 offset:1024                        // 000000002A48: DBFE0400 04000004
	ds_read_b128 a[8:11], v4 offset:2048                       // 000000002A50: DBFE0800 08000004
	ds_read_b128 a[12:15], v4 offset:3072                      // 000000002A58: DBFE0C00 0C000004
	ds_read_b128 a[16:19], v4 offset:4096                      // 000000002A60: DBFE1000 10000004
	ds_read_b128 a[20:23], v4 offset:5120                      // 000000002A68: DBFE1400 14000004
	ds_read_b128 a[24:27], v4 offset:6144                      // 000000002A70: DBFE1800 18000004
	ds_read_b128 a[28:31], v4 offset:7168                      // 000000002A78: DBFE1C00 1C000004
	ds_read_b128 a[32:35], v4 offset:8192                      // 000000002A80: DBFE2000 20000004
	v_mov_b32_e32 v74, 0                                       // 000000002A88: 7E940280
	v_mov_b32_e32 v75, 0                                       // 000000002A8C: 7E960280
	v_mov_b32_e32 v76, 0                                       // 000000002A90: 7E980280
	v_mov_b32_e32 v77, 0                                       // 000000002A94: 7E9A0280
	v_mov_b32_e32 v78, 0                                       // 000000002A98: 7E9C0280
	v_mov_b32_e32 v79, 0                                       // 000000002A9C: 7E9E0280
	v_mov_b32_e32 v80, 0                                       // 000000002AA0: 7EA00280
	v_mov_b32_e32 v81, 0                                       // 000000002AA4: 7EA20280
	v_mov_b32_e32 v82, 0                                       // 000000002AA8: 7EA40280
	v_mov_b32_e32 v83, 0                                       // 000000002AAC: 7EA60280
	v_mov_b32_e32 v84, 0                                       // 000000002AB0: 7EA80280
	v_mov_b32_e32 v85, 0                                       // 000000002AB4: 7EAA0280
	v_mov_b32_e32 v86, 0                                       // 000000002AB8: 7EAC0280
	v_mov_b32_e32 v87, 0                                       // 000000002ABC: 7EAE0280
	v_mov_b32_e32 v88, 0                                       // 000000002AC0: 7EB00280
	v_mov_b32_e32 v89, 0                                       // 000000002AC4: 7EB20280
	v_mov_b32_e32 v90, 0                                       // 000000002AC8: 7EB40280
	v_mov_b32_e32 v91, 0                                       // 000000002ACC: 7EB60280
	v_mov_b32_e32 v92, 0                                       // 000000002AD0: 7EB80280
	v_mov_b32_e32 v93, 0                                       // 000000002AD4: 7EBA0280
	v_mov_b32_e32 v94, 0                                       // 000000002AD8: 7EBC0280
	v_mov_b32_e32 v95, 0                                       // 000000002ADC: 7EBE0280
	v_mov_b32_e32 v96, 0                                       // 000000002AE0: 7EC00280
	v_mov_b32_e32 v97, 0                                       // 000000002AE4: 7EC20280
	v_mov_b32_e32 v98, 0                                       // 000000002AE8: 7EC40280
	v_mov_b32_e32 v99, 0                                       // 000000002AEC: 7EC60280
	v_mov_b32_e32 v100, 0                                      // 000000002AF0: 7EC80280
	v_mov_b32_e32 v101, 0                                      // 000000002AF4: 7ECA0280
	v_mov_b32_e32 v102, 0                                      // 000000002AF8: 7ECC0280
	v_mov_b32_e32 v103, 0                                      // 000000002AFC: 7ECE0280
	v_mov_b32_e32 v104, 0                                      // 000000002B00: 7ED00280
	v_mov_b32_e32 v105, 0                                      // 000000002B04: 7ED20280
	v_lshrrev_b32_e32 v28, 4, v0                               // 000000002B08: 20380084
	v_lshlrev_b32_e32 v28, 2, v28                              // 000000002B0C: 24383882
	v_sub_u32_e32 v29, 12, v28                                 // 000000002B10: 6A3A388C
	v_mov_b32_e32 v30, v29                                     // 000000002B14: 7E3C031D
	s_nop 1                                                    // 000000002B18: BF800001
	v_permlane16_swap_b32_e32 v29, v30                         // 000000002B1C: 7E3AB31E
	s_nop 1                                                    // 000000002B20: BF800001
	v_permlane16_swap_b32_e32 v30, v29                         // 000000002B24: 7E3CB31D
	s_mov_b32 s60, 0xff00ff00                                  // 000000002B28: BEBC00FF FF00FF00
	s_mov_b32 s61, 0xff00ff00                                  // 000000002B30: BEBD00FF FF00FF00
	v_cndmask_b32_e64 v30, v28, v29, s[60:61]                  // 000000002B38: D100001E 00F23B1C
	v_and_b32_e32 v28, 15, v0                                  // 000000002B40: 2638008F
	v_lshlrev_b32_e32 v28, 4, v28                              // 000000002B44: 24383884
	v_add_u32_e32 v20, v28, v30                                // 000000002B48: 68283D1C
	v_lshlrev_b32_e32 v21, 2, v20                              // 000000002B4C: 242A2882
	s_mov_b32 s60, 0x4000                                      // 000000002B50: BEBC00FF 00004000
	v_add_u32_e32 v20, s60, v21                                // 000000002B58: 68282A3C
	s_mov_b32 s61, 0x12000                                     // 000000002B5C: BEBD00FF 00012000
	v_add_u32_e32 v21, s61, v20                                // 000000002B64: 682A283D
	s_mov_b32 s60, 0x4800                                      // 000000002B68: BEBC00FF 00004800
	s_mul_i32 s60, s60, s7                                     // 000000002B70: 923C073C
	v_add_u32_e32 v20, s60, v20                                // 000000002B74: 6828283C
	v_add_u32_e32 v21, s60, v21                                // 000000002B78: 682A2A3C
	v_and_b32_e32 v28, 15, v0                                  // 000000002B7C: 2638008F
	v_lshrrev_b32_e32 v28, 1, v28                              // 000000002B80: 20383881
	v_lshlrev_b32_e32 v28, 4, v28                              // 000000002B84: 24383884
	v_and_b32_e32 v29, 1, v0                                   // 000000002B88: 263A0081
	v_lshlrev_b32_e32 v29, 1, v29                              // 000000002B8C: 243A3A81
	v_lshrrev_b32_e32 v31, 4, v0                               // 000000002B90: 203E0084
	s_mov_b32 s60, 0x900                                       // 000000002B94: BEBC00FF 00000900
	v_mul_i32_i24_e32 v31, s60, v31                            // 000000002B9C: 0C3E3E3C
	v_add_u32_e32 v6, v28, v29                                 // 000000002BA0: 680C3B1C
	v_add_u32_e32 v6, v31, v6                                  // 000000002BA4: 680C0D1F
	s_mov_b32 s60, 0x88                                        // 000000002BA8: BEBC00FF 00000088
	v_add_u32_e32 v7, s60, v6                                  // 000000002BB0: 680E0C3C
	s_mov_b32 s60, 8                                           // 000000002BB4: BEBC0088
	v_add_u32_e32 v8, s60, v6                                  // 000000002BB8: 68100C3C
	s_mov_b32 s60, 0x80                                        // 000000002BBC: BEBC00FF 00000080
	v_add_u32_e32 v9, s60, v6                                  // 000000002BC4: 68120C3C
	s_mov_b32 s60, s7                                          // 000000002BC8: BEBC0007
	s_mov_b32 s61, 0x200                                       // 000000002BCC: BEBD00FF 00000200
	s_mul_i32 s60, s61, s60                                    // 000000002BD4: 923C3C3D
	v_add_u32_e32 v6, s60, v6                                  // 000000002BD8: 680C0C3C
	v_add_u32_e32 v7, s60, v7                                  // 000000002BDC: 680E0E3C
	v_add_u32_e32 v8, s60, v8                                  // 000000002BE0: 6810103C
	v_add_u32_e32 v9, s60, v9                                  // 000000002BE4: 6812123C
	v_lshlrev_b32_e32 v6, 2, v6                                // 000000002BE8: 240C0C82
	v_lshlrev_b32_e32 v7, 2, v7                                // 000000002BEC: 240E0E82
	v_lshlrev_b32_e32 v8, 2, v8                                // 000000002BF0: 24101082
	v_lshlrev_b32_e32 v9, 2, v9                                // 000000002BF4: 24121282
	s_mov_b32 s60, 0x4000                                      // 000000002BF8: BEBC00FF 00004000
	v_add_u32_e32 v6, s60, v6                                  // 000000002C00: 680C0C3C
	v_add_u32_e32 v7, s60, v7                                  // 000000002C04: 680E0E3C
	v_add_u32_e32 v8, s60, v8                                  // 000000002C08: 6810103C
	v_add_u32_e32 v9, s60, v9                                  // 000000002C0C: 6812123C
	s_mov_b32 s60, 0x12000                                     // 000000002C10: BEBC00FF 00012000
	v_add_u32_e32 v10, s60, v6                                 // 000000002C18: 68140C3C
	v_add_u32_e32 v11, s60, v7                                 // 000000002C1C: 68160E3C
	v_add_u32_e32 v12, s60, v8                                 // 000000002C20: 6818103C
	v_add_u32_e32 v13, s60, v9                                 // 000000002C24: 681A123C
	v_mul_u32_u24_e64 v32, v24, s72                            // 000000002C28: D1080020 00009118
	v_mul_u32_u24_e64 v33, v25, s72                            // 000000002C30: D1080021 00009119
	v_add_u32_e32 v32, v32, v1                                 // 000000002C38: 68400320
	v_add_u32_e32 v33, v33, v1                                 // 000000002C3C: 68420321
	s_mov_b32 m0, s56                                          // 000000002C40: BEFC0038
	buffer_load_dwordx4 v32, s[20:23], 0 offen lds             // 000000002C44: E05D1000 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002C4C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:64 lds   // 000000002C54: E05D1040 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002C5C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:128 lds  // 000000002C64: E05D1080 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002C6C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:192 lds  // 000000002C74: E05D10C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002C7C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:256 lds  // 000000002C84: E05D1100 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002C8C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:320 lds  // 000000002C94: E05D1140 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002C9C: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:384 lds  // 000000002CA4: E05D1180 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002CAC: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:448 lds  // 000000002CB4: E05D11C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002CBC: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:512 lds  // 000000002CC4: E05D1200 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002CCC: 817CFF7C 000003C0
	s_mov_b32 m0, s57                                          // 000000002CD4: BEFC0039
	buffer_load_dwordx4 v33, s[20:23], 0 offen lds             // 000000002CD8: E05D1000 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002CE0: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:64 lds   // 000000002CE8: E05D1040 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002CF0: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:128 lds  // 000000002CF8: E05D1080 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002D00: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:192 lds  // 000000002D08: E05D10C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002D10: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:256 lds  // 000000002D18: E05D1100 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002D20: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:320 lds  // 000000002D28: E05D1140 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002D30: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:384 lds  // 000000002D38: E05D1180 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002D40: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:448 lds  // 000000002D48: E05D11C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002D50: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:512 lds  // 000000002D58: E05D1200 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000002D60: 817CFF7C 000003C0
	buffer_load_dword v24, v26, s[24:27], 0 offen              // 000000002D68: E0501000 8006181A
	buffer_load_dword v25, v27, s[24:27], 0 offen              // 000000002D70: E0501000 8006191B
	v_add_u32_e32 v26, s77, v26                                // 000000002D78: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 000000002D7C: 6836364D
	s_waitcnt vmcnt(20)                                        // 000000002D80: BF8C4F74
	s_barrier                                                  // 000000002D84: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 000000002D88: BF8CC07F
	ds_read_b64_tr_b8 a[120:121], v10                          // 000000002D8C: DBC40000 7800000A
	ds_read_b64_tr_b8 a[122:123], v11                          // 000000002D94: DBC40000 7A00000B
	ds_read_b64_tr_b8 a[124:125], v10 offset:36864             // 000000002D9C: DBC49000 7C00000A
	ds_read_b64_tr_b8 a[126:127], v11 offset:36864             // 000000002DA4: DBC49000 7E00000B
	ds_read_b64_tr_b8 a[128:129], v10 offset:16                // 000000002DAC: DBC40010 8000000A
	ds_read_b64_tr_b8 a[130:131], v11 offset:16                // 000000002DB4: DBC40010 8200000B
	ds_read_b64_tr_b8 a[132:133], v10 offset:36880             // 000000002DBC: DBC49010 8400000A
	ds_read_b64_tr_b8 a[134:135], v11 offset:36880             // 000000002DC4: DBC49010 8600000B
	ds_read_b64_tr_b8 a[136:137], v12                          // 000000002DCC: DBC40000 8800000C
	ds_read_b64_tr_b8 a[138:139], v13                          // 000000002DD4: DBC40000 8A00000D
	ds_read_b64_tr_b8 a[140:141], v12 offset:36864             // 000000002DDC: DBC49000 8C00000C
	ds_read_b64_tr_b8 a[142:143], v13 offset:36864             // 000000002DE4: DBC49000 8E00000D
	ds_read_b64_tr_b8 a[144:145], v12 offset:16                // 000000002DEC: DBC40010 9000000C
	ds_read_b64_tr_b8 a[146:147], v13 offset:16                // 000000002DF4: DBC40010 9200000D
	ds_read_b64_tr_b8 a[148:149], v12 offset:36880             // 000000002DFC: DBC49010 9400000C
	ds_read_b64_tr_b8 a[150:151], v13 offset:36880             // 000000002E04: DBC49010 9600000D
	ds_read_b64_tr_b8 a[152:153], v10 offset:1024              // 000000002E0C: DBC40400 9800000A
	ds_read_b64_tr_b8 a[154:155], v11 offset:1024              // 000000002E14: DBC40400 9A00000B
	ds_read_b64_tr_b8 a[156:157], v10 offset:37888             // 000000002E1C: DBC49400 9C00000A
	ds_read_b64_tr_b8 a[158:159], v11 offset:37888             // 000000002E24: DBC49400 9E00000B
	ds_read_b64_tr_b8 a[160:161], v10 offset:1040              // 000000002E2C: DBC40410 A000000A
	ds_read_b64_tr_b8 a[162:163], v11 offset:1040              // 000000002E34: DBC40410 A200000B
	ds_read_b64_tr_b8 a[164:165], v10 offset:37904             // 000000002E3C: DBC49410 A400000A
	ds_read_b64_tr_b8 a[166:167], v11 offset:37904             // 000000002E44: DBC49410 A600000B
	ds_read_b64_tr_b8 a[168:169], v12 offset:1024              // 000000002E4C: DBC40400 A800000C
	ds_read_b64_tr_b8 a[170:171], v13 offset:1024              // 000000002E54: DBC40400 AA00000D
	ds_read_b64_tr_b8 a[172:173], v12 offset:37888             // 000000002E5C: DBC49400 AC00000C
	ds_read_b64_tr_b8 a[174:175], v13 offset:37888             // 000000002E64: DBC49400 AE00000D
	ds_read_b64_tr_b8 a[176:177], v12 offset:1040              // 000000002E6C: DBC40410 B000000C
	ds_read_b64_tr_b8 a[178:179], v13 offset:1040              // 000000002E74: DBC40410 B200000D
	ds_read_b64_tr_b8 a[180:181], v12 offset:37904             // 000000002E7C: DBC49410 B400000C
	ds_read_b64_tr_b8 a[182:183], v13 offset:37904             // 000000002E84: DBC49410 B600000D
	ds_read_b128 a[40:43], v21                                 // 000000002E8C: DBFE0000 28000015
	ds_read_b128 a[44:47], v21 offset:1024                     // 000000002E94: DBFE0400 2C000015
	ds_read_b128 a[48:51], v21 offset:2048                     // 000000002E9C: DBFE0800 30000015
	ds_read_b128 a[52:55], v21 offset:3072                     // 000000002EA4: DBFE0C00 34000015
	ds_read_b128 a[56:59], v21 offset:4096                     // 000000002EAC: DBFE1000 38000015
	ds_read_b128 a[60:63], v21 offset:5120                     // 000000002EB4: DBFE1400 3C000015
	ds_read_b128 a[64:67], v21 offset:6144                     // 000000002EBC: DBFE1800 40000015
	ds_read_b128 a[68:71], v21 offset:7168                     // 000000002EC4: DBFE1C00 44000015
	ds_read_b128 a[72:75], v21 offset:8192                     // 000000002ECC: DBFE2000 48000015
	s_cmp_lt_u32 s75, 1                                        // 000000002ED4: BF0A814B
	s_cbranch_scc1 label_22F0                                  // 000000002ED8: BF8505C5
	s_cmp_lt_i32 s7, 2                                         // 000000002EDC: BF048207
	s_cbranch_scc0 label_176C                                  // 000000002EE0: BF8402E2

0000000000002ee4 <label_0BE4>:
	s_waitcnt lgkmcnt(4)                                       // 000000002EE4: BF8CC47F
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[40:47], a[0:7], 0  // 000000002EE8: D3AD0026 1A020128
	v_mul_u32_u24_e64 v32, v22, s72                            // 000000002EF0: D1080020 00009116
	v_mul_u32_u24_e64 v33, v23, s72                            // 000000002EF8: D1080021 00009117
	v_add_u32_e32 v32, v32, v1                                 // 000000002F00: 68400320
	v_add_u32_e32 v33, v33, v1                                 // 000000002F04: 68420321
	buffer_load_dword v22, v26, s[24:27], 0 offen              // 000000002F08: E0501000 8006161A
	buffer_load_dword v23, v27, s[24:27], 0 offen              // 000000002F10: E0501000 8006171B
	s_mov_b32 m0, s58                                          // 000000002F18: BEFC003A
	buffer_load_dwordx4 v32, s[20:23], 0 offen lds             // 000000002F1C: E05D1000 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002F24: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[48:55], a[8:15], v[38:41]// 000000002F2C: D3AD0026 1C9A1130
	ds_read_b128 a[80:83], v21 offset:9216                     // 000000002F34: DBFE2400 50000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[56:63], a[16:23], v[38:41]// 000000002F3C: D3AD0026 1C9A2138
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:64 lds   // 000000002F44: E05D1040 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002F4C: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[64:71], a[24:31], v[38:41]// 000000002F54: D3AD0026 1C9A3140
	ds_read_b128 a[84:87], v21 offset:10240                    // 000000002F5C: DBFE2800 54000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[72:79], a[32:39], v[38:41]// 000000002F64: D3AD0026 1C9A4148
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:128 lds  // 000000002F6C: E05D1080 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002F74: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:192 lds  // 000000002F7C: E05D10C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002F84: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:256 lds  // 000000002F8C: E05D1100 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002F94: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:320 lds  // 000000002F9C: E05D1140 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002FA4: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:384 lds  // 000000002FAC: E05D1180 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002FB4: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:448 lds  // 000000002FBC: E05D11C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002FC4: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:512 lds  // 000000002FCC: E05D1200 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000002FD4: 817CFF7C 000003C0
	ds_read_b128 a[88:91], v21 offset:11264                    // 000000002FDC: DBFE2C00 58000015
	ds_read_b128 a[92:95], v21 offset:12288                    // 000000002FE4: DBFE3000 5C000015
	ds_read_b128 a[96:99], v21 offset:13312                    // 000000002FEC: DBFE3400 60000015
	ds_read_b128 a[100:103], v21 offset:14336                  // 000000002FF4: DBFE3800 64000015
	ds_read_b128 a[104:107], v21 offset:15360                  // 000000002FFC: DBFE3C00 68000015
	ds_read_b128 a[108:111], v21 offset:16384                  // 000000003004: DBFE4000 6C000015
	ds_read_b128 a[112:115], v21 offset:17408                  // 00000000300C: DBFE4400 70000015
	v_add_u32_e32 v26, s77, v26                                // 000000003014: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 000000003018: 6836364D
	s_waitcnt lgkmcnt(0)                                       // 00000000301C: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[80:87], a[0:7], 0  // 000000003020: D3AD002A 1A020150
	s_mov_b32 m0, s59                                          // 000000003028: BEFC003B
	buffer_load_dwordx4 v33, s[20:23], 0 offen lds             // 00000000302C: E05D1000 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003034: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[88:95], a[8:15], v[42:45]// 00000000303C: D3AD002A 1CAA1158
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[96:103], a[16:23], v[42:45]// 000000003044: D3AD002A 1CAA2160
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:64 lds   // 00000000304C: E05D1040 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003054: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[104:111], a[24:31], v[42:45]// 00000000305C: D3AD002A 1CAA3168
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[112:119], a[32:39], v[42:45]// 000000003064: D3AD002A 1CAA4170
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:128 lds  // 00000000306C: E05D1080 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003074: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:192 lds  // 00000000307C: E05D10C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003084: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:256 lds  // 00000000308C: E05D1100 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003094: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:320 lds  // 00000000309C: E05D1140 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000030A4: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:384 lds  // 0000000030AC: E05D1180 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000030B4: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:448 lds  // 0000000030BC: E05D11C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000030C4: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:512 lds  // 0000000030CC: E05D1200 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000030D4: 817CFF7C 000003C0
	s_nop 2                                                    // 0000000030DC: BF800002
	v_mov_b32_e32 v29, v38                                     // 0000000030E0: 7E3A0326
	v_max3_f32 v29, v38, v39, v29                              // 0000000030E4: D1D3001D 04764F26
	v_max3_f32 v29, v40, v41, v29                              // 0000000030EC: D1D3001D 04765328
	v_max3_f32 v29, v42, v43, v29                              // 0000000030F4: D1D3001D 0476572A
	v_max3_f32 v29, v44, v45, v29                              // 0000000030FC: D1D3001D 04765B2C
	v_mov_b32_e32 v28, v29                                     // 000000003104: 7E38031D
	v_mov_b32_e32 v29, v29                                     // 000000003108: 7E3A031D
	s_nop 1                                                    // 00000000310C: BF800001
	v_permlane16_swap_b32_e32 v28, v29                         // 000000003110: 7E38B31D
	v_mov_b32_e32 v31, v28                                     // 000000003114: 7E3E031C
	v_mov_b32_e32 v30, v29                                     // 000000003118: 7E3C031D
	s_nop 1                                                    // 00000000311C: BF800001
	v_permlane32_swap_b32_e32 v28, v29                         // 000000003120: 7E38B51D
	v_permlane32_swap_b32_e32 v30, v31                         // 000000003124: 7E3CB51F
	v_max3_f32 v29, v28, v29, v29                              // 000000003128: D1D3001D 04763B1C
	v_max3_f32 v29, v30, v31, v29                              // 000000003130: D1D3001D 04763F1E
	ds_write_b32 v36, v29                                      // 000000003138: D81A0000 00001D24
	s_waitcnt lgkmcnt(0)                                       // 000000003140: BF8CC07F
	s_barrier                                                  // 000000003144: BF8A0000
	ds_read_b32 v46, v37                                       // 000000003148: D86C0000 2E000025
	ds_read_b32 v47, v37 offset:256                            // 000000003150: D86C0100 2F000025
	ds_read_b32 v48, v37 offset:512                            // 000000003158: D86C0200 30000025
	ds_read_b32 v49, v37 offset:768                            // 000000003160: D86C0300 31000025
	s_waitcnt lgkmcnt(0)                                       // 000000003168: BF8CC07F
	v_max3_f32 v29, v46, v47, v29                              // 00000000316C: D1D3001D 04765F2E
	v_max3_f32 v29, v48, v49, v29                              // 000000003174: D1D3001D 04766330
	v_mov_b32_e32 v28, 0xff800000                              // 00000000317C: 7E3802FF FF800000
	v_cmp_eq_u32_e64 s[36:37], v28, v2                         // 000000003184: D0CA0024 0002051C
	v_max_f32_e32 v29, v29, v2                                 // 00000000318C: 163A051D
	v_sub_f32_e32 v18, v2, v29                                 // 000000003190: 04243B02
	v_cndmask_b32_e64 v18, v18, 0, s[36:37]                    // 000000003194: D1000012 00910112
	v_mov_b32_e32 v2, v29                                      // 00000000319C: 7E04031D
	v_mul_f32_e32 v29, s100, v29                               // 0000000031A0: 0A3A3A64
	v_mul_f32_e32 v18, s100, v18                               // 0000000031A4: 0A242464
	v_exp_f32_e32 v18, v18                                     // 0000000031A8: 7E244112
	s_mov_b32 s101, s100                                       // 0000000031AC: BEE50064
	v_add_f32_e64 v30, 0, -v29                                 // 0000000031B0: D101001E 40023A80
	v_mov_b32_e32 v31, v30                                     // 0000000031B8: 7E3E031E
	v_pk_fma_f32 v[38:39], v[38:39], s[100:101], v[30:31]      // 0000000031BC: D3B04026 1C78C926
	v_pk_fma_f32 v[40:41], v[40:41], s[100:101], v[30:31]      // 0000000031C4: D3B04028 1C78C928
	v_pk_fma_f32 v[42:43], v[42:43], s[100:101], v[30:31]      // 0000000031CC: D3B0402A 1C78C92A
	v_pk_fma_f32 v[44:45], v[44:45], s[100:101], v[30:31]      // 0000000031D4: D3B0402C 1C78C92C
	v_exp_f32_e32 v38, v38                                     // 0000000031DC: 7E4C4126
	v_exp_f32_e32 v39, v39                                     // 0000000031E0: 7E4E4127
	v_exp_f32_e32 v40, v40                                     // 0000000031E4: 7E504128
	v_exp_f32_e32 v41, v41                                     // 0000000031E8: 7E524129
	v_exp_f32_e32 v42, v42                                     // 0000000031EC: 7E54412A
	v_exp_f32_e32 v43, v43                                     // 0000000031F0: 7E56412B
	v_exp_f32_e32 v44, v44                                     // 0000000031F4: 7E58412C
	v_exp_f32_e32 v45, v45                                     // 0000000031F8: 7E5A412D
	v_mul_f32_e32 v4, v18, v4                                  // 0000000031FC: 0A080912
	v_mov_b32_e32 v28, v38                                     // 000000003200: 7E380326
	v_add_f32_e32 v28, v39, v28                                // 000000003204: 02383927
	v_add_f32_e32 v28, v40, v28                                // 000000003208: 02383928
	v_add_f32_e32 v28, v41, v28                                // 00000000320C: 02383929
	v_add_f32_e32 v28, v42, v28                                // 000000003210: 0238392A
	v_add_f32_e32 v28, v43, v28                                // 000000003214: 0238392B
	v_add_f32_e32 v28, v44, v28                                // 000000003218: 0238392C
	v_add_f32_e32 v28, v45, v28                                // 00000000321C: 0238392D
	v_add_f32_e32 v4, v28, v4                                  // 000000003220: 0208091C
	v_cvt_pk_fp8_f32 v38, v38, v39                             // 000000003224: D2A20026 00024F26
	v_cvt_pk_fp8_f32 v38, v40, v41 op_sel:[0,0,1]              // 00000000322C: D2A24026 00025328
	v_cvt_pk_fp8_f32 v39, v42, v43                             // 000000003234: D2A20027 0002572A
	v_cvt_pk_fp8_f32 v39, v44, v45 op_sel:[0,0,1]              // 00000000323C: D2A24027 00025B2C
	s_nop 0                                                    // 000000003244: BF800000
	v_permlane16_swap_b32_e32 v38, v39                         // 000000003248: 7E4CB327
	ds_write_b64 v34, v[38:39]                                 // 00000000324C: D89A0000 00002622
	s_waitcnt lgkmcnt(0)                                       // 000000003254: BF8CC07F
	s_barrier                                                  // 000000003258: BF8A0000
	ds_read_b64 v[38:39], v35                                  // 00000000325C: D8EC0000 26000023
	ds_read_b64 v[40:41], v35 offset:256                       // 000000003264: D8EC0100 28000023
	ds_read_b64 v[42:43], v35 offset:1024                      // 00000000326C: D8EC0400 2A000023
	ds_read_b64 v[44:45], v35 offset:1280                      // 000000003274: D8EC0500 2C000023
	v_mul_f32_e32 v74, v18, v74                                // 00000000327C: 0A949512
	v_mul_f32_e32 v75, v18, v75                                // 000000003280: 0A969712
	v_mul_f32_e32 v76, v18, v76                                // 000000003284: 0A989912
	v_mul_f32_e32 v77, v18, v77                                // 000000003288: 0A9A9B12
	v_mul_f32_e32 v78, v18, v78                                // 00000000328C: 0A9C9D12
	v_mul_f32_e32 v79, v18, v79                                // 000000003290: 0A9E9F12
	v_mul_f32_e32 v80, v18, v80                                // 000000003294: 0AA0A112
	v_mul_f32_e32 v81, v18, v81                                // 000000003298: 0AA2A312
	v_mul_f32_e32 v82, v18, v82                                // 00000000329C: 0AA4A512
	v_mul_f32_e32 v83, v18, v83                                // 0000000032A0: 0AA6A712
	v_mul_f32_e32 v84, v18, v84                                // 0000000032A4: 0AA8A912
	v_mul_f32_e32 v85, v18, v85                                // 0000000032A8: 0AAAAB12
	v_mul_f32_e32 v86, v18, v86                                // 0000000032AC: 0AACAD12
	v_mul_f32_e32 v87, v18, v87                                // 0000000032B0: 0AAEAF12
	v_mul_f32_e32 v88, v18, v88                                // 0000000032B4: 0AB0B112
	v_mul_f32_e32 v89, v18, v89                                // 0000000032B8: 0AB2B312
	v_mul_f32_e32 v90, v18, v90                                // 0000000032BC: 0AB4B512
	v_mul_f32_e32 v91, v18, v91                                // 0000000032C0: 0AB6B712
	v_mul_f32_e32 v92, v18, v92                                // 0000000032C4: 0AB8B912
	v_mul_f32_e32 v93, v18, v93                                // 0000000032C8: 0ABABB12
	v_mul_f32_e32 v94, v18, v94                                // 0000000032CC: 0ABCBD12
	v_mul_f32_e32 v95, v18, v95                                // 0000000032D0: 0ABEBF12
	v_mul_f32_e32 v96, v18, v96                                // 0000000032D4: 0AC0C112
	v_mul_f32_e32 v97, v18, v97                                // 0000000032D8: 0AC2C312
	v_mul_f32_e32 v98, v18, v98                                // 0000000032DC: 0AC4C512
	v_mul_f32_e32 v99, v18, v99                                // 0000000032E0: 0AC6C712
	v_mul_f32_e32 v100, v18, v100                              // 0000000032E4: 0AC8C912
	v_mul_f32_e32 v101, v18, v101                              // 0000000032E8: 0ACACB12
	v_mul_f32_e32 v102, v18, v102                              // 0000000032EC: 0ACCCD12
	v_mul_f32_e32 v103, v18, v103                              // 0000000032F0: 0ACECF12
	v_mul_f32_e32 v104, v18, v104                              // 0000000032F4: 0AD0D112
	v_mul_f32_e32 v105, v18, v105                              // 0000000032F8: 0AD2D312
	s_waitcnt lgkmcnt(0)                                       // 0000000032FC: BF8CC07F
	s_waitcnt vmcnt(20)                                        // 000000003300: BF8C4F74
	s_barrier                                                  // 000000003304: BF8A0000
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], a[120:127], v[38:45], v[74:77]// 000000003308: D3AD004A 0D2A4D78
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], a[128:135], v[38:45], v[78:81]// 000000003310: D3AD004E 0D3A4D80
	ds_read_b64_tr_b8 a[128:129], v6 offset:16                 // 000000003318: DBC40010 80000006
	ds_read_b64_tr_b8 a[130:131], v7 offset:16                 // 000000003320: DBC40010 82000007
	ds_read_b64_tr_b8 a[132:133], v6 offset:36880              // 000000003328: DBC49010 84000006
	ds_read_b64_tr_b8 a[134:135], v7 offset:36880              // 000000003330: DBC49010 86000007
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], a[136:143], v[38:45], v[82:85]// 000000003338: D3AD0052 0D4A4D88
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], a[144:151], v[38:45], v[86:89]// 000000003340: D3AD0056 0D5A4D90
	ds_read_b64_tr_b8 a[144:145], v8 offset:16                 // 000000003348: DBC40010 90000008
	ds_read_b64_tr_b8 a[146:147], v9 offset:16                 // 000000003350: DBC40010 92000009
	ds_read_b64_tr_b8 a[148:149], v8 offset:36880              // 000000003358: DBC49010 94000008
	ds_read_b64_tr_b8 a[150:151], v9 offset:36880              // 000000003360: DBC49010 96000009
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], a[152:159], v[38:45], v[90:93]// 000000003368: D3AD005A 0D6A4D98
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], a[160:167], v[38:45], v[94:97]// 000000003370: D3AD005E 0D7A4DA0
	ds_read_b64_tr_b8 a[160:161], v6 offset:1040               // 000000003378: DBC40410 A0000006
	ds_read_b64_tr_b8 a[162:163], v7 offset:1040               // 000000003380: DBC40410 A2000007
	ds_read_b64_tr_b8 a[164:165], v6 offset:37904              // 000000003388: DBC49410 A4000006
	ds_read_b64_tr_b8 a[166:167], v7 offset:37904              // 000000003390: DBC49410 A6000007
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], a[168:175], v[38:45], v[98:101]// 000000003398: D3AD0062 0D8A4DA8
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], a[176:183], v[38:45], v[102:105]// 0000000033A0: D3AD0066 0D9A4DB0
	ds_read_b64_tr_b8 a[176:177], v8 offset:1040               // 0000000033A8: DBC40410 B0000008
	ds_read_b64_tr_b8 a[178:179], v9 offset:1040               // 0000000033B0: DBC40410 B2000009
	ds_read_b64_tr_b8 a[180:181], v8 offset:37904              // 0000000033B8: DBC49410 B4000008
	ds_read_b64_tr_b8 a[182:183], v9 offset:37904              // 0000000033C0: DBC49410 B6000009
	ds_read_b64_tr_b8 a[120:121], v6                           // 0000000033C8: DBC40000 78000006
	ds_read_b64_tr_b8 a[122:123], v7                           // 0000000033D0: DBC40000 7A000007
	ds_read_b64_tr_b8 a[124:125], v6 offset:36864              // 0000000033D8: DBC49000 7C000006
	ds_read_b64_tr_b8 a[126:127], v7 offset:36864              // 0000000033E0: DBC49000 7E000007
	ds_read_b64_tr_b8 a[136:137], v8                           // 0000000033E8: DBC40000 88000008
	ds_read_b64_tr_b8 a[138:139], v9                           // 0000000033F0: DBC40000 8A000009
	ds_read_b64_tr_b8 a[140:141], v8 offset:36864              // 0000000033F8: DBC49000 8C000008
	ds_read_b64_tr_b8 a[142:143], v9 offset:36864              // 000000003400: DBC49000 8E000009
	ds_read_b64_tr_b8 a[152:153], v6 offset:1024               // 000000003408: DBC40400 98000006
	ds_read_b64_tr_b8 a[154:155], v7 offset:1024               // 000000003410: DBC40400 9A000007
	ds_read_b64_tr_b8 a[156:157], v6 offset:37888              // 000000003418: DBC49400 9C000006
	ds_read_b64_tr_b8 a[158:159], v7 offset:37888              // 000000003420: DBC49400 9E000007
	ds_read_b64_tr_b8 a[168:169], v8 offset:1024               // 000000003428: DBC40400 A8000008
	ds_read_b64_tr_b8 a[170:171], v9 offset:1024               // 000000003430: DBC40400 AA000009
	ds_read_b64_tr_b8 a[172:173], v8 offset:37888              // 000000003438: DBC49400 AC000008
	ds_read_b64_tr_b8 a[174:175], v9 offset:37888              // 000000003440: DBC49400 AE000009
	s_waitcnt lgkmcnt(0)                                       // 000000003448: BF8CC07F
	ds_read_b128 a[44:47], v20 offset:1024                     // 00000000344C: DBFE0400 2C000014
	ds_read_b128 a[52:55], v20 offset:3072                     // 000000003454: DBFE0C00 34000014
	ds_read_b128 a[60:63], v20 offset:5120                     // 00000000345C: DBFE1400 3C000014
	ds_read_b128 a[68:71], v20 offset:7168                     // 000000003464: DBFE1C00 44000014
	ds_read_b128 a[40:43], v20                                 // 00000000346C: DBFE0000 28000014
	ds_read_b128 a[48:51], v20 offset:2048                     // 000000003474: DBFE0800 30000014
	ds_read_b128 a[56:59], v20 offset:4096                     // 00000000347C: DBFE1000 38000014
	ds_read_b128 a[64:67], v20 offset:6144                     // 000000003484: DBFE1800 40000014
	ds_read_b128 a[72:75], v20 offset:8192                     // 00000000348C: DBFE2000 48000014
	s_addk_i32 s74, 0x1                                        // 000000003494: B74A0001
	s_cmp_lt_i32 s74, s75                                      // 000000003498: BF044B4A
	s_cbranch_scc0 label_1760                                  // 00000000349C: BF840170
	s_waitcnt lgkmcnt(4)                                       // 0000000034A0: BF8CC47F
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[40:47], a[0:7], 0  // 0000000034A4: D3AD0026 1A020128
	v_mul_u32_u24_e64 v32, v24, s72                            // 0000000034AC: D1080020 00009118
	v_mul_u32_u24_e64 v33, v25, s72                            // 0000000034B4: D1080021 00009119
	v_add_u32_e32 v32, v32, v1                                 // 0000000034BC: 68400320
	v_add_u32_e32 v33, v33, v1                                 // 0000000034C0: 68420321
	buffer_load_dword v24, v26, s[24:27], 0 offen              // 0000000034C4: E0501000 8006181A
	buffer_load_dword v25, v27, s[24:27], 0 offen              // 0000000034CC: E0501000 8006191B
	s_mov_b32 m0, s56                                          // 0000000034D4: BEFC0038
	buffer_load_dwordx4 v32, s[20:23], 0 offen lds             // 0000000034D8: E05D1000 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000034E0: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[48:55], a[8:15], v[38:41]// 0000000034E8: D3AD0026 1C9A1130
	ds_read_b128 a[80:83], v20 offset:9216                     // 0000000034F0: DBFE2400 50000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[56:63], a[16:23], v[38:41]// 0000000034F8: D3AD0026 1C9A2138
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:64 lds   // 000000003500: E05D1040 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003508: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[64:71], a[24:31], v[38:41]// 000000003510: D3AD0026 1C9A3140
	ds_read_b128 a[84:87], v20 offset:10240                    // 000000003518: DBFE2800 54000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[72:79], a[32:39], v[38:41]// 000000003520: D3AD0026 1C9A4148
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:128 lds  // 000000003528: E05D1080 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003530: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:192 lds  // 000000003538: E05D10C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003540: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:256 lds  // 000000003548: E05D1100 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003550: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:320 lds  // 000000003558: E05D1140 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003560: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:384 lds  // 000000003568: E05D1180 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003570: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:448 lds  // 000000003578: E05D11C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003580: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:512 lds  // 000000003588: E05D1200 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003590: 817CFF7C 000003C0
	ds_read_b128 a[88:91], v20 offset:11264                    // 000000003598: DBFE2C00 58000014
	ds_read_b128 a[92:95], v20 offset:12288                    // 0000000035A0: DBFE3000 5C000014
	ds_read_b128 a[96:99], v20 offset:13312                    // 0000000035A8: DBFE3400 60000014
	ds_read_b128 a[100:103], v20 offset:14336                  // 0000000035B0: DBFE3800 64000014
	ds_read_b128 a[104:107], v20 offset:15360                  // 0000000035B8: DBFE3C00 68000014
	ds_read_b128 a[108:111], v20 offset:16384                  // 0000000035C0: DBFE4000 6C000014
	ds_read_b128 a[112:115], v20 offset:17408                  // 0000000035C8: DBFE4400 70000014
	v_add_u32_e32 v26, s77, v26                                // 0000000035D0: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 0000000035D4: 6836364D
	s_waitcnt lgkmcnt(0)                                       // 0000000035D8: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[80:87], a[0:7], 0  // 0000000035DC: D3AD002A 1A020150
	s_mov_b32 m0, s57                                          // 0000000035E4: BEFC0039
	buffer_load_dwordx4 v33, s[20:23], 0 offen lds             // 0000000035E8: E05D1000 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000035F0: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[88:95], a[8:15], v[42:45]// 0000000035F8: D3AD002A 1CAA1158
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[96:103], a[16:23], v[42:45]// 000000003600: D3AD002A 1CAA2160
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:64 lds   // 000000003608: E05D1040 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003610: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[104:111], a[24:31], v[42:45]// 000000003618: D3AD002A 1CAA3168
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[112:119], a[32:39], v[42:45]// 000000003620: D3AD002A 1CAA4170
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:128 lds  // 000000003628: E05D1080 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003630: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:192 lds  // 000000003638: E05D10C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003640: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:256 lds  // 000000003648: E05D1100 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003650: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:320 lds  // 000000003658: E05D1140 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003660: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:384 lds  // 000000003668: E05D1180 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003670: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:448 lds  // 000000003678: E05D11C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003680: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:512 lds  // 000000003688: E05D1200 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003690: 817CFF7C 000003C0
	s_nop 2                                                    // 000000003698: BF800002
	v_mov_b32_e32 v29, v38                                     // 00000000369C: 7E3A0326
	v_max3_f32 v29, v38, v39, v29                              // 0000000036A0: D1D3001D 04764F26
	v_max3_f32 v29, v40, v41, v29                              // 0000000036A8: D1D3001D 04765328
	v_max3_f32 v29, v42, v43, v29                              // 0000000036B0: D1D3001D 0476572A
	v_max3_f32 v29, v44, v45, v29                              // 0000000036B8: D1D3001D 04765B2C
	v_mov_b32_e32 v28, v29                                     // 0000000036C0: 7E38031D
	v_mov_b32_e32 v29, v29                                     // 0000000036C4: 7E3A031D
	s_nop 1                                                    // 0000000036C8: BF800001
	v_permlane16_swap_b32_e32 v28, v29                         // 0000000036CC: 7E38B31D
	v_mov_b32_e32 v31, v28                                     // 0000000036D0: 7E3E031C
	v_mov_b32_e32 v30, v29                                     // 0000000036D4: 7E3C031D
	s_nop 1                                                    // 0000000036D8: BF800001
	v_permlane32_swap_b32_e32 v28, v29                         // 0000000036DC: 7E38B51D
	v_permlane32_swap_b32_e32 v30, v31                         // 0000000036E0: 7E3CB51F
	v_max3_f32 v29, v28, v29, v29                              // 0000000036E4: D1D3001D 04763B1C
	v_max3_f32 v29, v30, v31, v29                              // 0000000036EC: D1D3001D 04763F1E
	ds_write_b32 v36, v29                                      // 0000000036F4: D81A0000 00001D24
	s_waitcnt lgkmcnt(0)                                       // 0000000036FC: BF8CC07F
	s_barrier                                                  // 000000003700: BF8A0000
	ds_read_b32 v46, v37                                       // 000000003704: D86C0000 2E000025
	ds_read_b32 v47, v37 offset:256                            // 00000000370C: D86C0100 2F000025
	ds_read_b32 v48, v37 offset:512                            // 000000003714: D86C0200 30000025
	ds_read_b32 v49, v37 offset:768                            // 00000000371C: D86C0300 31000025
	s_waitcnt lgkmcnt(0)                                       // 000000003724: BF8CC07F
	v_max3_f32 v29, v46, v47, v29                              // 000000003728: D1D3001D 04765F2E
	v_max3_f32 v29, v48, v49, v29                              // 000000003730: D1D3001D 04766330
	v_mov_b32_e32 v28, 0xff800000                              // 000000003738: 7E3802FF FF800000
	v_cmp_eq_u32_e64 s[36:37], v28, v2                         // 000000003740: D0CA0024 0002051C
	v_max_f32_e32 v29, v29, v2                                 // 000000003748: 163A051D
	v_sub_f32_e32 v18, v2, v29                                 // 00000000374C: 04243B02
	v_cndmask_b32_e64 v18, v18, 0, s[36:37]                    // 000000003750: D1000012 00910112
	v_mov_b32_e32 v2, v29                                      // 000000003758: 7E04031D
	v_mul_f32_e32 v29, s100, v29                               // 00000000375C: 0A3A3A64
	v_mul_f32_e32 v18, s100, v18                               // 000000003760: 0A242464
	v_exp_f32_e32 v18, v18                                     // 000000003764: 7E244112
	s_mov_b32 s101, s100                                       // 000000003768: BEE50064
	v_add_f32_e64 v30, 0, -v29                                 // 00000000376C: D101001E 40023A80
	v_mov_b32_e32 v31, v30                                     // 000000003774: 7E3E031E
	v_pk_fma_f32 v[38:39], v[38:39], s[100:101], v[30:31]      // 000000003778: D3B04026 1C78C926
	v_pk_fma_f32 v[40:41], v[40:41], s[100:101], v[30:31]      // 000000003780: D3B04028 1C78C928
	v_pk_fma_f32 v[42:43], v[42:43], s[100:101], v[30:31]      // 000000003788: D3B0402A 1C78C92A
	v_pk_fma_f32 v[44:45], v[44:45], s[100:101], v[30:31]      // 000000003790: D3B0402C 1C78C92C
	v_exp_f32_e32 v38, v38                                     // 000000003798: 7E4C4126
	v_exp_f32_e32 v39, v39                                     // 00000000379C: 7E4E4127
	v_exp_f32_e32 v40, v40                                     // 0000000037A0: 7E504128
	v_exp_f32_e32 v41, v41                                     // 0000000037A4: 7E524129
	v_exp_f32_e32 v42, v42                                     // 0000000037A8: 7E54412A
	v_exp_f32_e32 v43, v43                                     // 0000000037AC: 7E56412B
	v_exp_f32_e32 v44, v44                                     // 0000000037B0: 7E58412C
	v_exp_f32_e32 v45, v45                                     // 0000000037B4: 7E5A412D
	v_mul_f32_e32 v4, v18, v4                                  // 0000000037B8: 0A080912
	v_mov_b32_e32 v28, v38                                     // 0000000037BC: 7E380326
	v_add_f32_e32 v28, v39, v28                                // 0000000037C0: 02383927
	v_add_f32_e32 v28, v40, v28                                // 0000000037C4: 02383928
	v_add_f32_e32 v28, v41, v28                                // 0000000037C8: 02383929
	v_add_f32_e32 v28, v42, v28                                // 0000000037CC: 0238392A
	v_add_f32_e32 v28, v43, v28                                // 0000000037D0: 0238392B
	v_add_f32_e32 v28, v44, v28                                // 0000000037D4: 0238392C
	v_add_f32_e32 v28, v45, v28                                // 0000000037D8: 0238392D
	v_add_f32_e32 v4, v28, v4                                  // 0000000037DC: 0208091C
	v_cvt_pk_fp8_f32 v38, v38, v39                             // 0000000037E0: D2A20026 00024F26
	v_cvt_pk_fp8_f32 v38, v40, v41 op_sel:[0,0,1]              // 0000000037E8: D2A24026 00025328
	v_cvt_pk_fp8_f32 v39, v42, v43                             // 0000000037F0: D2A20027 0002572A
	v_cvt_pk_fp8_f32 v39, v44, v45 op_sel:[0,0,1]              // 0000000037F8: D2A24027 00025B2C
	s_nop 0                                                    // 000000003800: BF800000
	v_permlane16_swap_b32_e32 v38, v39                         // 000000003804: 7E4CB327
	ds_write_b64 v34, v[38:39]                                 // 000000003808: D89A0000 00002622
	s_waitcnt lgkmcnt(0)                                       // 000000003810: BF8CC07F
	s_barrier                                                  // 000000003814: BF8A0000
	ds_read_b64 v[38:39], v35                                  // 000000003818: D8EC0000 26000023
	ds_read_b64 v[40:41], v35 offset:256                       // 000000003820: D8EC0100 28000023
	ds_read_b64 v[42:43], v35 offset:1024                      // 000000003828: D8EC0400 2A000023
	ds_read_b64 v[44:45], v35 offset:1280                      // 000000003830: D8EC0500 2C000023
	v_mul_f32_e32 v74, v18, v74                                // 000000003838: 0A949512
	v_mul_f32_e32 v75, v18, v75                                // 00000000383C: 0A969712
	v_mul_f32_e32 v76, v18, v76                                // 000000003840: 0A989912
	v_mul_f32_e32 v77, v18, v77                                // 000000003844: 0A9A9B12
	v_mul_f32_e32 v78, v18, v78                                // 000000003848: 0A9C9D12
	v_mul_f32_e32 v79, v18, v79                                // 00000000384C: 0A9E9F12
	v_mul_f32_e32 v80, v18, v80                                // 000000003850: 0AA0A112
	v_mul_f32_e32 v81, v18, v81                                // 000000003854: 0AA2A312
	v_mul_f32_e32 v82, v18, v82                                // 000000003858: 0AA4A512
	v_mul_f32_e32 v83, v18, v83                                // 00000000385C: 0AA6A712
	v_mul_f32_e32 v84, v18, v84                                // 000000003860: 0AA8A912
	v_mul_f32_e32 v85, v18, v85                                // 000000003864: 0AAAAB12
	v_mul_f32_e32 v86, v18, v86                                // 000000003868: 0AACAD12
	v_mul_f32_e32 v87, v18, v87                                // 00000000386C: 0AAEAF12
	v_mul_f32_e32 v88, v18, v88                                // 000000003870: 0AB0B112
	v_mul_f32_e32 v89, v18, v89                                // 000000003874: 0AB2B312
	v_mul_f32_e32 v90, v18, v90                                // 000000003878: 0AB4B512
	v_mul_f32_e32 v91, v18, v91                                // 00000000387C: 0AB6B712
	v_mul_f32_e32 v92, v18, v92                                // 000000003880: 0AB8B912
	v_mul_f32_e32 v93, v18, v93                                // 000000003884: 0ABABB12
	v_mul_f32_e32 v94, v18, v94                                // 000000003888: 0ABCBD12
	v_mul_f32_e32 v95, v18, v95                                // 00000000388C: 0ABEBF12
	v_mul_f32_e32 v96, v18, v96                                // 000000003890: 0AC0C112
	v_mul_f32_e32 v97, v18, v97                                // 000000003894: 0AC2C312
	v_mul_f32_e32 v98, v18, v98                                // 000000003898: 0AC4C512
	v_mul_f32_e32 v99, v18, v99                                // 00000000389C: 0AC6C712
	v_mul_f32_e32 v100, v18, v100                              // 0000000038A0: 0AC8C912
	v_mul_f32_e32 v101, v18, v101                              // 0000000038A4: 0ACACB12
	v_mul_f32_e32 v102, v18, v102                              // 0000000038A8: 0ACCCD12
	v_mul_f32_e32 v103, v18, v103                              // 0000000038AC: 0ACECF12
	v_mul_f32_e32 v104, v18, v104                              // 0000000038B0: 0AD0D112
	v_mul_f32_e32 v105, v18, v105                              // 0000000038B4: 0AD2D312
	s_waitcnt lgkmcnt(0)                                       // 0000000038B8: BF8CC07F
	s_waitcnt vmcnt(20)                                        // 0000000038BC: BF8C4F74
	s_barrier                                                  // 0000000038C0: BF8A0000
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], a[120:127], v[38:45], v[74:77]// 0000000038C4: D3AD004A 0D2A4D78
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], a[128:135], v[38:45], v[78:81]// 0000000038CC: D3AD004E 0D3A4D80
	ds_read_b64_tr_b8 a[128:129], v10 offset:16                // 0000000038D4: DBC40010 8000000A
	ds_read_b64_tr_b8 a[130:131], v11 offset:16                // 0000000038DC: DBC40010 8200000B
	ds_read_b64_tr_b8 a[132:133], v10 offset:36880             // 0000000038E4: DBC49010 8400000A
	ds_read_b64_tr_b8 a[134:135], v11 offset:36880             // 0000000038EC: DBC49010 8600000B
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], a[136:143], v[38:45], v[82:85]// 0000000038F4: D3AD0052 0D4A4D88
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], a[144:151], v[38:45], v[86:89]// 0000000038FC: D3AD0056 0D5A4D90
	ds_read_b64_tr_b8 a[144:145], v12 offset:16                // 000000003904: DBC40010 9000000C
	ds_read_b64_tr_b8 a[146:147], v13 offset:16                // 00000000390C: DBC40010 9200000D
	ds_read_b64_tr_b8 a[148:149], v12 offset:36880             // 000000003914: DBC49010 9400000C
	ds_read_b64_tr_b8 a[150:151], v13 offset:36880             // 00000000391C: DBC49010 9600000D
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], a[152:159], v[38:45], v[90:93]// 000000003924: D3AD005A 0D6A4D98
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], a[160:167], v[38:45], v[94:97]// 00000000392C: D3AD005E 0D7A4DA0
	ds_read_b64_tr_b8 a[160:161], v10 offset:1040              // 000000003934: DBC40410 A000000A
	ds_read_b64_tr_b8 a[162:163], v11 offset:1040              // 00000000393C: DBC40410 A200000B
	ds_read_b64_tr_b8 a[164:165], v10 offset:37904             // 000000003944: DBC49410 A400000A
	ds_read_b64_tr_b8 a[166:167], v11 offset:37904             // 00000000394C: DBC49410 A600000B
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], a[168:175], v[38:45], v[98:101]// 000000003954: D3AD0062 0D8A4DA8
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], a[176:183], v[38:45], v[102:105]// 00000000395C: D3AD0066 0D9A4DB0
	ds_read_b64_tr_b8 a[176:177], v12 offset:1040              // 000000003964: DBC40410 B000000C
	ds_read_b64_tr_b8 a[178:179], v13 offset:1040              // 00000000396C: DBC40410 B200000D
	ds_read_b64_tr_b8 a[180:181], v12 offset:37904             // 000000003974: DBC49410 B400000C
	ds_read_b64_tr_b8 a[182:183], v13 offset:37904             // 00000000397C: DBC49410 B600000D
	ds_read_b64_tr_b8 a[120:121], v10                          // 000000003984: DBC40000 7800000A
	ds_read_b64_tr_b8 a[122:123], v11                          // 00000000398C: DBC40000 7A00000B
	ds_read_b64_tr_b8 a[124:125], v10 offset:36864             // 000000003994: DBC49000 7C00000A
	ds_read_b64_tr_b8 a[126:127], v11 offset:36864             // 00000000399C: DBC49000 7E00000B
	ds_read_b64_tr_b8 a[136:137], v12                          // 0000000039A4: DBC40000 8800000C
	ds_read_b64_tr_b8 a[138:139], v13                          // 0000000039AC: DBC40000 8A00000D
	ds_read_b64_tr_b8 a[140:141], v12 offset:36864             // 0000000039B4: DBC49000 8C00000C
	ds_read_b64_tr_b8 a[142:143], v13 offset:36864             // 0000000039BC: DBC49000 8E00000D
	ds_read_b64_tr_b8 a[152:153], v10 offset:1024              // 0000000039C4: DBC40400 9800000A
	ds_read_b64_tr_b8 a[154:155], v11 offset:1024              // 0000000039CC: DBC40400 9A00000B
	ds_read_b64_tr_b8 a[156:157], v10 offset:37888             // 0000000039D4: DBC49400 9C00000A
	ds_read_b64_tr_b8 a[158:159], v11 offset:37888             // 0000000039DC: DBC49400 9E00000B
	ds_read_b64_tr_b8 a[168:169], v12 offset:1024              // 0000000039E4: DBC40400 A800000C
	ds_read_b64_tr_b8 a[170:171], v13 offset:1024              // 0000000039EC: DBC40400 AA00000D
	ds_read_b64_tr_b8 a[172:173], v12 offset:37888             // 0000000039F4: DBC49400 AC00000C
	ds_read_b64_tr_b8 a[174:175], v13 offset:37888             // 0000000039FC: DBC49400 AE00000D
	s_waitcnt lgkmcnt(0)                                       // 000000003A04: BF8CC07F
	ds_read_b128 a[44:47], v21 offset:1024                     // 000000003A08: DBFE0400 2C000015
	ds_read_b128 a[52:55], v21 offset:3072                     // 000000003A10: DBFE0C00 34000015
	ds_read_b128 a[60:63], v21 offset:5120                     // 000000003A18: DBFE1400 3C000015
	ds_read_b128 a[68:71], v21 offset:7168                     // 000000003A20: DBFE1C00 44000015
	ds_read_b128 a[40:43], v21                                 // 000000003A28: DBFE0000 28000015
	ds_read_b128 a[48:51], v21 offset:2048                     // 000000003A30: DBFE0800 30000015
	ds_read_b128 a[56:59], v21 offset:4096                     // 000000003A38: DBFE1000 38000015
	ds_read_b128 a[64:67], v21 offset:6144                     // 000000003A40: DBFE1800 40000015
	ds_read_b128 a[72:75], v21 offset:8192                     // 000000003A48: DBFE2000 48000015
	s_addk_i32 s74, 0x1                                        // 000000003A50: B74A0001
	s_cmp_lt_i32 s74, s75                                      // 000000003A54: BF044B4A
	s_cbranch_scc0 label_1760                                  // 000000003A58: BF840001
	s_branch label_0BE4                                        // 000000003A5C: BF82FD21

0000000000003a60 <label_1760>:
	s_nop 0                                                    // 000000003A60: BF800000
	s_nop 0                                                    // 000000003A64: BF800000
	s_branch label_22E8                                        // 000000003A68: BF8202DF

0000000000003a6c <label_176C>:
	s_waitcnt lgkmcnt(4)                                       // 000000003A6C: BF8CC47F
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[40:47], a[0:7], 0  // 000000003A70: D3AD0026 1A020128
	v_mul_u32_u24_e64 v32, v22, s72                            // 000000003A78: D1080020 00009116
	v_mul_u32_u24_e64 v33, v23, s72                            // 000000003A80: D1080021 00009117
	v_add_u32_e32 v32, v32, v1                                 // 000000003A88: 68400320
	v_add_u32_e32 v33, v33, v1                                 // 000000003A8C: 68420321
	buffer_load_dword v22, v26, s[24:27], 0 offen              // 000000003A90: E0501000 8006161A
	buffer_load_dword v23, v27, s[24:27], 0 offen              // 000000003A98: E0501000 8006171B
	ds_read_b128 a[80:83], v21 offset:9216                     // 000000003AA0: DBFE2400 50000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[48:55], a[8:15], v[38:41]// 000000003AA8: D3AD0026 1C9A1130
	s_mov_b32 m0, s58                                          // 000000003AB0: BEFC003A
	buffer_load_dwordx4 v32, s[20:23], 0 offen lds             // 000000003AB4: E05D1000 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003ABC: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[56:63], a[16:23], v[38:41]// 000000003AC4: D3AD0026 1C9A2138
	ds_read_b128 a[84:87], v21 offset:10240                    // 000000003ACC: DBFE2800 54000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[64:71], a[24:31], v[38:41]// 000000003AD4: D3AD0026 1C9A3140
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:64 lds   // 000000003ADC: E05D1040 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003AE4: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[72:79], a[32:39], v[38:41]// 000000003AEC: D3AD0026 1C9A4148
	ds_read_b128 a[88:91], v21 offset:11264                    // 000000003AF4: DBFE2C00 58000015
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:128 lds  // 000000003AFC: E05D1080 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003B04: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:192 lds  // 000000003B0C: E05D10C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003B14: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:256 lds  // 000000003B1C: E05D1100 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003B24: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:320 lds  // 000000003B2C: E05D1140 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003B34: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:384 lds  // 000000003B3C: E05D1180 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003B44: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:448 lds  // 000000003B4C: E05D11C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003B54: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:512 lds  // 000000003B5C: E05D1200 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000003B64: 817CFF7C 000003C0
	ds_read_b128 a[92:95], v21 offset:12288                    // 000000003B6C: DBFE3000 5C000015
	ds_read_b128 a[96:99], v21 offset:13312                    // 000000003B74: DBFE3400 60000015
	ds_read_b128 a[100:103], v21 offset:14336                  // 000000003B7C: DBFE3800 64000015
	ds_read_b128 a[104:107], v21 offset:15360                  // 000000003B84: DBFE3C00 68000015
	ds_read_b128 a[108:111], v21 offset:16384                  // 000000003B8C: DBFE4000 6C000015
	ds_read_b128 a[112:115], v21 offset:17408                  // 000000003B94: DBFE4400 70000015
	v_add_u32_e32 v26, s77, v26                                // 000000003B9C: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 000000003BA0: 6836364D
	s_waitcnt lgkmcnt(0)                                       // 000000003BA4: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[80:87], a[0:7], 0  // 000000003BA8: D3AD002A 1A020150
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[88:95], a[8:15], v[42:45]// 000000003BB0: D3AD002A 1CAA1158
	s_mov_b32 m0, s59                                          // 000000003BB8: BEFC003B
	buffer_load_dwordx4 v33, s[20:23], 0 offen lds             // 000000003BBC: E05D1000 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003BC4: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[96:103], a[16:23], v[42:45]// 000000003BCC: D3AD002A 1CAA2160
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[104:111], a[24:31], v[42:45]// 000000003BD4: D3AD002A 1CAA3168
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:64 lds   // 000000003BDC: E05D1040 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003BE4: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[112:119], a[32:39], v[42:45]// 000000003BEC: D3AD002A 1CAA4170
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:128 lds  // 000000003BF4: E05D1080 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003BFC: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:192 lds  // 000000003C04: E05D10C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003C0C: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:256 lds  // 000000003C14: E05D1100 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003C1C: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:320 lds  // 000000003C24: E05D1140 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003C2C: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:384 lds  // 000000003C34: E05D1180 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003C3C: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:448 lds  // 000000003C44: E05D11C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003C4C: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:512 lds  // 000000003C54: E05D1200 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000003C5C: 817CFF7C 000003C0
	s_nop 2                                                    // 000000003C64: BF800002
	v_mov_b32_e32 v29, v38                                     // 000000003C68: 7E3A0326
	v_max3_f32 v29, v38, v39, v29                              // 000000003C6C: D1D3001D 04764F26
	v_max3_f32 v29, v40, v41, v29                              // 000000003C74: D1D3001D 04765328
	v_max3_f32 v29, v42, v43, v29                              // 000000003C7C: D1D3001D 0476572A
	v_max3_f32 v29, v44, v45, v29                              // 000000003C84: D1D3001D 04765B2C
	v_mov_b32_e32 v28, v29                                     // 000000003C8C: 7E38031D
	v_mov_b32_e32 v29, v29                                     // 000000003C90: 7E3A031D
	s_nop 1                                                    // 000000003C94: BF800001
	v_permlane16_swap_b32_e32 v28, v29                         // 000000003C98: 7E38B31D
	v_mov_b32_e32 v31, v28                                     // 000000003C9C: 7E3E031C
	v_mov_b32_e32 v30, v29                                     // 000000003CA0: 7E3C031D
	s_nop 1                                                    // 000000003CA4: BF800001
	v_permlane32_swap_b32_e32 v28, v29                         // 000000003CA8: 7E38B51D
	v_permlane32_swap_b32_e32 v30, v31                         // 000000003CAC: 7E3CB51F
	v_max3_f32 v29, v28, v29, v29                              // 000000003CB0: D1D3001D 04763B1C
	v_max3_f32 v29, v30, v31, v29                              // 000000003CB8: D1D3001D 04763F1E
	ds_write_b32 v36, v29                                      // 000000003CC0: D81A0000 00001D24
	s_waitcnt lgkmcnt(0)                                       // 000000003CC8: BF8CC07F
	s_barrier                                                  // 000000003CCC: BF8A0000
	ds_read_b32 v46, v37                                       // 000000003CD0: D86C0000 2E000025
	ds_read_b32 v47, v37 offset:256                            // 000000003CD8: D86C0100 2F000025
	ds_read_b32 v48, v37 offset:512                            // 000000003CE0: D86C0200 30000025
	ds_read_b32 v49, v37 offset:768                            // 000000003CE8: D86C0300 31000025
	s_waitcnt lgkmcnt(0)                                       // 000000003CF0: BF8CC07F
	v_max3_f32 v29, v46, v47, v29                              // 000000003CF4: D1D3001D 04765F2E
	v_max3_f32 v29, v48, v49, v29                              // 000000003CFC: D1D3001D 04766330
	v_mov_b32_e32 v28, 0xff800000                              // 000000003D04: 7E3802FF FF800000
	v_cmp_eq_u32_e64 s[36:37], v28, v2                         // 000000003D0C: D0CA0024 0002051C
	v_max_f32_e32 v29, v29, v2                                 // 000000003D14: 163A051D
	v_sub_f32_e32 v18, v2, v29                                 // 000000003D18: 04243B02
	v_cndmask_b32_e64 v18, v18, 0, s[36:37]                    // 000000003D1C: D1000012 00910112
	v_mov_b32_e32 v2, v29                                      // 000000003D24: 7E04031D
	v_mul_f32_e32 v29, s100, v29                               // 000000003D28: 0A3A3A64
	v_mul_f32_e32 v18, s100, v18                               // 000000003D2C: 0A242464
	v_exp_f32_e32 v18, v18                                     // 000000003D30: 7E244112
	s_mov_b32 s101, s100                                       // 000000003D34: BEE50064
	v_add_f32_e64 v30, 0, -v29                                 // 000000003D38: D101001E 40023A80
	v_mov_b32_e32 v31, v30                                     // 000000003D40: 7E3E031E
	v_pk_fma_f32 v[38:39], v[38:39], s[100:101], v[30:31]      // 000000003D44: D3B04026 1C78C926
	v_pk_fma_f32 v[40:41], v[40:41], s[100:101], v[30:31]      // 000000003D4C: D3B04028 1C78C928
	v_pk_fma_f32 v[42:43], v[42:43], s[100:101], v[30:31]      // 000000003D54: D3B0402A 1C78C92A
	v_pk_fma_f32 v[44:45], v[44:45], s[100:101], v[30:31]      // 000000003D5C: D3B0402C 1C78C92C
	v_exp_f32_e32 v38, v38                                     // 000000003D64: 7E4C4126
	v_exp_f32_e32 v39, v39                                     // 000000003D68: 7E4E4127
	v_exp_f32_e32 v40, v40                                     // 000000003D6C: 7E504128
	v_exp_f32_e32 v41, v41                                     // 000000003D70: 7E524129
	v_exp_f32_e32 v42, v42                                     // 000000003D74: 7E54412A
	v_exp_f32_e32 v43, v43                                     // 000000003D78: 7E56412B
	v_exp_f32_e32 v44, v44                                     // 000000003D7C: 7E58412C
	v_exp_f32_e32 v45, v45                                     // 000000003D80: 7E5A412D
	v_mul_f32_e32 v4, v18, v4                                  // 000000003D84: 0A080912
	v_mov_b32_e32 v28, v38                                     // 000000003D88: 7E380326
	v_add_f32_e32 v28, v39, v28                                // 000000003D8C: 02383927
	v_add_f32_e32 v28, v40, v28                                // 000000003D90: 02383928
	v_add_f32_e32 v28, v41, v28                                // 000000003D94: 02383929
	v_add_f32_e32 v28, v42, v28                                // 000000003D98: 0238392A
	v_add_f32_e32 v28, v43, v28                                // 000000003D9C: 0238392B
	v_add_f32_e32 v28, v44, v28                                // 000000003DA0: 0238392C
	v_add_f32_e32 v28, v45, v28                                // 000000003DA4: 0238392D
	v_add_f32_e32 v4, v28, v4                                  // 000000003DA8: 0208091C
	v_cvt_pk_fp8_f32 v38, v38, v39                             // 000000003DAC: D2A20026 00024F26
	v_cvt_pk_fp8_f32 v38, v40, v41 op_sel:[0,0,1]              // 000000003DB4: D2A24026 00025328
	v_cvt_pk_fp8_f32 v39, v42, v43                             // 000000003DBC: D2A20027 0002572A
	v_cvt_pk_fp8_f32 v39, v44, v45 op_sel:[0,0,1]              // 000000003DC4: D2A24027 00025B2C
	s_nop 0                                                    // 000000003DCC: BF800000
	v_permlane16_swap_b32_e32 v38, v39                         // 000000003DD0: 7E4CB327
	ds_write_b64 v34, v[38:39]                                 // 000000003DD4: D89A0000 00002622
	s_waitcnt lgkmcnt(0)                                       // 000000003DDC: BF8CC07F
	s_barrier                                                  // 000000003DE0: BF8A0000
	ds_read_b64 v[38:39], v35                                  // 000000003DE4: D8EC0000 26000023
	ds_read_b64 v[40:41], v35 offset:256                       // 000000003DEC: D8EC0100 28000023
	ds_read_b64 v[42:43], v35 offset:1024                      // 000000003DF4: D8EC0400 2A000023
	ds_read_b64 v[44:45], v35 offset:1280                      // 000000003DFC: D8EC0500 2C000023
	v_mul_f32_e32 v74, v18, v74                                // 000000003E04: 0A949512
	v_mul_f32_e32 v75, v18, v75                                // 000000003E08: 0A969712
	v_mul_f32_e32 v76, v18, v76                                // 000000003E0C: 0A989912
	v_mul_f32_e32 v77, v18, v77                                // 000000003E10: 0A9A9B12
	v_mul_f32_e32 v78, v18, v78                                // 000000003E14: 0A9C9D12
	v_mul_f32_e32 v79, v18, v79                                // 000000003E18: 0A9E9F12
	v_mul_f32_e32 v80, v18, v80                                // 000000003E1C: 0AA0A112
	v_mul_f32_e32 v81, v18, v81                                // 000000003E20: 0AA2A312
	v_mul_f32_e32 v82, v18, v82                                // 000000003E24: 0AA4A512
	v_mul_f32_e32 v83, v18, v83                                // 000000003E28: 0AA6A712
	v_mul_f32_e32 v84, v18, v84                                // 000000003E2C: 0AA8A912
	v_mul_f32_e32 v85, v18, v85                                // 000000003E30: 0AAAAB12
	v_mul_f32_e32 v86, v18, v86                                // 000000003E34: 0AACAD12
	v_mul_f32_e32 v87, v18, v87                                // 000000003E38: 0AAEAF12
	v_mul_f32_e32 v88, v18, v88                                // 000000003E3C: 0AB0B112
	v_mul_f32_e32 v89, v18, v89                                // 000000003E40: 0AB2B312
	v_mul_f32_e32 v90, v18, v90                                // 000000003E44: 0AB4B512
	v_mul_f32_e32 v91, v18, v91                                // 000000003E48: 0AB6B712
	v_mul_f32_e32 v92, v18, v92                                // 000000003E4C: 0AB8B912
	v_mul_f32_e32 v93, v18, v93                                // 000000003E50: 0ABABB12
	v_mul_f32_e32 v94, v18, v94                                // 000000003E54: 0ABCBD12
	v_mul_f32_e32 v95, v18, v95                                // 000000003E58: 0ABEBF12
	v_mul_f32_e32 v96, v18, v96                                // 000000003E5C: 0AC0C112
	v_mul_f32_e32 v97, v18, v97                                // 000000003E60: 0AC2C312
	v_mul_f32_e32 v98, v18, v98                                // 000000003E64: 0AC4C512
	v_mul_f32_e32 v99, v18, v99                                // 000000003E68: 0AC6C712
	v_mul_f32_e32 v100, v18, v100                              // 000000003E6C: 0AC8C912
	v_mul_f32_e32 v101, v18, v101                              // 000000003E70: 0ACACB12
	v_mul_f32_e32 v102, v18, v102                              // 000000003E74: 0ACCCD12
	v_mul_f32_e32 v103, v18, v103                              // 000000003E78: 0ACECF12
	v_mul_f32_e32 v104, v18, v104                              // 000000003E7C: 0AD0D112
	v_mul_f32_e32 v105, v18, v105                              // 000000003E80: 0AD2D312
	s_waitcnt lgkmcnt(0)                                       // 000000003E84: BF8CC07F
	s_waitcnt vmcnt(20)                                        // 000000003E88: BF8C4F74
	s_barrier                                                  // 000000003E8C: BF8A0000
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], a[120:127], v[38:45], v[74:77]// 000000003E90: D3AD004A 0D2A4D78
	ds_read_b64_tr_b8 a[120:121], v6                           // 000000003E98: DBC40000 78000006
	ds_read_b64_tr_b8 a[122:123], v7                           // 000000003EA0: DBC40000 7A000007
	ds_read_b64_tr_b8 a[124:125], v6 offset:36864              // 000000003EA8: DBC49000 7C000006
	ds_read_b64_tr_b8 a[126:127], v7 offset:36864              // 000000003EB0: DBC49000 7E000007
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], a[128:135], v[38:45], v[78:81]// 000000003EB8: D3AD004E 0D3A4D80
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], a[136:143], v[38:45], v[82:85]// 000000003EC0: D3AD0052 0D4A4D88
	ds_read_b64_tr_b8 a[136:137], v8                           // 000000003EC8: DBC40000 88000008
	ds_read_b64_tr_b8 a[138:139], v9                           // 000000003ED0: DBC40000 8A000009
	ds_read_b64_tr_b8 a[140:141], v8 offset:36864              // 000000003ED8: DBC49000 8C000008
	ds_read_b64_tr_b8 a[142:143], v9 offset:36864              // 000000003EE0: DBC49000 8E000009
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], a[144:151], v[38:45], v[86:89]// 000000003EE8: D3AD0056 0D5A4D90
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], a[152:159], v[38:45], v[90:93]// 000000003EF0: D3AD005A 0D6A4D98
	ds_read_b64_tr_b8 a[152:153], v6 offset:1024               // 000000003EF8: DBC40400 98000006
	ds_read_b64_tr_b8 a[154:155], v7 offset:1024               // 000000003F00: DBC40400 9A000007
	ds_read_b64_tr_b8 a[156:157], v6 offset:37888              // 000000003F08: DBC49400 9C000006
	ds_read_b64_tr_b8 a[158:159], v7 offset:37888              // 000000003F10: DBC49400 9E000007
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], a[160:167], v[38:45], v[94:97]// 000000003F18: D3AD005E 0D7A4DA0
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], a[168:175], v[38:45], v[98:101]// 000000003F20: D3AD0062 0D8A4DA8
	ds_read_b64_tr_b8 a[168:169], v8 offset:1024               // 000000003F28: DBC40400 A8000008
	ds_read_b64_tr_b8 a[170:171], v9 offset:1024               // 000000003F30: DBC40400 AA000009
	ds_read_b64_tr_b8 a[172:173], v8 offset:37888              // 000000003F38: DBC49400 AC000008
	ds_read_b64_tr_b8 a[174:175], v9 offset:37888              // 000000003F40: DBC49400 AE000009
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], a[176:183], v[38:45], v[102:105]// 000000003F48: D3AD0066 0D9A4DB0
	ds_read_b64_tr_b8 a[128:129], v6 offset:16                 // 000000003F50: DBC40010 80000006
	ds_read_b64_tr_b8 a[130:131], v7 offset:16                 // 000000003F58: DBC40010 82000007
	ds_read_b64_tr_b8 a[132:133], v6 offset:36880              // 000000003F60: DBC49010 84000006
	ds_read_b64_tr_b8 a[134:135], v7 offset:36880              // 000000003F68: DBC49010 86000007
	ds_read_b64_tr_b8 a[144:145], v8 offset:16                 // 000000003F70: DBC40010 90000008
	ds_read_b64_tr_b8 a[146:147], v9 offset:16                 // 000000003F78: DBC40010 92000009
	ds_read_b64_tr_b8 a[148:149], v8 offset:36880              // 000000003F80: DBC49010 94000008
	ds_read_b64_tr_b8 a[150:151], v9 offset:36880              // 000000003F88: DBC49010 96000009
	ds_read_b64_tr_b8 a[160:161], v6 offset:1040               // 000000003F90: DBC40410 A0000006
	ds_read_b64_tr_b8 a[162:163], v7 offset:1040               // 000000003F98: DBC40410 A2000007
	ds_read_b64_tr_b8 a[164:165], v6 offset:37904              // 000000003FA0: DBC49410 A4000006
	ds_read_b64_tr_b8 a[166:167], v7 offset:37904              // 000000003FA8: DBC49410 A6000007
	ds_read_b64_tr_b8 a[176:177], v8 offset:1040               // 000000003FB0: DBC40410 B0000008
	ds_read_b64_tr_b8 a[178:179], v9 offset:1040               // 000000003FB8: DBC40410 B2000009
	ds_read_b64_tr_b8 a[180:181], v8 offset:37904              // 000000003FC0: DBC49410 B4000008
	ds_read_b64_tr_b8 a[182:183], v9 offset:37904              // 000000003FC8: DBC49410 B6000009
	s_waitcnt lgkmcnt(0)                                       // 000000003FD0: BF8CC07F
	ds_read_b128 a[40:43], v20                                 // 000000003FD4: DBFE0000 28000014
	ds_read_b128 a[48:51], v20 offset:2048                     // 000000003FDC: DBFE0800 30000014
	ds_read_b128 a[56:59], v20 offset:4096                     // 000000003FE4: DBFE1000 38000014
	ds_read_b128 a[64:67], v20 offset:6144                     // 000000003FEC: DBFE1800 40000014
	ds_read_b128 a[72:75], v20 offset:8192                     // 000000003FF4: DBFE2000 48000014
	ds_read_b128 a[44:47], v20 offset:1024                     // 000000003FFC: DBFE0400 2C000014
	ds_read_b128 a[52:55], v20 offset:3072                     // 000000004004: DBFE0C00 34000014
	ds_read_b128 a[60:63], v20 offset:5120                     // 00000000400C: DBFE1400 3C000014
	ds_read_b128 a[68:71], v20 offset:7168                     // 000000004014: DBFE1C00 44000014
	s_addk_i32 s74, 0x1                                        // 00000000401C: B74A0001
	s_cmp_lt_i32 s74, s75                                      // 000000004020: BF044B4A
	s_cbranch_scc0 label_1760                                  // 000000004024: BF84FE8E
	s_waitcnt lgkmcnt(4)                                       // 000000004028: BF8CC47F
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[40:47], a[0:7], 0  // 00000000402C: D3AD0026 1A020128
	v_mul_u32_u24_e64 v32, v24, s72                            // 000000004034: D1080020 00009118
	v_mul_u32_u24_e64 v33, v25, s72                            // 00000000403C: D1080021 00009119
	v_add_u32_e32 v32, v32, v1                                 // 000000004044: 68400320
	v_add_u32_e32 v33, v33, v1                                 // 000000004048: 68420321
	buffer_load_dword v24, v26, s[24:27], 0 offen              // 00000000404C: E0501000 8006181A
	buffer_load_dword v25, v27, s[24:27], 0 offen              // 000000004054: E0501000 8006191B
	ds_read_b128 a[80:83], v20 offset:9216                     // 00000000405C: DBFE2400 50000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[48:55], a[8:15], v[38:41]// 000000004064: D3AD0026 1C9A1130
	s_mov_b32 m0, s56                                          // 00000000406C: BEFC0038
	buffer_load_dwordx4 v32, s[20:23], 0 offen lds             // 000000004070: E05D1000 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000004078: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[56:63], a[16:23], v[38:41]// 000000004080: D3AD0026 1C9A2138
	ds_read_b128 a[84:87], v20 offset:10240                    // 000000004088: DBFE2800 54000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[64:71], a[24:31], v[38:41]// 000000004090: D3AD0026 1C9A3140
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:64 lds   // 000000004098: E05D1040 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000040A0: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[72:79], a[32:39], v[38:41]// 0000000040A8: D3AD0026 1C9A4148
	ds_read_b128 a[88:91], v20 offset:11264                    // 0000000040B0: DBFE2C00 58000014
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:128 lds  // 0000000040B8: E05D1080 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000040C0: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:192 lds  // 0000000040C8: E05D10C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000040D0: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:256 lds  // 0000000040D8: E05D1100 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000040E0: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:320 lds  // 0000000040E8: E05D1140 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 0000000040F0: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:384 lds  // 0000000040F8: E05D1180 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000004100: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:448 lds  // 000000004108: E05D11C0 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000004110: 817CFF7C 000003C0
	buffer_load_dwordx4 v32, s[20:23], 0 offen offset:512 lds  // 000000004118: E05D1200 80050020
	s_add_i32 m0, m0, 0x3c0                                    // 000000004120: 817CFF7C 000003C0
	ds_read_b128 a[92:95], v20 offset:12288                    // 000000004128: DBFE3000 5C000014
	ds_read_b128 a[96:99], v20 offset:13312                    // 000000004130: DBFE3400 60000014
	ds_read_b128 a[100:103], v20 offset:14336                  // 000000004138: DBFE3800 64000014
	ds_read_b128 a[104:107], v20 offset:15360                  // 000000004140: DBFE3C00 68000014
	ds_read_b128 a[108:111], v20 offset:16384                  // 000000004148: DBFE4000 6C000014
	ds_read_b128 a[112:115], v20 offset:17408                  // 000000004150: DBFE4400 70000014
	v_add_u32_e32 v26, s77, v26                                // 000000004158: 6834344D
	v_add_u32_e32 v27, s77, v27                                // 00000000415C: 6836364D
	s_waitcnt lgkmcnt(0)                                       // 000000004160: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[80:87], a[0:7], 0  // 000000004164: D3AD002A 1A020150
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[88:95], a[8:15], v[42:45]// 00000000416C: D3AD002A 1CAA1158
	s_mov_b32 m0, s57                                          // 000000004174: BEFC0039
	buffer_load_dwordx4 v33, s[20:23], 0 offen lds             // 000000004178: E05D1000 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000004180: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[96:103], a[16:23], v[42:45]// 000000004188: D3AD002A 1CAA2160
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[104:111], a[24:31], v[42:45]// 000000004190: D3AD002A 1CAA3168
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:64 lds   // 000000004198: E05D1040 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000041A0: 817CFF7C 000003C0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[112:119], a[32:39], v[42:45]// 0000000041A8: D3AD002A 1CAA4170
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:128 lds  // 0000000041B0: E05D1080 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000041B8: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:192 lds  // 0000000041C0: E05D10C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000041C8: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:256 lds  // 0000000041D0: E05D1100 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000041D8: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:320 lds  // 0000000041E0: E05D1140 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000041E8: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:384 lds  // 0000000041F0: E05D1180 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 0000000041F8: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:448 lds  // 000000004200: E05D11C0 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000004208: 817CFF7C 000003C0
	buffer_load_dwordx4 v33, s[20:23], 0 offen offset:512 lds  // 000000004210: E05D1200 80050021
	s_add_i32 m0, m0, 0x3c0                                    // 000000004218: 817CFF7C 000003C0
	s_nop 2                                                    // 000000004220: BF800002
	v_mov_b32_e32 v29, v38                                     // 000000004224: 7E3A0326
	v_max3_f32 v29, v38, v39, v29                              // 000000004228: D1D3001D 04764F26
	v_max3_f32 v29, v40, v41, v29                              // 000000004230: D1D3001D 04765328
	v_max3_f32 v29, v42, v43, v29                              // 000000004238: D1D3001D 0476572A
	v_max3_f32 v29, v44, v45, v29                              // 000000004240: D1D3001D 04765B2C
	v_mov_b32_e32 v28, v29                                     // 000000004248: 7E38031D
	v_mov_b32_e32 v29, v29                                     // 00000000424C: 7E3A031D
	s_nop 1                                                    // 000000004250: BF800001
	v_permlane16_swap_b32_e32 v28, v29                         // 000000004254: 7E38B31D
	v_mov_b32_e32 v31, v28                                     // 000000004258: 7E3E031C
	v_mov_b32_e32 v30, v29                                     // 00000000425C: 7E3C031D
	s_nop 1                                                    // 000000004260: BF800001
	v_permlane32_swap_b32_e32 v28, v29                         // 000000004264: 7E38B51D
	v_permlane32_swap_b32_e32 v30, v31                         // 000000004268: 7E3CB51F
	v_max3_f32 v29, v28, v29, v29                              // 00000000426C: D1D3001D 04763B1C
	v_max3_f32 v29, v30, v31, v29                              // 000000004274: D1D3001D 04763F1E
	ds_write_b32 v36, v29                                      // 00000000427C: D81A0000 00001D24
	s_waitcnt lgkmcnt(0)                                       // 000000004284: BF8CC07F
	s_barrier                                                  // 000000004288: BF8A0000
	ds_read_b32 v46, v37                                       // 00000000428C: D86C0000 2E000025
	ds_read_b32 v47, v37 offset:256                            // 000000004294: D86C0100 2F000025
	ds_read_b32 v48, v37 offset:512                            // 00000000429C: D86C0200 30000025
	ds_read_b32 v49, v37 offset:768                            // 0000000042A4: D86C0300 31000025
	s_waitcnt lgkmcnt(0)                                       // 0000000042AC: BF8CC07F
	v_max3_f32 v29, v46, v47, v29                              // 0000000042B0: D1D3001D 04765F2E
	v_max3_f32 v29, v48, v49, v29                              // 0000000042B8: D1D3001D 04766330
	v_mov_b32_e32 v28, 0xff800000                              // 0000000042C0: 7E3802FF FF800000
	v_cmp_eq_u32_e64 s[36:37], v28, v2                         // 0000000042C8: D0CA0024 0002051C
	v_max_f32_e32 v29, v29, v2                                 // 0000000042D0: 163A051D
	v_sub_f32_e32 v18, v2, v29                                 // 0000000042D4: 04243B02
	v_cndmask_b32_e64 v18, v18, 0, s[36:37]                    // 0000000042D8: D1000012 00910112
	v_mov_b32_e32 v2, v29                                      // 0000000042E0: 7E04031D
	v_mul_f32_e32 v29, s100, v29                               // 0000000042E4: 0A3A3A64
	v_mul_f32_e32 v18, s100, v18                               // 0000000042E8: 0A242464
	v_exp_f32_e32 v18, v18                                     // 0000000042EC: 7E244112
	s_mov_b32 s101, s100                                       // 0000000042F0: BEE50064
	v_add_f32_e64 v30, 0, -v29                                 // 0000000042F4: D101001E 40023A80
	v_mov_b32_e32 v31, v30                                     // 0000000042FC: 7E3E031E
	v_pk_fma_f32 v[38:39], v[38:39], s[100:101], v[30:31]      // 000000004300: D3B04026 1C78C926
	v_pk_fma_f32 v[40:41], v[40:41], s[100:101], v[30:31]      // 000000004308: D3B04028 1C78C928
	v_pk_fma_f32 v[42:43], v[42:43], s[100:101], v[30:31]      // 000000004310: D3B0402A 1C78C92A
	v_pk_fma_f32 v[44:45], v[44:45], s[100:101], v[30:31]      // 000000004318: D3B0402C 1C78C92C
	v_exp_f32_e32 v38, v38                                     // 000000004320: 7E4C4126
	v_exp_f32_e32 v39, v39                                     // 000000004324: 7E4E4127
	v_exp_f32_e32 v40, v40                                     // 000000004328: 7E504128
	v_exp_f32_e32 v41, v41                                     // 00000000432C: 7E524129
	v_exp_f32_e32 v42, v42                                     // 000000004330: 7E54412A
	v_exp_f32_e32 v43, v43                                     // 000000004334: 7E56412B
	v_exp_f32_e32 v44, v44                                     // 000000004338: 7E58412C
	v_exp_f32_e32 v45, v45                                     // 00000000433C: 7E5A412D
	v_mul_f32_e32 v4, v18, v4                                  // 000000004340: 0A080912
	v_mov_b32_e32 v28, v38                                     // 000000004344: 7E380326
	v_add_f32_e32 v28, v39, v28                                // 000000004348: 02383927
	v_add_f32_e32 v28, v40, v28                                // 00000000434C: 02383928
	v_add_f32_e32 v28, v41, v28                                // 000000004350: 02383929
	v_add_f32_e32 v28, v42, v28                                // 000000004354: 0238392A
	v_add_f32_e32 v28, v43, v28                                // 000000004358: 0238392B
	v_add_f32_e32 v28, v44, v28                                // 00000000435C: 0238392C
	v_add_f32_e32 v28, v45, v28                                // 000000004360: 0238392D
	v_add_f32_e32 v4, v28, v4                                  // 000000004364: 0208091C
	v_cvt_pk_fp8_f32 v38, v38, v39                             // 000000004368: D2A20026 00024F26
	v_cvt_pk_fp8_f32 v38, v40, v41 op_sel:[0,0,1]              // 000000004370: D2A24026 00025328
	v_cvt_pk_fp8_f32 v39, v42, v43                             // 000000004378: D2A20027 0002572A
	v_cvt_pk_fp8_f32 v39, v44, v45 op_sel:[0,0,1]              // 000000004380: D2A24027 00025B2C
	s_nop 0                                                    // 000000004388: BF800000
	v_permlane16_swap_b32_e32 v38, v39                         // 00000000438C: 7E4CB327
	ds_write_b64 v34, v[38:39]                                 // 000000004390: D89A0000 00002622
	s_waitcnt lgkmcnt(0)                                       // 000000004398: BF8CC07F
	s_barrier                                                  // 00000000439C: BF8A0000
	ds_read_b64 v[38:39], v35                                  // 0000000043A0: D8EC0000 26000023
	ds_read_b64 v[40:41], v35 offset:256                       // 0000000043A8: D8EC0100 28000023
	ds_read_b64 v[42:43], v35 offset:1024                      // 0000000043B0: D8EC0400 2A000023
	ds_read_b64 v[44:45], v35 offset:1280                      // 0000000043B8: D8EC0500 2C000023
	v_mul_f32_e32 v74, v18, v74                                // 0000000043C0: 0A949512
	v_mul_f32_e32 v75, v18, v75                                // 0000000043C4: 0A969712
	v_mul_f32_e32 v76, v18, v76                                // 0000000043C8: 0A989912
	v_mul_f32_e32 v77, v18, v77                                // 0000000043CC: 0A9A9B12
	v_mul_f32_e32 v78, v18, v78                                // 0000000043D0: 0A9C9D12
	v_mul_f32_e32 v79, v18, v79                                // 0000000043D4: 0A9E9F12
	v_mul_f32_e32 v80, v18, v80                                // 0000000043D8: 0AA0A112
	v_mul_f32_e32 v81, v18, v81                                // 0000000043DC: 0AA2A312
	v_mul_f32_e32 v82, v18, v82                                // 0000000043E0: 0AA4A512
	v_mul_f32_e32 v83, v18, v83                                // 0000000043E4: 0AA6A712
	v_mul_f32_e32 v84, v18, v84                                // 0000000043E8: 0AA8A912
	v_mul_f32_e32 v85, v18, v85                                // 0000000043EC: 0AAAAB12
	v_mul_f32_e32 v86, v18, v86                                // 0000000043F0: 0AACAD12
	v_mul_f32_e32 v87, v18, v87                                // 0000000043F4: 0AAEAF12
	v_mul_f32_e32 v88, v18, v88                                // 0000000043F8: 0AB0B112
	v_mul_f32_e32 v89, v18, v89                                // 0000000043FC: 0AB2B312
	v_mul_f32_e32 v90, v18, v90                                // 000000004400: 0AB4B512
	v_mul_f32_e32 v91, v18, v91                                // 000000004404: 0AB6B712
	v_mul_f32_e32 v92, v18, v92                                // 000000004408: 0AB8B912
	v_mul_f32_e32 v93, v18, v93                                // 00000000440C: 0ABABB12
	v_mul_f32_e32 v94, v18, v94                                // 000000004410: 0ABCBD12
	v_mul_f32_e32 v95, v18, v95                                // 000000004414: 0ABEBF12
	v_mul_f32_e32 v96, v18, v96                                // 000000004418: 0AC0C112
	v_mul_f32_e32 v97, v18, v97                                // 00000000441C: 0AC2C312
	v_mul_f32_e32 v98, v18, v98                                // 000000004420: 0AC4C512
	v_mul_f32_e32 v99, v18, v99                                // 000000004424: 0AC6C712
	v_mul_f32_e32 v100, v18, v100                              // 000000004428: 0AC8C912
	v_mul_f32_e32 v101, v18, v101                              // 00000000442C: 0ACACB12
	v_mul_f32_e32 v102, v18, v102                              // 000000004430: 0ACCCD12
	v_mul_f32_e32 v103, v18, v103                              // 000000004434: 0ACECF12
	v_mul_f32_e32 v104, v18, v104                              // 000000004438: 0AD0D112
	v_mul_f32_e32 v105, v18, v105                              // 00000000443C: 0AD2D312
	s_waitcnt lgkmcnt(0)                                       // 000000004440: BF8CC07F
	s_waitcnt vmcnt(20)                                        // 000000004444: BF8C4F74
	s_barrier                                                  // 000000004448: BF8A0000
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], a[120:127], v[38:45], v[74:77]// 00000000444C: D3AD004A 0D2A4D78
	ds_read_b64_tr_b8 a[120:121], v10                          // 000000004454: DBC40000 7800000A
	ds_read_b64_tr_b8 a[122:123], v11                          // 00000000445C: DBC40000 7A00000B
	ds_read_b64_tr_b8 a[124:125], v10 offset:36864             // 000000004464: DBC49000 7C00000A
	ds_read_b64_tr_b8 a[126:127], v11 offset:36864             // 00000000446C: DBC49000 7E00000B
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], a[128:135], v[38:45], v[78:81]// 000000004474: D3AD004E 0D3A4D80
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], a[136:143], v[38:45], v[82:85]// 00000000447C: D3AD0052 0D4A4D88
	ds_read_b64_tr_b8 a[136:137], v12                          // 000000004484: DBC40000 8800000C
	ds_read_b64_tr_b8 a[138:139], v13                          // 00000000448C: DBC40000 8A00000D
	ds_read_b64_tr_b8 a[140:141], v12 offset:36864             // 000000004494: DBC49000 8C00000C
	ds_read_b64_tr_b8 a[142:143], v13 offset:36864             // 00000000449C: DBC49000 8E00000D
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], a[144:151], v[38:45], v[86:89]// 0000000044A4: D3AD0056 0D5A4D90
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], a[152:159], v[38:45], v[90:93]// 0000000044AC: D3AD005A 0D6A4D98
	ds_read_b64_tr_b8 a[152:153], v10 offset:1024              // 0000000044B4: DBC40400 9800000A
	ds_read_b64_tr_b8 a[154:155], v11 offset:1024              // 0000000044BC: DBC40400 9A00000B
	ds_read_b64_tr_b8 a[156:157], v10 offset:37888             // 0000000044C4: DBC49400 9C00000A
	ds_read_b64_tr_b8 a[158:159], v11 offset:37888             // 0000000044CC: DBC49400 9E00000B
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], a[160:167], v[38:45], v[94:97]// 0000000044D4: D3AD005E 0D7A4DA0
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], a[168:175], v[38:45], v[98:101]// 0000000044DC: D3AD0062 0D8A4DA8
	ds_read_b64_tr_b8 a[168:169], v12 offset:1024              // 0000000044E4: DBC40400 A800000C
	ds_read_b64_tr_b8 a[170:171], v13 offset:1024              // 0000000044EC: DBC40400 AA00000D
	ds_read_b64_tr_b8 a[172:173], v12 offset:37888             // 0000000044F4: DBC49400 AC00000C
	ds_read_b64_tr_b8 a[174:175], v13 offset:37888             // 0000000044FC: DBC49400 AE00000D
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], a[176:183], v[38:45], v[102:105]// 000000004504: D3AD0066 0D9A4DB0
	ds_read_b64_tr_b8 a[128:129], v10 offset:16                // 00000000450C: DBC40010 8000000A
	ds_read_b64_tr_b8 a[130:131], v11 offset:16                // 000000004514: DBC40010 8200000B
	ds_read_b64_tr_b8 a[132:133], v10 offset:36880             // 00000000451C: DBC49010 8400000A
	ds_read_b64_tr_b8 a[134:135], v11 offset:36880             // 000000004524: DBC49010 8600000B
	ds_read_b64_tr_b8 a[144:145], v12 offset:16                // 00000000452C: DBC40010 9000000C
	ds_read_b64_tr_b8 a[146:147], v13 offset:16                // 000000004534: DBC40010 9200000D
	ds_read_b64_tr_b8 a[148:149], v12 offset:36880             // 00000000453C: DBC49010 9400000C
	ds_read_b64_tr_b8 a[150:151], v13 offset:36880             // 000000004544: DBC49010 9600000D
	ds_read_b64_tr_b8 a[160:161], v10 offset:1040              // 00000000454C: DBC40410 A000000A
	ds_read_b64_tr_b8 a[162:163], v11 offset:1040              // 000000004554: DBC40410 A200000B
	ds_read_b64_tr_b8 a[164:165], v10 offset:37904             // 00000000455C: DBC49410 A400000A
	ds_read_b64_tr_b8 a[166:167], v11 offset:37904             // 000000004564: DBC49410 A600000B
	ds_read_b64_tr_b8 a[176:177], v12 offset:1040              // 00000000456C: DBC40410 B000000C
	ds_read_b64_tr_b8 a[178:179], v13 offset:1040              // 000000004574: DBC40410 B200000D
	ds_read_b64_tr_b8 a[180:181], v12 offset:37904             // 00000000457C: DBC49410 B400000C
	ds_read_b64_tr_b8 a[182:183], v13 offset:37904             // 000000004584: DBC49410 B600000D
	s_waitcnt lgkmcnt(0)                                       // 00000000458C: BF8CC07F
	ds_read_b128 a[40:43], v21                                 // 000000004590: DBFE0000 28000015
	ds_read_b128 a[48:51], v21 offset:2048                     // 000000004598: DBFE0800 30000015
	ds_read_b128 a[56:59], v21 offset:4096                     // 0000000045A0: DBFE1000 38000015
	ds_read_b128 a[64:67], v21 offset:6144                     // 0000000045A8: DBFE1800 40000015
	ds_read_b128 a[72:75], v21 offset:8192                     // 0000000045B0: DBFE2000 48000015
	ds_read_b128 a[44:47], v21 offset:1024                     // 0000000045B8: DBFE0400 2C000015
	ds_read_b128 a[52:55], v21 offset:3072                     // 0000000045C0: DBFE0C00 34000015
	ds_read_b128 a[60:63], v21 offset:5120                     // 0000000045C8: DBFE1400 3C000015
	ds_read_b128 a[68:71], v21 offset:7168                     // 0000000045D0: DBFE1C00 44000015
	s_addk_i32 s74, 0x1                                        // 0000000045D8: B74A0001
	s_cmp_lt_i32 s74, s75                                      // 0000000045DC: BF044B4A
	s_cbranch_scc0 label_1760                                  // 0000000045E0: BF84FD1F
	s_branch label_176C                                        // 0000000045E4: BF82FD21

00000000000045e8 <label_22E8>:
	s_cmp_eq_i32 s48, 0                                        // 0000000045E8: BF008030
	s_cbranch_scc1 label_2B84                                  // 0000000045EC: BF850203

00000000000045f0 <label_22F0>:
	s_and_b32 s60, s75, 1                                      // 0000000045F0: 863C814B
	s_cmp_eq_i32 s60, 1                                        // 0000000045F4: BF00813C
	s_cbranch_scc1 label_26FC                                  // 0000000045F8: BF850100
	s_waitcnt lgkmcnt(0)                                       // 0000000045FC: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[40:47], a[0:7], 0  // 000000004600: D3AD0026 1A020128
	ds_read_b128 a[80:83], v21 offset:9216                     // 000000004608: DBFE2400 50000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[48:55], a[8:15], v[38:41]// 000000004610: D3AD0026 1C9A1130
	ds_read_b128 a[84:87], v21 offset:10240                    // 000000004618: DBFE2800 54000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[56:63], a[16:23], v[38:41]// 000000004620: D3AD0026 1C9A2138
	ds_read_b128 a[88:91], v21 offset:11264                    // 000000004628: DBFE2C00 58000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[64:71], a[24:31], v[38:41]// 000000004630: D3AD0026 1C9A3140
	ds_read_b128 a[92:95], v21 offset:12288                    // 000000004638: DBFE3000 5C000015
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[72:79], a[32:39], v[38:41]// 000000004640: D3AD0026 1C9A4148
	ds_read_b128 a[96:99], v21 offset:13312                    // 000000004648: DBFE3400 60000015
	ds_read_b128 a[100:103], v21 offset:14336                  // 000000004650: DBFE3800 64000015
	ds_read_b128 a[104:107], v21 offset:15360                  // 000000004658: DBFE3C00 68000015
	ds_read_b128 a[108:111], v21 offset:16384                  // 000000004660: DBFE4000 6C000015
	ds_read_b128 a[112:115], v21 offset:17408                  // 000000004668: DBFE4400 70000015
	s_waitcnt lgkmcnt(0)                                       // 000000004670: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[80:87], a[0:7], 0  // 000000004674: D3AD002A 1A020150
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[88:95], a[8:15], v[42:45]// 00000000467C: D3AD002A 1CAA1158
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[96:103], a[16:23], v[42:45]// 000000004684: D3AD002A 1CAA2160
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[104:111], a[24:31], v[42:45]// 00000000468C: D3AD002A 1CAA3168
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[112:119], a[32:39], v[42:45]// 000000004694: D3AD002A 1CAA4170
	s_nop 8                                                    // 00000000469C: BF800008
	s_and_b32 s60, s48, 0xff                                   // 0000000046A0: 863CFF30 000000FF
	v_mov_b32_e32 v29, s60                                     // 0000000046A8: 7E3A023C
	v_lshrrev_b32_e32 v108, 4, v0                              // 0000000046AC: 20D80084
	v_mul_i32_i24_e32 v108, 4, v108                            // 0000000046B0: 0CD8D884
	s_mov_b32 s60, 32                                          // 0000000046B4: BEBC00A0
	s_mul_i32 s60, s60, s7                                     // 0000000046B8: 923C073C
	v_add_u32_e32 v108, s60, v108                              // 0000000046BC: 68D8D83C
	v_add_u32_e32 v109, 1, v108                                // 0000000046C0: 68DAD881
	v_add_u32_e32 v110, 2, v108                                // 0000000046C4: 68DCD882
	v_add_u32_e32 v111, 3, v108                                // 0000000046C8: 68DED883
	v_mov_b32_e32 v28, 0xff800000                              // 0000000046CC: 7E3802FF FF800000
	v_cmp_lt_u32_e64 s[36:37], v108, v29                       // 0000000046D4: D0C90024 00023B6C
	v_add_u32_e32 v108, 16, v108                               // 0000000046DC: 68D8D890
	s_nop 0                                                    // 0000000046E0: BF800000
	v_cndmask_b32_e64 v38, v28, v38, s[36:37]                  // 0000000046E4: D1000026 00924D1C
	v_cmp_lt_u32_e64 s[36:37], v109, v29                       // 0000000046EC: D0C90024 00023B6D
	v_add_u32_e32 v109, 16, v109                               // 0000000046F4: 68DADA90
	s_nop 0                                                    // 0000000046F8: BF800000
	v_cndmask_b32_e64 v39, v28, v39, s[36:37]                  // 0000000046FC: D1000027 00924F1C
	v_cmp_lt_u32_e64 s[36:37], v110, v29                       // 000000004704: D0C90024 00023B6E
	v_add_u32_e32 v110, 16, v110                               // 00000000470C: 68DCDC90
	s_nop 0                                                    // 000000004710: BF800000
	v_cndmask_b32_e64 v40, v28, v40, s[36:37]                  // 000000004714: D1000028 0092511C
	v_cmp_lt_u32_e64 s[36:37], v111, v29                       // 00000000471C: D0C90024 00023B6F
	v_add_u32_e32 v111, 16, v111                               // 000000004724: 68DEDE90
	s_nop 0                                                    // 000000004728: BF800000
	v_cndmask_b32_e64 v41, v28, v41, s[36:37]                  // 00000000472C: D1000029 0092531C
	v_cmp_lt_u32_e64 s[36:37], v108, v29                       // 000000004734: D0C90024 00023B6C
	v_add_u32_e32 v108, 16, v108                               // 00000000473C: 68D8D890
	s_nop 0                                                    // 000000004740: BF800000
	v_cndmask_b32_e64 v42, v28, v42, s[36:37]                  // 000000004744: D100002A 0092551C
	v_cmp_lt_u32_e64 s[36:37], v109, v29                       // 00000000474C: D0C90024 00023B6D
	v_add_u32_e32 v109, 16, v109                               // 000000004754: 68DADA90
	s_nop 0                                                    // 000000004758: BF800000
	v_cndmask_b32_e64 v43, v28, v43, s[36:37]                  // 00000000475C: D100002B 0092571C
	v_cmp_lt_u32_e64 s[36:37], v110, v29                       // 000000004764: D0C90024 00023B6E
	v_add_u32_e32 v110, 16, v110                               // 00000000476C: 68DCDC90
	s_nop 0                                                    // 000000004770: BF800000
	v_cndmask_b32_e64 v44, v28, v44, s[36:37]                  // 000000004774: D100002C 0092591C
	v_cmp_lt_u32_e64 s[36:37], v111, v29                       // 00000000477C: D0C90024 00023B6F
	v_add_u32_e32 v111, 16, v111                               // 000000004784: 68DEDE90
	s_nop 0                                                    // 000000004788: BF800000
	v_cndmask_b32_e64 v45, v28, v45, s[36:37]                  // 00000000478C: D100002D 00925B1C
	s_nop 2                                                    // 000000004794: BF800002
	v_mov_b32_e32 v29, v38                                     // 000000004798: 7E3A0326
	v_max3_f32 v29, v38, v39, v29                              // 00000000479C: D1D3001D 04764F26
	v_max3_f32 v29, v40, v41, v29                              // 0000000047A4: D1D3001D 04765328
	v_max3_f32 v29, v42, v43, v29                              // 0000000047AC: D1D3001D 0476572A
	v_max3_f32 v29, v44, v45, v29                              // 0000000047B4: D1D3001D 04765B2C
	v_mov_b32_e32 v28, v29                                     // 0000000047BC: 7E38031D
	v_mov_b32_e32 v29, v29                                     // 0000000047C0: 7E3A031D
	s_nop 1                                                    // 0000000047C4: BF800001
	v_permlane16_swap_b32_e32 v28, v29                         // 0000000047C8: 7E38B31D
	v_mov_b32_e32 v31, v28                                     // 0000000047CC: 7E3E031C
	v_mov_b32_e32 v30, v29                                     // 0000000047D0: 7E3C031D
	s_nop 1                                                    // 0000000047D4: BF800001
	v_permlane32_swap_b32_e32 v28, v29                         // 0000000047D8: 7E38B51D
	v_permlane32_swap_b32_e32 v30, v31                         // 0000000047DC: 7E3CB51F
	v_max3_f32 v29, v28, v29, v29                              // 0000000047E0: D1D3001D 04763B1C
	v_max3_f32 v29, v30, v31, v29                              // 0000000047E8: D1D3001D 04763F1E
	ds_write_b32 v36, v29                                      // 0000000047F0: D81A0000 00001D24
	s_waitcnt lgkmcnt(0)                                       // 0000000047F8: BF8CC07F
	s_barrier                                                  // 0000000047FC: BF8A0000
	ds_read_b32 v46, v37                                       // 000000004800: D86C0000 2E000025
	ds_read_b32 v47, v37 offset:256                            // 000000004808: D86C0100 2F000025
	ds_read_b32 v48, v37 offset:512                            // 000000004810: D86C0200 30000025
	ds_read_b32 v49, v37 offset:768                            // 000000004818: D86C0300 31000025
	s_waitcnt lgkmcnt(0)                                       // 000000004820: BF8CC07F
	v_max3_f32 v29, v46, v47, v29                              // 000000004824: D1D3001D 04765F2E
	v_max3_f32 v29, v48, v49, v29                              // 00000000482C: D1D3001D 04766330
	v_mov_b32_e32 v28, 0xff800000                              // 000000004834: 7E3802FF FF800000
	v_cmp_eq_u32_e64 s[36:37], v28, v2                         // 00000000483C: D0CA0024 0002051C
	v_max_f32_e32 v29, v29, v2                                 // 000000004844: 163A051D
	v_sub_f32_e32 v18, v2, v29                                 // 000000004848: 04243B02
	v_cndmask_b32_e64 v18, v18, 0, s[36:37]                    // 00000000484C: D1000012 00910112
	v_mov_b32_e32 v2, v29                                      // 000000004854: 7E04031D
	v_mul_f32_e32 v29, s100, v29                               // 000000004858: 0A3A3A64
	v_mul_f32_e32 v18, s100, v18                               // 00000000485C: 0A242464
	v_exp_f32_e32 v18, v18                                     // 000000004860: 7E244112
	s_mov_b32 s101, s100                                       // 000000004864: BEE50064
	v_add_f32_e64 v30, 0, -v29                                 // 000000004868: D101001E 40023A80
	v_mov_b32_e32 v31, v30                                     // 000000004870: 7E3E031E
	v_pk_fma_f32 v[38:39], v[38:39], s[100:101], v[30:31]      // 000000004874: D3B04026 1C78C926
	v_pk_fma_f32 v[40:41], v[40:41], s[100:101], v[30:31]      // 00000000487C: D3B04028 1C78C928
	v_pk_fma_f32 v[42:43], v[42:43], s[100:101], v[30:31]      // 000000004884: D3B0402A 1C78C92A
	v_pk_fma_f32 v[44:45], v[44:45], s[100:101], v[30:31]      // 00000000488C: D3B0402C 1C78C92C
	v_exp_f32_e32 v38, v38                                     // 000000004894: 7E4C4126
	v_exp_f32_e32 v39, v39                                     // 000000004898: 7E4E4127
	v_exp_f32_e32 v40, v40                                     // 00000000489C: 7E504128
	v_exp_f32_e32 v41, v41                                     // 0000000048A0: 7E524129
	v_exp_f32_e32 v42, v42                                     // 0000000048A4: 7E54412A
	v_exp_f32_e32 v43, v43                                     // 0000000048A8: 7E56412B
	v_exp_f32_e32 v44, v44                                     // 0000000048AC: 7E58412C
	v_exp_f32_e32 v45, v45                                     // 0000000048B0: 7E5A412D
	v_mul_f32_e32 v4, v18, v4                                  // 0000000048B4: 0A080912
	v_mov_b32_e32 v28, v38                                     // 0000000048B8: 7E380326
	v_add_f32_e32 v28, v39, v28                                // 0000000048BC: 02383927
	v_add_f32_e32 v28, v40, v28                                // 0000000048C0: 02383928
	v_add_f32_e32 v28, v41, v28                                // 0000000048C4: 02383929
	v_add_f32_e32 v28, v42, v28                                // 0000000048C8: 0238392A
	v_add_f32_e32 v28, v43, v28                                // 0000000048CC: 0238392B
	v_add_f32_e32 v28, v44, v28                                // 0000000048D0: 0238392C
	v_add_f32_e32 v28, v45, v28                                // 0000000048D4: 0238392D
	v_add_f32_e32 v4, v28, v4                                  // 0000000048D8: 0208091C
	v_cvt_pk_fp8_f32 v38, v38, v39                             // 0000000048DC: D2A20026 00024F26
	v_cvt_pk_fp8_f32 v38, v40, v41 op_sel:[0,0,1]              // 0000000048E4: D2A24026 00025328
	v_cvt_pk_fp8_f32 v39, v42, v43                             // 0000000048EC: D2A20027 0002572A
	v_cvt_pk_fp8_f32 v39, v44, v45 op_sel:[0,0,1]              // 0000000048F4: D2A24027 00025B2C
	s_nop 0                                                    // 0000000048FC: BF800000
	v_permlane16_swap_b32_e32 v38, v39                         // 000000004900: 7E4CB327
	ds_write_b64 v34, v[38:39]                                 // 000000004904: D89A0000 00002622
	s_waitcnt lgkmcnt(0)                                       // 00000000490C: BF8CC07F
	s_barrier                                                  // 000000004910: BF8A0000
	ds_read_b64 v[38:39], v35                                  // 000000004914: D8EC0000 26000023
	ds_read_b64 v[40:41], v35 offset:256                       // 00000000491C: D8EC0100 28000023
	ds_read_b64 v[42:43], v35 offset:1024                      // 000000004924: D8EC0400 2A000023
	ds_read_b64 v[44:45], v35 offset:1280                      // 00000000492C: D8EC0500 2C000023
	v_mul_f32_e32 v74, v18, v74                                // 000000004934: 0A949512
	v_mul_f32_e32 v75, v18, v75                                // 000000004938: 0A969712
	v_mul_f32_e32 v76, v18, v76                                // 00000000493C: 0A989912
	v_mul_f32_e32 v77, v18, v77                                // 000000004940: 0A9A9B12
	v_mul_f32_e32 v78, v18, v78                                // 000000004944: 0A9C9D12
	v_mul_f32_e32 v79, v18, v79                                // 000000004948: 0A9E9F12
	v_mul_f32_e32 v80, v18, v80                                // 00000000494C: 0AA0A112
	v_mul_f32_e32 v81, v18, v81                                // 000000004950: 0AA2A312
	v_mul_f32_e32 v82, v18, v82                                // 000000004954: 0AA4A512
	v_mul_f32_e32 v83, v18, v83                                // 000000004958: 0AA6A712
	v_mul_f32_e32 v84, v18, v84                                // 00000000495C: 0AA8A912
	v_mul_f32_e32 v85, v18, v85                                // 000000004960: 0AAAAB12
	v_mul_f32_e32 v86, v18, v86                                // 000000004964: 0AACAD12
	v_mul_f32_e32 v87, v18, v87                                // 000000004968: 0AAEAF12
	v_mul_f32_e32 v88, v18, v88                                // 00000000496C: 0AB0B112
	v_mul_f32_e32 v89, v18, v89                                // 000000004970: 0AB2B312
	v_mul_f32_e32 v90, v18, v90                                // 000000004974: 0AB4B512
	v_mul_f32_e32 v91, v18, v91                                // 000000004978: 0AB6B712
	v_mul_f32_e32 v92, v18, v92                                // 00000000497C: 0AB8B912
	v_mul_f32_e32 v93, v18, v93                                // 000000004980: 0ABABB12
	v_mul_f32_e32 v94, v18, v94                                // 000000004984: 0ABCBD12
	v_mul_f32_e32 v95, v18, v95                                // 000000004988: 0ABEBF12
	v_mul_f32_e32 v96, v18, v96                                // 00000000498C: 0AC0C112
	v_mul_f32_e32 v97, v18, v97                                // 000000004990: 0AC2C312
	v_mul_f32_e32 v98, v18, v98                                // 000000004994: 0AC4C512
	v_mul_f32_e32 v99, v18, v99                                // 000000004998: 0AC6C712
	v_mul_f32_e32 v100, v18, v100                              // 00000000499C: 0AC8C912
	v_mul_f32_e32 v101, v18, v101                              // 0000000049A0: 0ACACB12
	v_mul_f32_e32 v102, v18, v102                              // 0000000049A4: 0ACCCD12
	v_mul_f32_e32 v103, v18, v103                              // 0000000049A8: 0ACECF12
	v_mul_f32_e32 v104, v18, v104                              // 0000000049AC: 0AD0D112
	v_mul_f32_e32 v105, v18, v105                              // 0000000049B0: 0AD2D312
	s_waitcnt lgkmcnt(0)                                       // 0000000049B4: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], a[120:127], v[38:45], v[74:77]// 0000000049B8: D3AD004A 0D2A4D78
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], a[128:135], v[38:45], v[78:81]// 0000000049C0: D3AD004E 0D3A4D80
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], a[136:143], v[38:45], v[82:85]// 0000000049C8: D3AD0052 0D4A4D88
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], a[144:151], v[38:45], v[86:89]// 0000000049D0: D3AD0056 0D5A4D90
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], a[152:159], v[38:45], v[90:93]// 0000000049D8: D3AD005A 0D6A4D98
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], a[160:167], v[38:45], v[94:97]// 0000000049E0: D3AD005E 0D7A4DA0
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], a[168:175], v[38:45], v[98:101]// 0000000049E8: D3AD0062 0D8A4DA8
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], a[176:183], v[38:45], v[102:105]// 0000000049F0: D3AD0066 0D9A4DB0
	s_branch label_2B84                                        // 0000000049F8: BF820100

00000000000049fc <label_26FC>:
	s_waitcnt lgkmcnt(0)                                       // 0000000049FC: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[40:47], a[0:7], 0  // 000000004A00: D3AD0026 1A020128
	ds_read_b128 a[80:83], v20 offset:9216                     // 000000004A08: DBFE2400 50000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[48:55], a[8:15], v[38:41]// 000000004A10: D3AD0026 1C9A1130
	ds_read_b128 a[84:87], v20 offset:10240                    // 000000004A18: DBFE2800 54000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[56:63], a[16:23], v[38:41]// 000000004A20: D3AD0026 1C9A2138
	ds_read_b128 a[88:91], v20 offset:11264                    // 000000004A28: DBFE2C00 58000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[64:71], a[24:31], v[38:41]// 000000004A30: D3AD0026 1C9A3140
	ds_read_b128 a[92:95], v20 offset:12288                    // 000000004A38: DBFE3000 5C000014
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[72:79], a[32:39], v[38:41]// 000000004A40: D3AD0026 1C9A4148
	ds_read_b128 a[96:99], v20 offset:13312                    // 000000004A48: DBFE3400 60000014
	ds_read_b128 a[100:103], v20 offset:14336                  // 000000004A50: DBFE3800 64000014
	ds_read_b128 a[104:107], v20 offset:15360                  // 000000004A58: DBFE3C00 68000014
	ds_read_b128 a[108:111], v20 offset:16384                  // 000000004A60: DBFE4000 6C000014
	ds_read_b128 a[112:115], v20 offset:17408                  // 000000004A68: DBFE4400 70000014
	s_waitcnt lgkmcnt(0)                                       // 000000004A70: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[80:87], a[0:7], 0  // 000000004A74: D3AD002A 1A020150
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[88:95], a[8:15], v[42:45]// 000000004A7C: D3AD002A 1CAA1158
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[96:103], a[16:23], v[42:45]// 000000004A84: D3AD002A 1CAA2160
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[104:111], a[24:31], v[42:45]// 000000004A8C: D3AD002A 1CAA3168
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], a[112:119], a[32:39], v[42:45]// 000000004A94: D3AD002A 1CAA4170
	s_nop 8                                                    // 000000004A9C: BF800008
	s_and_b32 s60, s48, 0xff                                   // 000000004AA0: 863CFF30 000000FF
	v_mov_b32_e32 v29, s60                                     // 000000004AA8: 7E3A023C
	v_lshrrev_b32_e32 v108, 4, v0                              // 000000004AAC: 20D80084
	v_mul_i32_i24_e32 v108, 4, v108                            // 000000004AB0: 0CD8D884
	s_mov_b32 s60, 32                                          // 000000004AB4: BEBC00A0
	s_mul_i32 s60, s60, s7                                     // 000000004AB8: 923C073C
	v_add_u32_e32 v108, s60, v108                              // 000000004ABC: 68D8D83C
	v_add_u32_e32 v109, 1, v108                                // 000000004AC0: 68DAD881
	v_add_u32_e32 v110, 2, v108                                // 000000004AC4: 68DCD882
	v_add_u32_e32 v111, 3, v108                                // 000000004AC8: 68DED883
	v_mov_b32_e32 v28, 0xff800000                              // 000000004ACC: 7E3802FF FF800000
	v_cmp_lt_u32_e64 s[36:37], v108, v29                       // 000000004AD4: D0C90024 00023B6C
	v_add_u32_e32 v108, 16, v108                               // 000000004ADC: 68D8D890
	s_nop 0                                                    // 000000004AE0: BF800000
	v_cndmask_b32_e64 v38, v28, v38, s[36:37]                  // 000000004AE4: D1000026 00924D1C
	v_cmp_lt_u32_e64 s[36:37], v109, v29                       // 000000004AEC: D0C90024 00023B6D
	v_add_u32_e32 v109, 16, v109                               // 000000004AF4: 68DADA90
	s_nop 0                                                    // 000000004AF8: BF800000
	v_cndmask_b32_e64 v39, v28, v39, s[36:37]                  // 000000004AFC: D1000027 00924F1C
	v_cmp_lt_u32_e64 s[36:37], v110, v29                       // 000000004B04: D0C90024 00023B6E
	v_add_u32_e32 v110, 16, v110                               // 000000004B0C: 68DCDC90
	s_nop 0                                                    // 000000004B10: BF800000
	v_cndmask_b32_e64 v40, v28, v40, s[36:37]                  // 000000004B14: D1000028 0092511C
	v_cmp_lt_u32_e64 s[36:37], v111, v29                       // 000000004B1C: D0C90024 00023B6F
	v_add_u32_e32 v111, 16, v111                               // 000000004B24: 68DEDE90
	s_nop 0                                                    // 000000004B28: BF800000
	v_cndmask_b32_e64 v41, v28, v41, s[36:37]                  // 000000004B2C: D1000029 0092531C
	v_cmp_lt_u32_e64 s[36:37], v108, v29                       // 000000004B34: D0C90024 00023B6C
	v_add_u32_e32 v108, 16, v108                               // 000000004B3C: 68D8D890
	s_nop 0                                                    // 000000004B40: BF800000
	v_cndmask_b32_e64 v42, v28, v42, s[36:37]                  // 000000004B44: D100002A 0092551C
	v_cmp_lt_u32_e64 s[36:37], v109, v29                       // 000000004B4C: D0C90024 00023B6D
	v_add_u32_e32 v109, 16, v109                               // 000000004B54: 68DADA90
	s_nop 0                                                    // 000000004B58: BF800000
	v_cndmask_b32_e64 v43, v28, v43, s[36:37]                  // 000000004B5C: D100002B 0092571C
	v_cmp_lt_u32_e64 s[36:37], v110, v29                       // 000000004B64: D0C90024 00023B6E
	v_add_u32_e32 v110, 16, v110                               // 000000004B6C: 68DCDC90
	s_nop 0                                                    // 000000004B70: BF800000
	v_cndmask_b32_e64 v44, v28, v44, s[36:37]                  // 000000004B74: D100002C 0092591C
	v_cmp_lt_u32_e64 s[36:37], v111, v29                       // 000000004B7C: D0C90024 00023B6F
	v_add_u32_e32 v111, 16, v111                               // 000000004B84: 68DEDE90
	s_nop 0                                                    // 000000004B88: BF800000
	v_cndmask_b32_e64 v45, v28, v45, s[36:37]                  // 000000004B8C: D100002D 00925B1C
	s_nop 2                                                    // 000000004B94: BF800002
	v_mov_b32_e32 v29, v38                                     // 000000004B98: 7E3A0326
	v_max3_f32 v29, v38, v39, v29                              // 000000004B9C: D1D3001D 04764F26
	v_max3_f32 v29, v40, v41, v29                              // 000000004BA4: D1D3001D 04765328
	v_max3_f32 v29, v42, v43, v29                              // 000000004BAC: D1D3001D 0476572A
	v_max3_f32 v29, v44, v45, v29                              // 000000004BB4: D1D3001D 04765B2C
	v_mov_b32_e32 v28, v29                                     // 000000004BBC: 7E38031D
	v_mov_b32_e32 v29, v29                                     // 000000004BC0: 7E3A031D
	s_nop 1                                                    // 000000004BC4: BF800001
	v_permlane16_swap_b32_e32 v28, v29                         // 000000004BC8: 7E38B31D
	v_mov_b32_e32 v31, v28                                     // 000000004BCC: 7E3E031C
	v_mov_b32_e32 v30, v29                                     // 000000004BD0: 7E3C031D
	s_nop 1                                                    // 000000004BD4: BF800001
	v_permlane32_swap_b32_e32 v28, v29                         // 000000004BD8: 7E38B51D
	v_permlane32_swap_b32_e32 v30, v31                         // 000000004BDC: 7E3CB51F
	v_max3_f32 v29, v28, v29, v29                              // 000000004BE0: D1D3001D 04763B1C
	v_max3_f32 v29, v30, v31, v29                              // 000000004BE8: D1D3001D 04763F1E
	ds_write_b32 v36, v29                                      // 000000004BF0: D81A0000 00001D24
	s_waitcnt lgkmcnt(0)                                       // 000000004BF8: BF8CC07F
	s_barrier                                                  // 000000004BFC: BF8A0000
	ds_read_b32 v46, v37                                       // 000000004C00: D86C0000 2E000025
	ds_read_b32 v47, v37 offset:256                            // 000000004C08: D86C0100 2F000025
	ds_read_b32 v48, v37 offset:512                            // 000000004C10: D86C0200 30000025
	ds_read_b32 v49, v37 offset:768                            // 000000004C18: D86C0300 31000025
	s_waitcnt lgkmcnt(0)                                       // 000000004C20: BF8CC07F
	v_max3_f32 v29, v46, v47, v29                              // 000000004C24: D1D3001D 04765F2E
	v_max3_f32 v29, v48, v49, v29                              // 000000004C2C: D1D3001D 04766330
	v_mov_b32_e32 v28, 0xff800000                              // 000000004C34: 7E3802FF FF800000
	v_cmp_eq_u32_e64 s[36:37], v28, v2                         // 000000004C3C: D0CA0024 0002051C
	v_max_f32_e32 v29, v29, v2                                 // 000000004C44: 163A051D
	v_sub_f32_e32 v18, v2, v29                                 // 000000004C48: 04243B02
	v_cndmask_b32_e64 v18, v18, 0, s[36:37]                    // 000000004C4C: D1000012 00910112
	v_mov_b32_e32 v2, v29                                      // 000000004C54: 7E04031D
	v_mul_f32_e32 v29, s100, v29                               // 000000004C58: 0A3A3A64
	v_mul_f32_e32 v18, s100, v18                               // 000000004C5C: 0A242464
	v_exp_f32_e32 v18, v18                                     // 000000004C60: 7E244112
	s_mov_b32 s101, s100                                       // 000000004C64: BEE50064
	v_add_f32_e64 v30, 0, -v29                                 // 000000004C68: D101001E 40023A80
	v_mov_b32_e32 v31, v30                                     // 000000004C70: 7E3E031E
	v_pk_fma_f32 v[38:39], v[38:39], s[100:101], v[30:31]      // 000000004C74: D3B04026 1C78C926
	v_pk_fma_f32 v[40:41], v[40:41], s[100:101], v[30:31]      // 000000004C7C: D3B04028 1C78C928
	v_pk_fma_f32 v[42:43], v[42:43], s[100:101], v[30:31]      // 000000004C84: D3B0402A 1C78C92A
	v_pk_fma_f32 v[44:45], v[44:45], s[100:101], v[30:31]      // 000000004C8C: D3B0402C 1C78C92C
	v_exp_f32_e32 v38, v38                                     // 000000004C94: 7E4C4126
	v_exp_f32_e32 v39, v39                                     // 000000004C98: 7E4E4127
	v_exp_f32_e32 v40, v40                                     // 000000004C9C: 7E504128
	v_exp_f32_e32 v41, v41                                     // 000000004CA0: 7E524129
	v_exp_f32_e32 v42, v42                                     // 000000004CA4: 7E54412A
	v_exp_f32_e32 v43, v43                                     // 000000004CA8: 7E56412B
	v_exp_f32_e32 v44, v44                                     // 000000004CAC: 7E58412C
	v_exp_f32_e32 v45, v45                                     // 000000004CB0: 7E5A412D
	v_mul_f32_e32 v4, v18, v4                                  // 000000004CB4: 0A080912
	v_mov_b32_e32 v28, v38                                     // 000000004CB8: 7E380326
	v_add_f32_e32 v28, v39, v28                                // 000000004CBC: 02383927
	v_add_f32_e32 v28, v40, v28                                // 000000004CC0: 02383928
	v_add_f32_e32 v28, v41, v28                                // 000000004CC4: 02383929
	v_add_f32_e32 v28, v42, v28                                // 000000004CC8: 0238392A
	v_add_f32_e32 v28, v43, v28                                // 000000004CCC: 0238392B
	v_add_f32_e32 v28, v44, v28                                // 000000004CD0: 0238392C
	v_add_f32_e32 v28, v45, v28                                // 000000004CD4: 0238392D
	v_add_f32_e32 v4, v28, v4                                  // 000000004CD8: 0208091C
	v_cvt_pk_fp8_f32 v38, v38, v39                             // 000000004CDC: D2A20026 00024F26
	v_cvt_pk_fp8_f32 v38, v40, v41 op_sel:[0,0,1]              // 000000004CE4: D2A24026 00025328
	v_cvt_pk_fp8_f32 v39, v42, v43                             // 000000004CEC: D2A20027 0002572A
	v_cvt_pk_fp8_f32 v39, v44, v45 op_sel:[0,0,1]              // 000000004CF4: D2A24027 00025B2C
	s_nop 0                                                    // 000000004CFC: BF800000
	v_permlane16_swap_b32_e32 v38, v39                         // 000000004D00: 7E4CB327
	ds_write_b64 v34, v[38:39]                                 // 000000004D04: D89A0000 00002622
	s_waitcnt lgkmcnt(0)                                       // 000000004D0C: BF8CC07F
	s_barrier                                                  // 000000004D10: BF8A0000
	ds_read_b64 v[38:39], v35                                  // 000000004D14: D8EC0000 26000023
	ds_read_b64 v[40:41], v35 offset:256                       // 000000004D1C: D8EC0100 28000023
	ds_read_b64 v[42:43], v35 offset:1024                      // 000000004D24: D8EC0400 2A000023
	ds_read_b64 v[44:45], v35 offset:1280                      // 000000004D2C: D8EC0500 2C000023
	v_mul_f32_e32 v74, v18, v74                                // 000000004D34: 0A949512
	v_mul_f32_e32 v75, v18, v75                                // 000000004D38: 0A969712
	v_mul_f32_e32 v76, v18, v76                                // 000000004D3C: 0A989912
	v_mul_f32_e32 v77, v18, v77                                // 000000004D40: 0A9A9B12
	v_mul_f32_e32 v78, v18, v78                                // 000000004D44: 0A9C9D12
	v_mul_f32_e32 v79, v18, v79                                // 000000004D48: 0A9E9F12
	v_mul_f32_e32 v80, v18, v80                                // 000000004D4C: 0AA0A112
	v_mul_f32_e32 v81, v18, v81                                // 000000004D50: 0AA2A312
	v_mul_f32_e32 v82, v18, v82                                // 000000004D54: 0AA4A512
	v_mul_f32_e32 v83, v18, v83                                // 000000004D58: 0AA6A712
	v_mul_f32_e32 v84, v18, v84                                // 000000004D5C: 0AA8A912
	v_mul_f32_e32 v85, v18, v85                                // 000000004D60: 0AAAAB12
	v_mul_f32_e32 v86, v18, v86                                // 000000004D64: 0AACAD12
	v_mul_f32_e32 v87, v18, v87                                // 000000004D68: 0AAEAF12
	v_mul_f32_e32 v88, v18, v88                                // 000000004D6C: 0AB0B112
	v_mul_f32_e32 v89, v18, v89                                // 000000004D70: 0AB2B312
	v_mul_f32_e32 v90, v18, v90                                // 000000004D74: 0AB4B512
	v_mul_f32_e32 v91, v18, v91                                // 000000004D78: 0AB6B712
	v_mul_f32_e32 v92, v18, v92                                // 000000004D7C: 0AB8B912
	v_mul_f32_e32 v93, v18, v93                                // 000000004D80: 0ABABB12
	v_mul_f32_e32 v94, v18, v94                                // 000000004D84: 0ABCBD12
	v_mul_f32_e32 v95, v18, v95                                // 000000004D88: 0ABEBF12
	v_mul_f32_e32 v96, v18, v96                                // 000000004D8C: 0AC0C112
	v_mul_f32_e32 v97, v18, v97                                // 000000004D90: 0AC2C312
	v_mul_f32_e32 v98, v18, v98                                // 000000004D94: 0AC4C512
	v_mul_f32_e32 v99, v18, v99                                // 000000004D98: 0AC6C712
	v_mul_f32_e32 v100, v18, v100                              // 000000004D9C: 0AC8C912
	v_mul_f32_e32 v101, v18, v101                              // 000000004DA0: 0ACACB12
	v_mul_f32_e32 v102, v18, v102                              // 000000004DA4: 0ACCCD12
	v_mul_f32_e32 v103, v18, v103                              // 000000004DA8: 0ACECF12
	v_mul_f32_e32 v104, v18, v104                              // 000000004DAC: 0AD0D112
	v_mul_f32_e32 v105, v18, v105                              // 000000004DB0: 0AD2D312
	s_waitcnt lgkmcnt(0)                                       // 000000004DB4: BF8CC07F
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], a[120:127], v[38:45], v[74:77]// 000000004DB8: D3AD004A 0D2A4D78
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], a[128:135], v[38:45], v[78:81]// 000000004DC0: D3AD004E 0D3A4D80
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], a[136:143], v[38:45], v[82:85]// 000000004DC8: D3AD0052 0D4A4D88
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], a[144:151], v[38:45], v[86:89]// 000000004DD0: D3AD0056 0D5A4D90
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], a[152:159], v[38:45], v[90:93]// 000000004DD8: D3AD005A 0D6A4D98
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], a[160:167], v[38:45], v[94:97]// 000000004DE0: D3AD005E 0D7A4DA0
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], a[168:175], v[38:45], v[98:101]// 000000004DE8: D3AD0062 0D8A4DA8
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], a[176:183], v[38:45], v[102:105]// 000000004DF0: D3AD0066 0D9A4DB0
	s_branch label_2B84                                        // 000000004DF8: BF820000

0000000000004dfc <label_2B84>:
	v_mov_b32_e32 v28, v4                                      // 000000004DFC: 7E380304
	v_mov_b32_e32 v29, v4                                      // 000000004E00: 7E3A0304
	s_nop 1                                                    // 000000004E04: BF800001
	v_permlane16_swap_b32_e32 v28, v29                         // 000000004E08: 7E38B31D
	v_mov_b32_e32 v31, v28                                     // 000000004E0C: 7E3E031C
	v_mov_b32_e32 v30, v29                                     // 000000004E10: 7E3C031D
	s_nop 1                                                    // 000000004E14: BF800001
	v_permlane32_swap_b32_e32 v28, v29                         // 000000004E18: 7E38B51D
	v_permlane32_swap_b32_e32 v30, v31                         // 000000004E1C: 7E3CB51F
	v_mov_b32_e32 v4, 0                                        // 000000004E20: 7E080280
	v_add_f32_e32 v4, v28, v4                                  // 000000004E24: 0208091C
	v_add_f32_e32 v4, v29, v4                                  // 000000004E28: 0208091D
	v_add_f32_e32 v4, v30, v4                                  // 000000004E2C: 0208091E
	v_add_f32_e32 v4, v31, v4                                  // 000000004E30: 0208091F
	ds_write_b32 v36, v4                                       // 000000004E34: D81A0000 00000424
	s_waitcnt lgkmcnt(0)                                       // 000000004E3C: BF8CC07F
	s_barrier                                                  // 000000004E40: BF8A0000
	ds_read_b32 v46, v37                                       // 000000004E44: D86C0000 2E000025
	ds_read_b32 v47, v37 offset:256                            // 000000004E4C: D86C0100 2F000025
	ds_read_b32 v48, v37 offset:512                            // 000000004E54: D86C0200 30000025
	ds_read_b32 v49, v37 offset:768                            // 000000004E5C: D86C0300 31000025
	v_mov_b32_e32 v29, 0                                       // 000000004E64: 7E3A0280
	s_waitcnt lgkmcnt(0)                                       // 000000004E68: BF8CC07F
	v_add_f32_e32 v29, v46, v29                                // 000000004E6C: 023A3B2E
	v_add_f32_e32 v29, v47, v29                                // 000000004E70: 023A3B2F
	v_add_f32_e32 v29, v48, v29                                // 000000004E74: 023A3B30
	v_add_f32_e32 v29, v49, v29                                // 000000004E78: 023A3B31
	v_mov_b32_e32 v4, v29                                      // 000000004E7C: 7E08031D
	v_mov_b32_e32 v28, 0                                       // 000000004E80: 7E380280
	v_cmp_eq_u32_e64 s[36:37], v28, v4                         // 000000004E84: D0CA0024 0002091C
	v_mul_f32_e64 v28, v2, s68                                 // 000000004E8C: D105001C 00008902
	v_log_f32_e32 v29, v4                                      // 000000004E94: 7E3A4304
	s_nop 1                                                    // 000000004E98: BF800001
	v_rcp_f32_e32 v4, v4                                       // 000000004E9C: 7E084504
	s_nop 1                                                    // 000000004EA0: BF800001
	v_fma_f32 v1, v29, s67, v28                                // 000000004EA4: D1CB0001 0470871D
	v_mul_f32_e32 v4, s65, v4                                  // 000000004EAC: 0A080841
	v_mul_f32_e32 v74, v4, v74                                 // 000000004EB0: 0A949504
	v_mul_f32_e32 v75, v4, v75                                 // 000000004EB4: 0A969704
	v_mul_f32_e32 v76, v4, v76                                 // 000000004EB8: 0A989904
	v_mul_f32_e32 v77, v4, v77                                 // 000000004EBC: 0A9A9B04
	v_mul_f32_e32 v78, v4, v78                                 // 000000004EC0: 0A9C9D04
	v_mul_f32_e32 v79, v4, v79                                 // 000000004EC4: 0A9E9F04
	v_mul_f32_e32 v80, v4, v80                                 // 000000004EC8: 0AA0A104
	v_mul_f32_e32 v81, v4, v81                                 // 000000004ECC: 0AA2A304
	v_mul_f32_e32 v82, v4, v82                                 // 000000004ED0: 0AA4A504
	v_mul_f32_e32 v83, v4, v83                                 // 000000004ED4: 0AA6A704
	v_mul_f32_e32 v84, v4, v84                                 // 000000004ED8: 0AA8A904
	v_mul_f32_e32 v85, v4, v85                                 // 000000004EDC: 0AAAAB04
	v_mul_f32_e32 v86, v4, v86                                 // 000000004EE0: 0AACAD04
	v_mul_f32_e32 v87, v4, v87                                 // 000000004EE4: 0AAEAF04
	v_mul_f32_e32 v88, v4, v88                                 // 000000004EE8: 0AB0B104
	v_mul_f32_e32 v89, v4, v89                                 // 000000004EEC: 0AB2B304
	v_mul_f32_e32 v90, v4, v90                                 // 000000004EF0: 0AB4B504
	v_mul_f32_e32 v91, v4, v91                                 // 000000004EF4: 0AB6B704
	v_mul_f32_e32 v92, v4, v92                                 // 000000004EF8: 0AB8B904
	v_mul_f32_e32 v93, v4, v93                                 // 000000004EFC: 0ABABB04
	v_mul_f32_e32 v94, v4, v94                                 // 000000004F00: 0ABCBD04
	v_mul_f32_e32 v95, v4, v95                                 // 000000004F04: 0ABEBF04
	v_mul_f32_e32 v96, v4, v96                                 // 000000004F08: 0AC0C104
	v_mul_f32_e32 v97, v4, v97                                 // 000000004F0C: 0AC2C304
	v_mul_f32_e32 v98, v4, v98                                 // 000000004F10: 0AC4C504
	v_mul_f32_e32 v99, v4, v99                                 // 000000004F14: 0AC6C704
	v_mul_f32_e32 v100, v4, v100                               // 000000004F18: 0AC8C904
	v_mul_f32_e32 v101, v4, v101                               // 000000004F1C: 0ACACB04
	v_mul_f32_e32 v102, v4, v102                               // 000000004F20: 0ACCCD04
	v_mul_f32_e32 v103, v4, v103                               // 000000004F24: 0ACECF04
	v_mul_f32_e32 v104, v4, v104                               // 000000004F28: 0AD0D104
	v_mul_f32_e32 v105, v4, v105                               // 000000004F2C: 0AD2D304
	s_cmp_lt_i32 s91, 0                                        // 000000004F30: BF04805B
	s_cbranch_scc1 label_2F4C                                  // 000000004F34: BF8500A3
	s_mul_i32 s79, 0x800, 16                                   // 000000004F38: 924F90FF 00000800
	s_mul_i32 s60, s91, s79                                    // 000000004F40: 923C4F5B
	s_add_u32 s8, s60, s8                                      // 000000004F44: 8008083C
	s_addc_u32 s9, 0, s9                                       // 000000004F48: 82090980
	s_mul_i32 s60, s79, s85                                    // 000000004F4C: 923C554F
	s_mov_b32 s10, s60                                         // 000000004F50: BE8A003C
	s_mul_i32 s60, 0x200, s7                                   // 000000004F54: 923C07FF 00000200
	v_lshrrev_b32_e32 v28, 5, v0                               // 000000004F5C: 20380085
	s_mov_b32 s61, 0x800                                       // 000000004F60: BEBD00FF 00000800
	v_mul_i32_i24_e32 v28, s61, v28                            // 000000004F68: 0C38383D
	v_and_b32_e32 v5, 31, v0                                   // 000000004F6C: 260A009F
	v_lshlrev_b32_e32 v5, 4, v5                                // 000000004F70: 240A0A84
	v_add_u32_e32 v5, v5, v28                                  // 000000004F74: 680A3905
	v_add_u32_e64 v5, v5, s60                                  // 000000004F78: D1340005 00007905
	s_mul_i32 s61, 4, 16                                       // 000000004F80: 923D9084
	s_mul_i32 s60, s91, s61                                    // 000000004F84: 923C3D5B
	s_add_u32 s12, s60, s12                                    // 000000004F88: 800C0C3C
	s_addc_u32 s13, 0, s13                                     // 000000004F8C: 820D0D80
	s_mul_i32 s62, s61, s85                                    // 000000004F90: 923E553D
	s_mov_b32 s14, s62                                         // 000000004F94: BE8E003E
	v_and_b32_e32 v4, 15, v0                                   // 000000004F98: 2608008F
	v_lshlrev_b32_e32 v4, 2, v4                                // 000000004F9C: 24080882
	s_mul_i32 s60, s61, s7                                     // 000000004FA0: 923C073D
	v_add_u32_e64 v4, v4, s60                                  // 000000004FA4: D1340004 00007904
	v_lshlrev_b32_e32 v2, 4, v0                                // 000000004FAC: 24040084
	s_mov_b32 s60, 0x2400                                      // 000000004FB0: BEBC00FF 00002400
	s_mul_i32 s60, s7, s60                                     // 000000004FB8: 923C3C07
	v_add_u32_e32 v2, s60, v2                                  // 000000004FBC: 6804043C
	ds_write_b128 v2, v[74:77]                                 // 000000004FC0: D9BE0000 00004A02
	s_mov_b32 s60, 0x410                                       // 000000004FC8: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 000000004FD0: 6804043C
	ds_write_b128 v2, v[78:81]                                 // 000000004FD4: D9BE0000 00004E02
	s_mov_b32 s60, 0x410                                       // 000000004FDC: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 000000004FE4: 6804043C
	ds_write_b128 v2, v[82:85]                                 // 000000004FE8: D9BE0000 00005202
	s_mov_b32 s60, 0x410                                       // 000000004FF0: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 000000004FF8: 6804043C
	ds_write_b128 v2, v[86:89]                                 // 000000004FFC: D9BE0000 00005602
	s_mov_b32 s60, 0x410                                       // 000000005004: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 00000000500C: 6804043C
	ds_write_b128 v2, v[90:93]                                 // 000000005010: D9BE0000 00005A02
	s_mov_b32 s60, 0x410                                       // 000000005018: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 000000005020: 6804043C
	ds_write_b128 v2, v[94:97]                                 // 000000005024: D9BE0000 00005E02
	s_mov_b32 s60, 0x410                                       // 00000000502C: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 000000005034: 6804043C
	ds_write_b128 v2, v[98:101]                                // 000000005038: D9BE0000 00006202
	s_mov_b32 s60, 0x410                                       // 000000005040: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 000000005048: 6804043C
	ds_write_b128 v2, v[102:105]                               // 00000000504C: D9BE0000 00006602
	s_mov_b32 s60, 0x410                                       // 000000005054: BEBC00FF 00000410
	v_add_u32_e32 v2, s60, v2                                  // 00000000505C: 6804043C
	v_and_b32_e32 v28, 3, v0                                   // 000000005060: 26380083
	s_mov_b32 s60, 0x100                                       // 000000005064: BEBC00FF 00000100
	v_mul_i32_i24_e32 v28, s60, v28                            // 00000000506C: 0C38383C
	v_and_b32_e32 v29, 31, v0                                  // 000000005070: 263A009F
	v_lshrrev_b32_e32 v29, 2, v29                              // 000000005074: 203A3A82
	s_mov_b32 s60, 0x410                                       // 000000005078: BEBC00FF 00000410
	v_mul_i32_i24_e32 v29, s60, v29                            // 000000005080: 0C3A3A3C
	v_lshrrev_b32_e32 v30, 5, v0                               // 000000005084: 203C0085
	s_mov_b32 s60, 16                                          // 000000005088: BEBC0090
	v_mul_i32_i24_e32 v30, s60, v30                            // 00000000508C: 0C3C3C3C
	v_add_u32_e32 v2, v28, v29                                 // 000000005090: 68043B1C
	v_add_u32_e32 v2, v30, v2                                  // 000000005094: 6804051E
	s_mov_b32 s61, 0x2400                                      // 000000005098: BEBD00FF 00002400
	s_mul_i32 s60, s61, s7                                     // 0000000050A0: 923C073D
	v_add_u32_e32 v2, s60, v2                                  // 0000000050A4: 6804043C
	v_and_b32_e32 v5, 31, v0                                   // 0000000050A8: 260A009F
	v_lshlrev_b32_e32 v5, 4, v5                                // 0000000050AC: 240A0A84
	v_lshrrev_b32_e32 v28, 5, v0                               // 0000000050B0: 20380085
	s_mov_b32 s60, 0x800                                       // 0000000050B4: BEBC00FF 00000800
	v_mul_i32_i24_e32 v28, s60, v28                            // 0000000050BC: 0C38383C
	v_add_u32_e32 v5, v28, v5                                  // 0000000050C0: 680A0B1C
	s_mov_b32 s61, 0x200                                       // 0000000050C4: BEBD00FF 00000200
	s_mul_i32 s60, s7, s61                                     // 0000000050CC: 923C3D07
	v_add_u32_e32 v5, s60, v5                                  // 0000000050D0: 680A0A3C
	s_waitcnt lgkmcnt(0)                                       // 0000000050D4: BF8CC07F
	ds_read_b128 v[74:77], v2                                  // 0000000050D8: D9FE0000 4A000002
	ds_read_b128 v[78:81], v2 offset:32                        // 0000000050E0: D9FE0020 4E000002
	s_waitcnt lgkmcnt(1)                                       // 0000000050E8: BF8CC17F
	buffer_store_dwordx4 v[74:77], v5, s[8:11], 0 offen        // 0000000050EC: E07C1000 80024A05
	v_add_u32_e32 v5, 0x1000, v5                               // 0000000050F4: 680A0AFF 00001000
	ds_read_b128 v[82:85], v2 offset:64                        // 0000000050FC: D9FE0040 52000002
	s_waitcnt lgkmcnt(1)                                       // 000000005104: BF8CC17F
	buffer_store_dwordx4 v[78:81], v5, s[8:11], 0 offen        // 000000005108: E07C1000 80024E05
	v_add_u32_e32 v5, 0x1000, v5                               // 000000005110: 680A0AFF 00001000
	ds_read_b128 v[86:89], v2 offset:96                        // 000000005118: D9FE0060 56000002
	s_waitcnt lgkmcnt(1)                                       // 000000005120: BF8CC17F
	buffer_store_dwordx4 v[82:85], v5, s[8:11], 0 offen        // 000000005124: E07C1000 80025205
	v_add_u32_e32 v5, 0x1000, v5                               // 00000000512C: 680A0AFF 00001000
	s_waitcnt lgkmcnt(0)                                       // 000000005134: BF8CC07F
	buffer_store_dwordx4 v[86:89], v5, s[8:11], 0 offen        // 000000005138: E07C1000 80025605
	v_add_u32_e32 v5, 0x1000, v5                               // 000000005140: 680A0AFF 00001000
	ds_read_b128 v[90:93], v2 offset:128                       // 000000005148: D9FE0080 5A000002
	ds_read_b128 v[94:97], v2 offset:160                       // 000000005150: D9FE00A0 5E000002
	s_waitcnt lgkmcnt(1)                                       // 000000005158: BF8CC17F
	buffer_store_dwordx4 v[90:93], v5, s[8:11], 0 offen        // 00000000515C: E07C1000 80025A05
	v_add_u32_e32 v5, 0x1000, v5                               // 000000005164: 680A0AFF 00001000
	ds_read_b128 v[98:101], v2 offset:192                      // 00000000516C: D9FE00C0 62000002
	s_waitcnt lgkmcnt(1)                                       // 000000005174: BF8CC17F
	buffer_store_dwordx4 v[94:97], v5, s[8:11], 0 offen        // 000000005178: E07C1000 80025E05
	v_add_u32_e32 v5, 0x1000, v5                               // 000000005180: 680A0AFF 00001000
	ds_read_b128 v[102:105], v2 offset:224                     // 000000005188: D9FE00E0 66000002
	s_waitcnt lgkmcnt(1)                                       // 000000005190: BF8CC17F
	buffer_store_dwordx4 v[98:101], v5, s[8:11], 0 offen       // 000000005194: E07C1000 80026205
	v_add_u32_e32 v5, 0x1000, v5                               // 00000000519C: 680A0AFF 00001000
	s_waitcnt lgkmcnt(0)                                       // 0000000051A4: BF8CC07F
	buffer_store_dwordx4 v[102:105], v5, s[8:11], 0 offen      // 0000000051A8: E07C1000 80026605
	v_add_u32_e32 v5, 0x1000, v5                               // 0000000051B0: 680A0AFF 00001000
	buffer_store_dword v1, v4, s[12:15], 0 offen               // 0000000051B8: E0701000 80030104
	s_branch label_34E0                                        // 0000000051C0: BF820165

00000000000051c4 <label_2F4C>:
	s_mul_i32 s79, 0x400, 16                                   // 0000000051C4: 924F90FF 00000400
	s_mul_i32 s60, s82, s79                                    // 0000000051CC: 923C4F52
	s_add_u32 s92, s60, s92                                    // 0000000051D0: 805C5C3C
	s_addc_u32 s93, 0, s93                                     // 0000000051D4: 825D5D80
	s_mul_i32 s60, s79, s85                                    // 0000000051D8: 923C554F
	s_mov_b32 s94, s60                                         // 0000000051DC: BEDE003C
	s_mul_i32 s60, 0x100, s7                                   // 0000000051E0: 923C07FF 00000100
	v_lshrrev_b32_e32 v28, 5, v0                               // 0000000051E8: 20380085
	s_mov_b32 s61, 0x400                                       // 0000000051EC: BEBD00FF 00000400
	v_mul_i32_i24_e32 v28, s61, v28                            // 0000000051F4: 0C38383D
	v_and_b32_e32 v5, 31, v0                                   // 0000000051F8: 260A009F
	v_lshlrev_b32_e32 v5, 4, v5                                // 0000000051FC: 240A0A84
	v_add_u32_e32 v5, v5, v28                                  // 000000005200: 680A3905
	v_add_u32_e64 v5, v5, s60                                  // 000000005204: D1340005 00007905
	v_lshlrev_b32_e32 v2, 3, v0                                // 00000000520C: 24040083
	s_mov_b32 s60, 0x1400                                      // 000000005210: BEBC00FF 00001400
	s_mul_i32 s60, s7, s60                                     // 000000005218: 923C3C07
	v_add_u32_e32 v2, s60, v2                                  // 00000000521C: 6804043C
	v_cmp_u_f32_e64 s[36:37], v74, v74                         // 000000005220: D0480024 0002954A
	v_add3_u32 v70, v74, v73, 1                                // 000000005228: D1FF0046 0206934A
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005230: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v75, v75                         // 000000005238: D0480024 0002974B
	v_add3_u32 v70, v75, v73, 1                                // 000000005240: D1FF0046 0206934B
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005248: D100001D 00929146
	v_perm_b32 v74, v29, v28, s52                              // 000000005250: D1ED004A 00D2391D
	v_cmp_u_f32_e64 s[36:37], v76, v76                         // 000000005258: D0480024 0002994C
	v_add3_u32 v70, v76, v73, 1                                // 000000005260: D1FF0046 0206934C
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005268: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v77, v77                         // 000000005270: D0480024 00029B4D
	v_add3_u32 v70, v77, v73, 1                                // 000000005278: D1FF0046 0206934D
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005280: D100001D 00929146
	v_perm_b32 v75, v29, v28, s52                              // 000000005288: D1ED004B 00D2391D
	v_cmp_u_f32_e64 s[36:37], v78, v78                         // 000000005290: D0480024 00029D4E
	v_add3_u32 v70, v78, v73, 1                                // 000000005298: D1FF0046 0206934E
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 0000000052A0: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v79, v79                         // 0000000052A8: D0480024 00029F4F
	v_add3_u32 v70, v79, v73, 1                                // 0000000052B0: D1FF0046 0206934F
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 0000000052B8: D100001D 00929146
	v_perm_b32 v76, v29, v28, s52                              // 0000000052C0: D1ED004C 00D2391D
	v_cmp_u_f32_e64 s[36:37], v80, v80                         // 0000000052C8: D0480024 0002A150
	v_add3_u32 v70, v80, v73, 1                                // 0000000052D0: D1FF0046 02069350
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 0000000052D8: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v81, v81                         // 0000000052E0: D0480024 0002A351
	v_add3_u32 v70, v81, v73, 1                                // 0000000052E8: D1FF0046 02069351
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 0000000052F0: D100001D 00929146
	v_perm_b32 v77, v29, v28, s52                              // 0000000052F8: D1ED004D 00D2391D
	v_cmp_u_f32_e64 s[36:37], v82, v82                         // 000000005300: D0480024 0002A552
	v_add3_u32 v70, v82, v73, 1                                // 000000005308: D1FF0046 02069352
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005310: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v83, v83                         // 000000005318: D0480024 0002A753
	v_add3_u32 v70, v83, v73, 1                                // 000000005320: D1FF0046 02069353
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005328: D100001D 00929146
	v_perm_b32 v78, v29, v28, s52                              // 000000005330: D1ED004E 00D2391D
	v_cmp_u_f32_e64 s[36:37], v84, v84                         // 000000005338: D0480024 0002A954
	v_add3_u32 v70, v84, v73, 1                                // 000000005340: D1FF0046 02069354
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005348: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v85, v85                         // 000000005350: D0480024 0002AB55
	v_add3_u32 v70, v85, v73, 1                                // 000000005358: D1FF0046 02069355
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005360: D100001D 00929146
	v_perm_b32 v79, v29, v28, s52                              // 000000005368: D1ED004F 00D2391D
	v_cmp_u_f32_e64 s[36:37], v86, v86                         // 000000005370: D0480024 0002AD56
	v_add3_u32 v70, v86, v73, 1                                // 000000005378: D1FF0046 02069356
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005380: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v87, v87                         // 000000005388: D0480024 0002AF57
	v_add3_u32 v70, v87, v73, 1                                // 000000005390: D1FF0046 02069357
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005398: D100001D 00929146
	v_perm_b32 v80, v29, v28, s52                              // 0000000053A0: D1ED0050 00D2391D
	v_cmp_u_f32_e64 s[36:37], v88, v88                         // 0000000053A8: D0480024 0002B158
	v_add3_u32 v70, v88, v73, 1                                // 0000000053B0: D1FF0046 02069358
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 0000000053B8: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v89, v89                         // 0000000053C0: D0480024 0002B359
	v_add3_u32 v70, v89, v73, 1                                // 0000000053C8: D1FF0046 02069359
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 0000000053D0: D100001D 00929146
	v_perm_b32 v81, v29, v28, s52                              // 0000000053D8: D1ED0051 00D2391D
	v_cmp_u_f32_e64 s[36:37], v90, v90                         // 0000000053E0: D0480024 0002B55A
	v_add3_u32 v70, v90, v73, 1                                // 0000000053E8: D1FF0046 0206935A
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 0000000053F0: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v91, v91                         // 0000000053F8: D0480024 0002B75B
	v_add3_u32 v70, v91, v73, 1                                // 000000005400: D1FF0046 0206935B
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005408: D100001D 00929146
	v_perm_b32 v82, v29, v28, s52                              // 000000005410: D1ED0052 00D2391D
	v_cmp_u_f32_e64 s[36:37], v92, v92                         // 000000005418: D0480024 0002B95C
	v_add3_u32 v70, v92, v73, 1                                // 000000005420: D1FF0046 0206935C
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005428: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v93, v93                         // 000000005430: D0480024 0002BB5D
	v_add3_u32 v70, v93, v73, 1                                // 000000005438: D1FF0046 0206935D
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005440: D100001D 00929146
	v_perm_b32 v83, v29, v28, s52                              // 000000005448: D1ED0053 00D2391D
	v_cmp_u_f32_e64 s[36:37], v94, v94                         // 000000005450: D0480024 0002BD5E
	v_add3_u32 v70, v94, v73, 1                                // 000000005458: D1FF0046 0206935E
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005460: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v95, v95                         // 000000005468: D0480024 0002BF5F
	v_add3_u32 v70, v95, v73, 1                                // 000000005470: D1FF0046 0206935F
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005478: D100001D 00929146
	v_perm_b32 v84, v29, v28, s52                              // 000000005480: D1ED0054 00D2391D
	v_cmp_u_f32_e64 s[36:37], v96, v96                         // 000000005488: D0480024 0002C160
	v_add3_u32 v70, v96, v73, 1                                // 000000005490: D1FF0046 02069360
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005498: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v97, v97                         // 0000000054A0: D0480024 0002C361
	v_add3_u32 v70, v97, v73, 1                                // 0000000054A8: D1FF0046 02069361
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 0000000054B0: D100001D 00929146
	v_perm_b32 v85, v29, v28, s52                              // 0000000054B8: D1ED0055 00D2391D
	v_cmp_u_f32_e64 s[36:37], v98, v98                         // 0000000054C0: D0480024 0002C562
	v_add3_u32 v70, v98, v73, 1                                // 0000000054C8: D1FF0046 02069362
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 0000000054D0: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v99, v99                         // 0000000054D8: D0480024 0002C763
	v_add3_u32 v70, v99, v73, 1                                // 0000000054E0: D1FF0046 02069363
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 0000000054E8: D100001D 00929146
	v_perm_b32 v86, v29, v28, s52                              // 0000000054F0: D1ED0056 00D2391D
	v_cmp_u_f32_e64 s[36:37], v100, v100                       // 0000000054F8: D0480024 0002C964
	v_add3_u32 v70, v100, v73, 1                               // 000000005500: D1FF0046 02069364
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005508: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v101, v101                       // 000000005510: D0480024 0002CB65
	v_add3_u32 v70, v101, v73, 1                               // 000000005518: D1FF0046 02069365
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005520: D100001D 00929146
	v_perm_b32 v87, v29, v28, s52                              // 000000005528: D1ED0057 00D2391D
	v_cmp_u_f32_e64 s[36:37], v102, v102                       // 000000005530: D0480024 0002CD66
	v_add3_u32 v70, v102, v73, 1                               // 000000005538: D1FF0046 02069366
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005540: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v103, v103                       // 000000005548: D0480024 0002CF67
	v_add3_u32 v70, v103, v73, 1                               // 000000005550: D1FF0046 02069367
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005558: D100001D 00929146
	v_perm_b32 v88, v29, v28, s52                              // 000000005560: D1ED0058 00D2391D
	v_cmp_u_f32_e64 s[36:37], v104, v104                       // 000000005568: D0480024 0002D168
	v_add3_u32 v70, v104, v73, 1                               // 000000005570: D1FF0046 02069368
	v_cndmask_b32_e64 v28, v70, v72, s[36:37]                  // 000000005578: D100001C 00929146
	v_cmp_u_f32_e64 s[36:37], v105, v105                       // 000000005580: D0480024 0002D369
	v_add3_u32 v70, v105, v73, 1                               // 000000005588: D1FF0046 02069369
	v_cndmask_b32_e64 v29, v70, v72, s[36:37]                  // 000000005590: D100001D 00929146
	v_perm_b32 v89, v29, v28, s52                              // 000000005598: D1ED0059 00D2391D
	ds_write_b64 v2, v[74:75]                                  // 0000000055A0: D89A0000 00004A02
	s_mov_b32 s60, 0x208                                       // 0000000055A8: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 0000000055B0: 6804043C
	ds_write_b64 v2, v[76:77]                                  // 0000000055B4: D89A0000 00004C02
	s_mov_b32 s60, 0x208                                       // 0000000055BC: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 0000000055C4: 6804043C
	ds_write_b64 v2, v[78:79]                                  // 0000000055C8: D89A0000 00004E02
	s_mov_b32 s60, 0x208                                       // 0000000055D0: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 0000000055D8: 6804043C
	ds_write_b64 v2, v[80:81]                                  // 0000000055DC: D89A0000 00005002
	s_mov_b32 s60, 0x208                                       // 0000000055E4: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 0000000055EC: 6804043C
	ds_write_b64 v2, v[82:83]                                  // 0000000055F0: D89A0000 00005202
	s_mov_b32 s60, 0x208                                       // 0000000055F8: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 000000005600: 6804043C
	ds_write_b64 v2, v[84:85]                                  // 000000005604: D89A0000 00005402
	s_mov_b32 s60, 0x208                                       // 00000000560C: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 000000005614: 6804043C
	ds_write_b64 v2, v[86:87]                                  // 000000005618: D89A0000 00005602
	s_mov_b32 s60, 0x208                                       // 000000005620: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 000000005628: 6804043C
	ds_write_b64 v2, v[88:89]                                  // 00000000562C: D89A0000 00005802
	s_mov_b32 s60, 0x208                                       // 000000005634: BEBC00FF 00000208
	v_add_u32_e32 v2, s60, v2                                  // 00000000563C: 6804043C
	v_and_b32_e32 v28, 1, v0                                   // 000000005640: 26380081
	s_mov_b32 s60, 0x100                                       // 000000005644: BEBC00FF 00000100
	v_mul_i32_i24_e32 v28, s60, v28                            // 00000000564C: 0C38383C
	v_and_b32_e32 v29, 15, v0                                  // 000000005650: 263A008F
	v_lshrrev_b32_e32 v29, 1, v29                              // 000000005654: 203A3A81
	s_mov_b32 s60, 0x208                                       // 000000005658: BEBC00FF 00000208
	v_mul_i32_i24_e32 v29, s60, v29                            // 000000005660: 0C3A3A3C
	v_lshrrev_b32_e32 v30, 4, v0                               // 000000005664: 203C0084
	s_mov_b32 s60, 8                                           // 000000005668: BEBC0088
	v_mul_i32_i24_e32 v30, s60, v30                            // 00000000566C: 0C3C3C3C
	v_add_u32_e32 v2, v28, v29                                 // 000000005670: 68043B1C
	v_add_u32_e32 v2, v30, v2                                  // 000000005674: 6804051E
	s_mov_b32 s60, 0x80                                        // 000000005678: BEBC00FF 00000080
	v_add_u32_e32 v3, s60, v2                                  // 000000005680: 6806043C
	s_mov_b32 s61, 0x1400                                      // 000000005684: BEBD00FF 00001400
	s_mul_i32 s60, s61, s7                                     // 00000000568C: 923C073D
	v_add_u32_e32 v2, s60, v2                                  // 000000005690: 6804043C
	v_add_u32_e32 v3, s60, v3                                  // 000000005694: 6806063C
	s_mov_b32 s60, 0x100                                       // 000000005698: BEBC00FF 00000100
	s_mul_i32 s60, s7, s60                                     // 0000000056A0: 923C3C07
	v_lshrrev_b32_e32 v28, 4, v0                               // 0000000056A4: 20380084
	s_mov_b32 s61, 0x400                                       // 0000000056A8: BEBD00FF 00000400
	v_mul_i32_i24_e32 v28, s61, v28                            // 0000000056B0: 0C38383D
	v_and_b32_e32 v5, 15, v0                                   // 0000000056B4: 260A008F
	v_lshlrev_b32_e32 v5, 4, v5                                // 0000000056B8: 240A0A84
	v_add_u32_e32 v5, s60, v5                                  // 0000000056BC: 680A0A3C
	v_add_u32_e32 v5, v28, v5                                  // 0000000056C0: 680A0B1C
	s_waitcnt lgkmcnt(0)                                       // 0000000056C4: BF8CC07F
	ds_read_b64 v[74:75], v2                                   // 0000000056C8: D8EC0000 4A000002
	ds_read_b64 v[76:77], v3                                   // 0000000056D0: D8EC0000 4C000003
	ds_read_b64 v[78:79], v2 offset:32                         // 0000000056D8: D8EC0020 4E000002
	ds_read_b64 v[80:81], v3 offset:32                         // 0000000056E0: D8EC0020 50000003
	s_waitcnt lgkmcnt(2)                                       // 0000000056E8: BF8CC27F
	buffer_store_dwordx4 v[74:77], v5, s[92:95], 0 offen       // 0000000056EC: E07C1000 80174A05
	v_add_u32_e32 v5, 0x1000, v5                               // 0000000056F4: 680A0AFF 00001000
	ds_read_b64 v[82:83], v2 offset:64                         // 0000000056FC: D8EC0040 52000002
	ds_read_b64 v[84:85], v3 offset:64                         // 000000005704: D8EC0040 54000003
	s_waitcnt lgkmcnt(2)                                       // 00000000570C: BF8CC27F
	buffer_store_dwordx4 v[78:81], v5, s[92:95], 0 offen       // 000000005710: E07C1000 80174E05
	v_add_u32_e32 v5, 0x1000, v5                               // 000000005718: 680A0AFF 00001000
	ds_read_b64 v[86:87], v2 offset:96                         // 000000005720: D8EC0060 56000002
	ds_read_b64 v[88:89], v3 offset:96                         // 000000005728: D8EC0060 58000003
	s_waitcnt lgkmcnt(2)                                       // 000000005730: BF8CC27F
	buffer_store_dwordx4 v[82:85], v5, s[92:95], 0 offen       // 000000005734: E07C1000 80175205
	v_add_u32_e32 v5, 0x1000, v5                               // 00000000573C: 680A0AFF 00001000
	s_waitcnt lgkmcnt(0)                                       // 000000005744: BF8CC07F
	buffer_store_dwordx4 v[86:89], v5, s[92:95], 0 offen       // 000000005748: E07C1000 80175605
	v_add_u32_e32 v5, 0x1000, v5                               // 000000005750: 680A0AFF 00001000

0000000000005758 <label_34E0>:
	s_mov_b32 s60, 32                                          // 000000005758: BEBC00A0
	s_addk_i32 s89, 0x1                                        // 00000000575C: B7590001
	s_cmp_lt_i32 s89, s90                                      // 000000005760: BF045A59
	s_cbranch_scc1 label_00A4                                  // 000000005764: BF85F30F

0000000000005768 <label_34F0>:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000005768: BF8C0000
	s_endpgm                                                   // 00000000576C: BF810000
