=== SICK RELATEDNESS ===================================================================================================
## similarity (tuple, continuous)
model   pearsons_r_avg(3runs)   pearsons_r_best time    desc
TREE(*one* test file)       0.8532  0:16h   SICK/multi/TREE_ps10/avfFALSE_avzFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0_ddtTRUE_fc_kp0.9_leaffc0_lr0.003_lc-1_dpth20_mtSIMTUPLE_nbrt10000_nbrtt1000_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl-1_st150_dataMERGED_teHTUREDUCESUMMAPGRU_tfidfFALSE/0/test
TREE    0.8472  0.8497      avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
GRU	0.85233	0.8547      avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teFLATCONCATGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
SUM	0.84987	0.8508      avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teFLATSUM_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/1
TFIDF	0.80066	0.8018      avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teTFIDF_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
TREE+TFIDF	0.84740	0.8482  avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfTRUE_vvrFALSE_vvzFALSE/0
TREE2	0.8515	0.8521  avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teHTUREDUCESUMMAPGRU2_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
TREE@ps3	0.8470	0.8487  avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzTRUE/0
TREE2@ps3	0.8523	0.8551  avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teHTUREDUCESUMMAPGRU2_ccFALSE_tfidfFALSE_vvrFALSE_vvzTRUE/1


=== SICK ENTAILMENT ====================================================================================================
## classification (tuple, discrete, exclusive)
model   accuracy_t66_avg(3runs)  accuracy_t66_best  time    desc
TREE	0.8444	0.8466      avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/1
TFIDF	0.7016	0.7031      avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teTFIDF_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
GRU	0.8481	0.8488      avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teFLATCONCATGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
SUM	0.8271	0.8318      avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teFLATSUM_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
TREE+TFIDF	0.7962	0.7979      avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfTRUE_vvrFALSE_vvzFALSE/1
TREE2	0.8563	0.8626      avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teHTUREDUCESUMMAPGRU2_ccFALSE_tfidfFALSE_vvrTRUE_vvzFALSE/2
TREE@ps3    0.8548  0.8610      avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzTRUE/1
TREE2@ps3   0.8579  0.8596      avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teHTUREDUCESUMMAPGRU2_ccFALSE_tfidfFALSE_vvrFALSE_vvzTRUE/0


=== IMDB SENTIMENT =====================================================================================================
## classification (single, discrete, binary)
model   f1_t33_avg(3runs)  f1_t33_best  time    desc
TREE(*one* test file)	0.8947  ... ... IMDB/multi/TREE_ps1/avfFALSE_avzFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0_dtFALSE_fc400_kp0.9_leaffc300_lr0.0003_lc-1_dpth17_mtMULTICLASS_nbrt10000_nbrtt1000_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl-1_st350_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE/8/test
TREE	0.8913	0.8923	0:50h   avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/1
TFIDF	0.884	0.8850	0:03h   avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teTFIDF_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
GRU	0.8977	0.8981	3:30h   avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teFLATCONCATGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
SUM	0.8559	0.8563	... avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teFLATSUM_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
TREE+TFIDF    0.8951	0.8959  ... avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfTRUE_vvrFALSE_vvzFALSE/1


=== BIOASQ MULTICLASS ==================================================================================================
## classification (single, discrete, independent)



=== DBPEDIANIF RELATEDNESS =============================================================================================
## similarity (multiple tuples, continuous, exclusive) -> trained with negative samples



=== LANGUAGE MODEL / EMBEDDINGS ========================================================================================
## consistency (multiple singles, continuous, exclusive) -> trained with negative samples
