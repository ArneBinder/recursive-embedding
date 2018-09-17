=== SICK RELATEDNESS ===================================================================================================
## similarity (tuple, continuous)
# TREE: pearsons_r: 0.8532 (*one* test file)
SICK/multi/TREE_ps10/avfFALSE_avzFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0_ddtTRUE_fc_kp0.9_leaffc0_lr0.003_lc-1_dpth20_mtSIMTUPLE_nbrt10000_nbrtt1000_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl-1_st150_dataMERGED_teHTUREDUCESUMMAPGRU_tfidfFALSE/0/test
# TREE: pearsons_r: ~0.8472 (3 runs), 0.8497
avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
# GRU: pearsons_r: ~0.85233 (3 runs), 0.8547
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teFLATCONCATGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
# SUM: pearsons_r: ~0.84987 (3 runs), 0.8508
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teFLATSUM_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/1
# TFIDF: pearsons_r: ~0.80066 (3 runs), 0.8018
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teTFIDF_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
# TREE + TFIDF: pearsons_r: ~0.84740 (3 runs), 0.8482
avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc_kp0.9_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc300_sl50_st150_tk_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfTRUE_vvrFALSE_vvzFALSE/0


=== SICK ENTAILMENT ====================================================================================================
## classification (tuple, discrete, exclusive)
# TREE: f1_t33: ~0.7713 (3 runs), 0.7744
avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/1
# TFIDF: f1_t33: ~0.6235 (3 runs), 0.6266
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teTFIDF_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
# GRU: f1_t33: ~0.7764 (3 runs), 0.7779
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teFLATCONCATGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
# SUM: f1_t33: ~0.7440 (3 runs), 0.7512
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teFLATSUM_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2
# TREE+ TFIDF: f1_t33: ~0.6988 (3 runs), 0.7010
avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc1000_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtTUPLECLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl50_st150_tkENTAILMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfTRUE_vvrFALSE_vvzFALSE/1


=== IMDB SENTIMENT =====================================================================================================
## classification (single, discrete, binary)
# TREE: f1_t33: 0.8947 (*one* test file)
IMDB/multi/TREE_ps1/avfFALSE_avzFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0_dtFALSE_fc400_kp0.9_leaffc300_lr0.0003_lc-1_dpth17_mtMULTICLASS_nbrt10000_nbrtt1000_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl-1_st350_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE/8/test
# TREE: 0:50h; accuracy_50: ~0.891 (3 runs), 0.8923
avfFALSE_bs100_clp5.0_cmTREE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teHTUREDUCESUMMAPGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/1
# TFIDF: 0:03h; accuracy_50: ~0.884 (3 runs), 0.8850
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc300_lr0.0001_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teTFIDF_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/0
# GRU: 3:30h; accuracy_50: ~0.8977 (3 runs), 0.8981
avfFALSE_bs100_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtFALSE_fc400_kp0.9_leaffc0_lr0.0003_lc-1_dpth20_mtMULTICLASS_ns20_nfvFALSE_optADAMOPTIMIZER_rootfc0_sl500_st350_tkSENTIMENT_dataMERGED_teFLATCONCATGRU_ccFALSE_tfidfFALSE_vvrFALSE_vvzFALSE/2



=== BIOASQ MULTICLASS ==================================================================================================
## classification (single, discrete, independent)



=== DBPEDIANIF RELATEDNESS =============================================================================================
## similarity (multiple tuples, continuous, exclusive) -> trained with negative samples



=== LANGUAGE MODEL / EMBEDDINGS ========================================================================================
## consistency (multiple singles, continuous, exclusive) -> trained with negative samples
