2018-08-13
==========
 * unsupervised reroot models on BIOASQ and DBPEDIANIF:
    * BIOASQ/split10000_xa_to_xy/reroot_paragraphs/biter_bs100_clp5.0_cmTREE_cntxt0_devfidx0_fc0_kp1.0_leaffc600_learnr0.0003_lc-1_dpth8_mtREROOT_nbrt100000_nbrtt10000_ns20_nfxTRUE_optADAMOPTIMIZER_rootfc0_state900_dataMERGEDMIN100PARAGRAPHSFOREST_teTREEEMBEDDINGHTUREDUCESUMMAPGRU
    * DBPEDIANIF_MIN100/reroot/biter_bs100_clp5.0_cmTREE_cntxt0_devfidx0_fc0_kp1.0_leaffc600_learnr0.0003_lc-1_dpth8_mtREROOT_nbrt100000_nbrtt10000_ns20_nfxTRUE_optADAMOPTIMIZER_rootfc0_state900_dataMERGEDMIN100FOREST_teTREEEMBEDDINGHTUREDUCESUMMAPGRU

2018-08-14
==========
 * use unsupervised models from 2018-08-13 to initialize supervised BIOASQ training (supervised/log/DEBUG/BIOASQ/split10000_xa_to_xy/TF1/)
    * supervised/log/DEBUG/BIOASQ/split10000_xa_to_xy/TF1/PRETRAINED_BIOASQ/biter_bs100_clp5.0_cmTREE_cntxt0_devfidx0_fc0_kp1.0_leaffc600_learnr0.0003_lc-1_dpth8_mtREROOT_nbrt100000_nbrtt10000_ns20_nfxTRUE_optADAMOPTIMIZER_rootfc0_state900_dataMERGEDMIN100PARAGRAPHSFOREST_teTREEEMBEDDINGHTUREDUCESUMMAPGRU

