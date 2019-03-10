#!/bin/sh

USE_GPUS="$1"
echo "USE_GPUS=$USE_GPUS"
shift

ENV_GENERAL="train-settings/general/gpu-train-dev.env"
## use next argument as gpu nbr, if available
if [ -n "$1" ]; then
    ENV_GENERAL="$1"
    shift
fi
echo "ENV_GENERAL=$ENV_GENERAL"

## SICK relatedness prediction

LOGDIR_CONTINUE=RELATEDNESS/SICK/DIRECT/RECNN2/aEDG_avfF_bs100_dF_bPOS-NIFW-PAR_clp5.0_cmTREE_cntxt0_dfidx4-5_dtF_ec_fc_kp0.9_kpb1.0_kpn1.0_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_n10000_ns20_nfvF_rootfc300_sm_sl50_st150_tk_dataCORENLPNONERRECEMBMC2_teHTUREDUCESUMMAPGRU_ccF_tfidfF_vvrF_vvzF/0,RELATEDNESS/SICK/DIRECT/RECNN/aEDG_avfF_bs100_dF_bPOS-NIFW-PAR_clp5.0_cmTREE_cntxt0_dfidx0-1_dtF_ec_fc_kp0.9_kpb1.0_kpn1.0_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_n10000_ns20_nfvF_rootfc300_sm_sl50_st150_tk_dataCORENLPNONERRECEMBMC2_teHTUREDUCEMAXMAPGRU_ccF_tfidfF_vvrF_vvzF/1,RELATEDNESS/SICK/DIRECT/RNN/aEDG_avfF_bs100_dF_b_clp5.0_cmAGGREGATE_cntxt0_dfidx4-5_dtF_fc_kp0.9_kpb1.0_kpn1.0_leaffc150_lr0.001_lc-1_dpth20_mtSIMTUPLE_n10000_ns20_nfvF_rootfc300_sm_sl50_st0_tk_dataCORENLPNONERRECEMBMC2_teFLATSUM_ccF_tfidfF_vvrF_vvzF/0,RELATEDNESS/SICK/DIRECT/BOW/aEDG_avfF_bs100_dF_b_clp5.0_cmAGGREGATE_cntxt0_dfidx0-1_dtF_ec_fc_kp0.9_kpb1.0_kpn1.0_leaffc150_lr0.001_lc-1_dpth20_mtSIMTUPLE_n10000_ns20_nfvF_rootfc300_sm_sl50_st0_tk_dataCORENLPNONERRECEMBMC2_teFLATMAX_ccF_tfidfF_vvrF_vvzF/2,RELATEDNESS/SICK/DIRECT/BOW/aEDG_avfF_bs100_dF_b_clp5.0_cmAGGREGATE_cntxt0_dfidx4-5_dtF_fc_kp0.9_kpb1.0_kpn1.0_leaffc0_lr0.001_lc-1_dpth20_mtSIMTUPLE_n10000_ns20_nfvF_rootfc300_sm_sl50_st150_tk_dataCORENLPNONERRECEMBMC2_teFLATGRU_ccF_tfidfF_vvrF_vvzF/1

export LOGDIR_CONTINUE

## CONTINUE

# RECNN
./train.sh "$USE_GPUS" RELATEDNESS/SICK/CONTINUE "$ENV_GENERAL"