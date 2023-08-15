#!/usr/bin/env bash


stage=3
nj=8

# data=AMI_Headset_dev
# sets=dev
# affix=VBx

data=AMI_Headset_test
sets=test
affix=VBx

#data=dipco_U02_CH1
#data=chime7_dev_U06_CH1
#data=mixer6_dev_CH4
#affix=VBx_OracleVAD
#affix=Oracle_OracleVAD
#affix=MAMSE_it1_OracleVAD
train_set_speaker_embedding_list=exp/nnet3_recipe_ivector/ivectors_AMI_Headset_train/ivectors_spk.txt
#train_set_speaker_embedding_list=/export/corpus/exp/kaldi/egs/tsvad/s5/exp/nnet3_recipe_ivector/ivectors_AMI_VoxConverse_LibriSim/ivectors_spk.txt

model_path=model/MULTI_SE_MA_MSE_NSD/Batchsize48_4speakers_Segment800s_configs3_4Speakers_ivector128_xvectors128_2Classes_Mixup0.5_AMI_Headset/MULTI_SE_MA_MSE_NSD.model50
#model_path=model/MULTI_SE_MA_MSE_NSD/Batchsize96_4speakers_Segment800s_configs3_4Speakers_ivector128_xvectors128_2Classes_AMI_RAW+WPE_Array1_train/MULTI_SE_MA_MSE_NSD.model11

model_config=configs3_4Speakers_ivector_ivector128_xvectors128_2Classes
#model_config=configs_8Speakers_ivector_ivector128_xvectors128_2Classes


#diarized_rttm=data/AMI_Headset_dev/VBx_OracleVAD/sys.rttm
diarized_rttm=data/AMI_Headset_test/VBx_OracleVAD/sys.rttm



max_utt_durance=800
max_speaker=8
batch_size=32
ivector_dir=/export/corpus/exp/kaldi/egs/tsvad/s5/exp/nnet3_recipe_ivector/
do_vad=true
gpu=2

th_s=40
th_e=50

export PATH=/usr/bin/:$PATH

. path.sh
. utils/parse_options.sh

oracle_vad=data/AMI_Headset_${sets}/${sets}.lab
echo $set $stage $model_path $train_set_speaker_embedding_list  $model_config $data $th_s $th_e $oracle_vad


if [ $stage -le 1 ]; then
  local/extract_feature.sh --stage 0 --nj $nj \
      --sample_rate _16k --ivector_dir $ivector_dir \
      --max_speaker 4 --affix _$affix \
      --rttm $diarized_rttm --data $data
fi

if [ $stage -le 2 ]; then
    CUDA_VISIBLE_DEVICES=$gpu python decode_MULTI_SE_MA_MSE_NSD.py \
        --feature_list data/${data}/cmn_slide_fbank_htk.list \
        --embedding_list ${ivector_dir}/ivectors_${data}_${affix}/ivectors_spk.txt  \
        --train_set_speaker_embedding_list ${train_set_speaker_embedding_list} \
        --model_path ${model_path} \
        --output_dir ${model_path}_${data}_${affix} \
        --max_speaker $max_speaker \
        --init_rttm ${diarized_rttm} \
        --model_config ${model_config} \
        --max_utt_durance ${max_utt_durance} \
        --batch_size $batch_size \
        --remove_overlap
fi

if  [ $stage -le 3 ]; then
  for th in `seq $th_s 5 $th_e`; do
    echo "$th "
    #if false;then
    python postprocessing.py --threshold $th --medianfilter -1 \
        --prob_array_dir ${model_path}_${data}_$affix  --min_segments 0 \
        --min_dur 0.00 --segment_padding 0.00 --max_dur 0.20
    ./analysis_diarization.sh data/$data/oracle.rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp 2>/dev/null | grep ALL
    
    if $do_vad;then
      python rttm_filter_with_vad.py \
          --input_rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp \
          --output_rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp_oraclevad \
          --oracle_vad ${oracle_vad}
      ./analysis_diarization.sh data/$data/oracle.rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp_oraclevad 2>/dev/null | grep ALL
    fi
  done
fi
