# -*- coding: utf-8 -*-


configs_2Speakers_ivectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/cluster_center_128.npy",
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs_2Speakers_ivector_ivector128_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 896*2,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs2_2Speakers_ivector_ivector128_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 512*2,
"Shared_BLSTM_dim": 512,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 512,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs_SC_Multiple_4Speakers_xvectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/cluster_center_128.npy",
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}


configs_4Speakers_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivector_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivector_xvectors64_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/xvector_cluster_center_64.npy",
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivector_xvectors256_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/xvector_cluster_center_256.npy",
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivector_ivector128_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs2_4Speakers_ivector_ivector128_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 896,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}


configs3_4Speakers_ivector_ivector128_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 896*2,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs3_4Speakers_ivector_ivector64_xvectors64_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_64.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_64.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 896*2,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs3_4Speakers_ivector_ivector256_xvectors256_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_256.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_256.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 896*2,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_xvectors128_2Classes2 = {"input_dim": 64,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"fea_dim": 32*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 32*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}


configs_4Speakers_ivectors128_heads16_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 16,
"embedding_path": "embedding_raw/voxceleb/cluster_center_128.npy",
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivectors128_heads4_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 4,
"embedding_path": "embedding_raw/voxceleb/cluster_center_128.npy",
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivectors128_heads32_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 32,
"embedding_path": "embedding_raw/voxceleb/cluster_center_128.npy",
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivectors64_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/cluster_center_64.npy",
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_ivectors256_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"fea_dim": 20*128,
"n_heads": 8,
"embedding_path": "embedding_raw/voxceleb/cluster_center_256.npy",
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_SC_Multiple_8Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 8
}

configs_8Speakers_ivector_ivector128_xvectors128_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 20*128,
"n_heads1": 8,
"embedding_path1": "embedding_raw/voxceleb/cluster_center_128.npy",
"n_heads2": 8,
"embedding_path2": "embedding_raw/voxceleb/xvector_cluster_center_128.npy",
"splice_size": 20*128+100+100+256,
"Linear_dim": 896*2,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 8
}


configs = {
    "configs_SC_Multiple_8Speakers_ivectors_2Classes": configs_SC_Multiple_8Speakers_ivectors_2Classes,
    "configs_2Speakers_ivectors128_2Classes": configs_2Speakers_ivectors128_2Classes,
    "configs_4Speakers_ivectors128_2Classes": configs_4Speakers_ivectors128_2Classes,
    "configs_4Speakers_ivectors64_2Classes": configs_4Speakers_ivectors64_2Classes,
    "configs_4Speakers_ivectors256_2Classes": configs_4Speakers_ivectors256_2Classes,
    "configs_SC_Multiple_4Speakers_xvectors_2Classes": configs_SC_Multiple_4Speakers_xvectors_2Classes,
    "configs_4Speakers_ivectors128_heads4_2Classes": configs_4Speakers_ivectors128_heads4_2Classes,
    "configs_4Speakers_ivectors128_heads16_2Classes": configs_4Speakers_ivectors128_heads16_2Classes,
    "configs_4Speakers_ivectors128_heads32_2Classes": configs_4Speakers_ivectors128_heads32_2Classes,
    "configs_4Speakers_xvectors128_2Classes": configs_4Speakers_xvectors128_2Classes,
    "configs_4Speakers_ivector_xvectors128_2Classes": configs_4Speakers_ivector_xvectors128_2Classes,
    "configs_4Speakers_ivector_xvectors64_2Classes": configs_4Speakers_ivector_xvectors64_2Classes,
    "configs_4Speakers_ivector_xvectors256_2Classes": configs_4Speakers_ivector_xvectors256_2Classes,
    "configs_4Speakers_ivector_ivector128_xvectors128_2Classes": configs_4Speakers_ivector_ivector128_xvectors128_2Classes,
    "configs2_4Speakers_ivector_ivector128_xvectors128_2Classes": configs2_4Speakers_ivector_ivector128_xvectors128_2Classes,
    "configs3_4Speakers_ivector_ivector128_xvectors128_2Classes": configs3_4Speakers_ivector_ivector128_xvectors128_2Classes,
    "configs_8Speakers_ivector_ivector128_xvectors128_2Classes": configs_8Speakers_ivector_ivector128_xvectors128_2Classes,
    "configs_2Speakers_ivector_ivector128_xvectors128_2Classes": configs_2Speakers_ivector_ivector128_xvectors128_2Classes,
    "configs2_2Speakers_ivector_ivector128_xvectors128_2Classes": configs2_2Speakers_ivector_ivector128_xvectors128_2Classes,
    "configs3_4Speakers_ivector_ivector64_xvectors64_2Classes": configs3_4Speakers_ivector_ivector64_xvectors64_2Classes,
    "configs3_4Speakers_ivector_ivector256_xvectors256_2Classes": configs3_4Speakers_ivector_ivector256_xvectors256_2Classes
}