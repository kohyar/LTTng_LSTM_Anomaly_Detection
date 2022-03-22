traces_folder_path = "" # To beset: This folder contains csv files

fields = {
    'msgTrace:reqrep': {
        'alias': True,
        'procname': True
    },
    'ddmZmq:OAL_LOG_DEBUG':{
        'msg': True,
        'procname': True
    },
    'oal:OAL_LOG_DEBUG': {
        'func': True,
        'procname': True
    },
    'all': {
        'procname': True
    }
}

# Hyperparameters
num_epochs = 300
batch_size = 2048

# num_classes = 2327  # 2327 or 189
input_size = 1
num_candidates = 10
num_layers = 2
hidden_size = 64
window_size = 17
test_percentage = 0.1

# model_path = '/home/imkoh/source/ciena_span_sequence_project/nlp_projrct/model/Model_windowsize=17_prob.pt'
model_path = '/home/imkoh/source/ciena_span_sequence_project/nlp_projrct/model/Model_windowsize=17_prob_noise.pt'
# model_path = '/home/imkoh/source/ciena_span_sequence_project/nlp_projrct/model/batch_size=2048_epoch=200_releases6_withunknown_in_train_len_15.pt'