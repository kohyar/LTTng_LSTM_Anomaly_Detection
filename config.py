traces_folder_path = "" # To be set: This folder contains csv files

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
num_epochs = 600 # To be set
batch_size = 2048 # To be set

window_size = 17 # To be set
test_percentage = 0.1 # To be set
input_size = 1
num_candidates = 10
num_layers = 2
hidden_size = 64
