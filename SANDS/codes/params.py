model_params = {'pretrain_batch': 128,
                'test_batch': 512,
                'semisup_batch': 512,
                'pretrain_epochs': 1,
                'semisup_epochs': 15,
                'pretrain_dropout': 0.1,
                'semisup_dropout': 0.3,
                'lr': 0.0001,
                'min_degree': 20}
data_params =          {'vocab': '../Data/Data/VOCAB',
                        'hashtag': '../Data/Data/HASHTAG',
                        'batch': '../Data/Data/BATCHES',
                        'EW': '../Data/Data/EW',
                        'userlist': '../Data/Data/USERLIST',
                        'first_tweet_text': '../Data/Data/first_tweet_array_text.npy',
                        'first_tweet_hashtag': '../Data/Data/first_tweet_array_hashtag.npy',
                        'word_embedding': '../Data/Data/glove_matrix.npy',
                        'train_text': '../Data/TrainData/train_text.npy',
                        'train_hashtag': '../Data/TrainData/train_hashtag.npy',
                        'train_labels': '../Data/TrainData/train_labels.npy',
                        'test_text': '../Data/TestData/test_text.npy',
                        'test_hashtag': '../Data/TestData/test_hashtag.npy',
                        'test_labels': '../Data/TestData/test_labels.npy',
                        'checkpoint_path': 'Model/'}

def _split_location(data_name, split_size, file_param):
    return '/'.join(file_param.split('/')[:-1])+'_'+data_name+'_'+split_size+'/'+file_param.split('/')[-1]
def _common_location(data_name, file_param):
    return '/'.join(file_param.split('/')[:-1])+'_'+data_name+'/'+file_param.split('/')[-1]
