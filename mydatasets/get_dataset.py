from .sleep_dataset import EdfDataset, EdfFreqTimeDataset, EdfCoTrainDataset, EdfAugFreqTimeDataset, EdfSTFTDataset, EdfCoTrainFreqDataset
from .ISRUC_dataset import ISRUCDataset, ISRUCSTFTDataset, ISRUCCoTrainFreqDataset

def get_dataset(data_name, data_path, seq_len, is_train, is_contrastive=False, random_shuffle=42, choice_idx=[]):
    """
    data_name: ['edf39_signal', 'edf153_signal', 'edf39_timefreq', 'edf153_timefreq']
    """
    dataset = None
    if data_name == 'edf39_signal':
        dataset = EdfDataset(data_path, seq_len=seq_len, is_train=is_train, is_EDF39=True, shuffle_seed=random_shuffle)
    elif data_name == 'edf153_signal':
        dataset = EdfDataset(data_path, seq_len=seq_len, is_train=is_train, is_EDF39=False, shuffle_seed=random_shuffle)
    elif data_name == 'edf39_timefreq':
        dataset = EdfFreqTimeDataset(data_path, seq_len=seq_len, is_train=is_train, is_EDF39=True, shuffle_seed=random_shuffle, is_contrastive=is_contrastive)
    elif data_name == 'edf153_timefreq':
        dataset = EdfFreqTimeDataset(data_path, seq_len=seq_len, is_train=is_train, is_EDF39=False, shuffle_seed=random_shuffle)
    elif data_name == 'aug_edf39_timefreq':
        dataset = EdfAugFreqTimeDataset(data_path, seq_len=seq_len, is_train=is_train, shuffle_seed=random_shuffle, is_EDF39=True)
    elif data_name == 'isruc':
        dataset = ISRUCDataset(data_path, signal="F3_A2", is_train=is_train, shuffle_seed=random_shuffle, choice_idx=[])
    elif data_name == 'isruc_C3_A2':
        dataset = ISRUCDataset(data_path, signal="C3_A2", is_train=is_train, shuffle_seed=random_shuffle, choice_idx=[])
    elif data_name == 'isruc_stft':
        dataset = ISRUCSTFTDataset(data_path, seq_len, is_train=is_train, shuffle_seed=random_shuffle, choice_idx=choice_idx)
    elif data_name == 'edf39_stft':
        dataset = EdfSTFTDataset(data_path, seq_len, is_train=is_train, shuffle_seed=random_shuffle, is_EDF39=True, choice_idx=choice_idx)
    elif data_name == 'edf153_stft':
        dataset = EdfSTFTDataset(data_path, seq_len, is_train=is_train, shuffle_seed=random_shuffle, is_EDF39=False, choice_idx=choice_idx)
    else:
        raise NotImplementedError
    
    return dataset


def get_co_dataset(data_name, data_path1, data_path2, seq_len=16, random_shuffle=42):
    dataset = None
    if data_name == 'edf39_cotrain':
        dataset = EdfCoTrainDataset(data_path1, data_path2, seq_len, is_train=True, is_EDF39=True, shuffle_seed=random_shuffle)
    elif data_name == 'edf153_cotrain':
        dataset = EdfCoTrainDataset(data_path1, data_path2, seq_len, is_train=True, is_EDF39=True, shuffle_seed=random_shuffle)
    elif data_name == 'edf39_cotrain_freq':
        dataset = EdfCoTrainFreqDataset(data_path1=data_path1, data_path2=data_path2, seq_len=seq_len, is_train=True, shuffle_seed=random_shuffle, is_EDF39=True)
    elif data_name == 'edf153_cotrain_freq':
        dataset = EdfCoTrainFreqDataset(data_path1=data_path1, data_path2=data_path2, seq_len=seq_len, is_train=True, shuffle_seed=random_shuffle, is_EDF39=False)
    elif data_name == 'isruc_cotrain':
        dataset = ISRUCCoTrainFreqDataset(data_path1, data_path2, seq_len, is_train=True, shuffle_seed=random_shuffle)
    else:
        raise NotImplementedError
    
    return dataset