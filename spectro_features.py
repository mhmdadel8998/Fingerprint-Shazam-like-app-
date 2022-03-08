import librosa
from PIL import Image
import imagehash
from imagehash import hex_to_hash
from scipy import signal

def creat_dic(file_name):
    song_dict = {
        file_name: {"spectrogram_Hash": None,
        "melspectrogram_Hash": None,
        "mfcc_Hash": None,
        "chroma_stft_Hash": None}
    }
    return song_dict

def Features(file_data, sr, spectro, file_name):
    filename = file_name.split(".")[-1]
    #spectro
    melspectro = librosa.feature.melspectrogram(file_data, sr=sr, S=spectro)
    melspectro_image = Image.fromarray(melspectro, mode='RGB')
    melspectro_image.save("songs/spectro_features/"+filename+"_melspectro.png")
    #chroma_stft
    chroma_stft = librosa.feature.chroma_stft(file_data, sr=sr, S=spectro)
    chroma_stft_image = Image.fromarray(chroma_stft, mode='RGB')
    chroma_stft_image.save("songs/spectro_features/"+filename+"_chroma_stft.png")
    #mfccs
    mfccs = librosa.feature.mfcc(file_data.astype('float64'), sr=sr)
    mfccs_image = Image.fromarray(mfccs, mode='RGB')
    mfccs_image.save("songs/spectro_features/"+filename+"_mfccs.png")
    
    return [melspectro, mfccs, chroma_stft]

def Hash(feature):
    data = Image.fromarray(feature)
    return imagehash.phash(data, hash_size = 16).__str__()

def Load_Song(file_name,file_data,sr):
    filename = file_name.split(".")[-1]
    #Loads audio file, create a spectrogram, extract some features and hash them.
    song = creat_dic(file_name)
    #colorMesh = sxx = spectrogram ndarray
    f, t, colorMesh = signal.spectrogram(file_data, fs=sr, window='hann')
    
    spectro_image = Image.fromarray(colorMesh, mode='RGB')
    spectro_image.save("songs/spectro_features/"+filename+"_spectro.png")
    
    features = Features(file_data, sr, colorMesh, file_name)
    
    song[file_name]["spectrogram_Hash"] = Hash(colorMesh)
    song[file_name]['melspectrogram_Hash'] = Hash(features[0])
    song[file_name]['mfcc_Hash'] = Hash(features[1])
    song[file_name]['chroma_stft_Hash'] = Hash(features[2])
    return song

def get_hamming(hash1, hash2):
    similarity = 1 - ( hex_to_hash(hash1) - hex_to_hash(hash2) )/256.0
    return similarity