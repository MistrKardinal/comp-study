import socket

def get_computer_specific_folder():
    computer_name = socket.gethostname().lower()
    
    folder_mapping = {
        'lvcovsky-ansys': 'C:/Users/kaprn/Desktop/MPS_directory/TensorNetworks',  # для компьютера с именем 'comp1'
        'desktop-2tu6s2s': 'C:/Users/Nik/Desktop/MPS/MPS_directory',  # для компьютера с именем 'comp2'
        'qrobot-server0': '/home/nkap/Tensor_networks'   # для компьютера с именем 'comp3'
    }
    
    return folder_mapping.get(computer_name, '/')